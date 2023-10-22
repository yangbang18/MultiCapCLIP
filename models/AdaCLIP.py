import torch
import torch.nn as nn
import torch.nn.functional as F

import clip
import configs
import numpy as np
import transformers

from transformers import BertConfig, BertTokenizer
from PIL import Image

from models.language_model import LanguageModel, LanguageModelWithCrossAttention
from models.prompt_learner import build_prompt_learner
from models.clip_projecter import build_clip_projecter


class AdaCLIP(nn.Module):
    def __init__(self, config, adapt=True, only_keep_visual=False) -> None:
        super().__init__()

        self.adapt = adapt
        self.max_tokens = config['max_tokens']
        self.normalize = config.get('normalize', False)
        self.use_all_text_tokens = config.get('use_all_text_tokens', False)

        print(f'Use all text tokens: {self.use_all_text_tokens}')

        # =============== CLIP ===============
        model, preprocess = clip.load(
            config['clip_arch'], 
            device='cpu', 
            jit=False, 
            download_root=config['clip_root'],
        )

        self.clip = model.float()
        self.preprocess = preprocess
        embed_dim = self.clip.text_projection.size(-1)
        if config.get('i3d', False):
            embed_dim = 1024

        self.freeze_visual_backbone = config.get('freeze_visual_backbone', True)
        
        if only_keep_visual or not self.freeze_visual_backbone:
            # delete unnecessary components
            self.clip.transformer = None
            self.clip.token_embedding = None
            self.clip.positional_embedding = None
            self.clip.ln_final = None
            self.clip.text_projection = None
            self.clip.logit_scale = None

        # freeze CLIP
        for n, p in self.clip.named_parameters():
            if 'visual' in n and not self.freeze_visual_backbone:
                continue
            p.requires_grad = False
        
        if config.get('multilingual_clip', False):
            from sentence_transformers import SentenceTransformer
            assert config['clip_arch'] == 'ViT-B/32', config['clip_arch']
            self.text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1', cache_folder=config['clip_root'])
            for p in self.text_model.parameters():
                p.requires_grad = False
        
        self.init_params = [] # apply multiple learning rate on these params (see utils.optim)

        # ============= Decoder ============
        TOKENIZER = BertTokenizer
        CONFIG = BertConfig
        DECODER_MODULE = LanguageModelWithCrossAttention \
            if config.get('add_cross_attention') else LanguageModel

        self.tokenizer = TOKENIZER.from_pretrained(config['text_model'])
        print('Vocab size:', self.tokenizer.vocab_size)

        decoder_config = CONFIG.from_json_file(config['decoder_config'])
        decoder_config.update({'vocab_size': self.tokenizer.vocab_size}) # override the vocab_size
        if config.get('num_hidden_layers'):
            decoder_config.update({'num_hidden_layers': config['num_hidden_layers']})
        if config.get('multilingual'):
            print('type_vocab_size is set to', configs.max_languages) 
            decoder_config.update({'type_vocab_size': configs.max_languages})
        
        decoder_config.update({'prompt_within_layers': config.get('prompt_within_layers', None)})

        if config.get('init_with_bert', False):
            print('### Loading pre-trained weights from', config['text_model'])
            self.decoder, msg = DECODER_MODULE.from_pretrained(
                config['text_model'],
                config['label_smoothing'], 
                config=decoder_config, 
                output_loading_info=True
            )
            
            for k, v in msg.items():
                print(k, v)
        else:
            print('### Randomly initialize the decoder')
            self.decoder = DECODER_MODULE(decoder_config, config['label_smoothing'])
            self.init_params.extend(['decoder.' + n for n, _ in self.decoder.named_parameters()])

        if config.get('tie_encoder_decoder', False):
            assert hasattr(self, 'text_model')
            pretrained_embs = self.text_model[0].auto_model.get_input_embeddings()
            self.decoder._tie_or_clone_weights(self.decoder.get_input_embeddings(), pretrained_embs)
            self.decoder._tie_or_clone_weights(self.decoder.get_output_embeddings(), pretrained_embs)

        # ======== CLIP Projecter ==========
        self.clip_projecter = build_clip_projecter(config, embed_dim, decoder_config.hidden_size)
        if self.clip_projecter is not None:
            self.init_params.extend(['clip_projecter.' + n for n, _ in self.clip_projecter.named_parameters()])
            if config.get('freeze_projector', False):
                print('### Freeze projector')
                for n, p in self.clip_projecter.named_parameters():
                    p.requires_grad = False

        # ======== Prompt Learner ==========
        self.prefixer_flag = config.get('ConceptPromptPrefixer', False)
        self.prompt_learner = build_prompt_learner(config, embed_dim, decoder_config.hidden_size)
        if self.prompt_learner is not None:
            self.init_params.extend(['prompt_learner.' + n for n, _ in self.prompt_learner.named_parameters()])

        # ==== Manual Prompt (Optional) ====
        # only take effect during inference
        self.prompt = config.get('prompt', '')

        # ==== Structual Noise ====
        self.noise_std = config.get('noise_std', 0)
        print('### noise std:', self.noise_std)

        # ==== Concept Prompts ====
        self.concepts = config.get('concepts', False)
        self.concepts_path = config.get('concepts_path', None)
        self.templates = config.get('templates', ['{}'])
        self.concept_prompts = None
        if self.concepts:
            print('### templates:', self.templates)

        # ==== Context Tokens for Prompting CLIP's Text Encoder ====
        self.n_clip_prompts = config.get('n_clip_prompts', 0)
        if self.n_clip_prompts:
            print('### Prompting CLIP\'s text encoder with prompt size', self.n_clip_prompts)
            transformer_width = self.clip.ln_final.weight.shape[0]
            self.clip_prompts = nn.Parameter(torch.FloatTensor(1, self.n_clip_prompts, transformer_width))
            nn.init.trunc_normal_(self.clip_prompts, std=0.02)
    
    def prepare_concept_prompts(self, concepts_path, topk=1000, batch_size=128):
        if concepts_path is None:
            return None
        
        print(f'Preparing concept prompts! path: {concepts_path}, topk: {topk}')
        
        concepts = open(concepts_path, 'r').read().strip().split('\n')[:topk]
        self.raw_concepts = np.array(concepts)

        templates = self.templates

        concept_text = []

        for template in templates:
            concept_text.extend([template.format(c) for c in concepts])

        tokenized_ids = clip.tokenize(concept_text, truncate=True)

        concept_prompts = []
        start = 0
        while True:
            end = start + batch_size
            text_embs, *_ = self.encode_text(tokenized_ids[start:end].to(self.device))
            text_embs = F.normalize(text_embs, dim=-1)
            concept_prompts.append(text_embs)
            
            start += batch_size
            if start >= len(tokenized_ids):
                break
        
        concept_prompts = torch.cat(concept_prompts, dim=0)
        concept_prompts = torch.stack(concept_prompts.chunk(len(templates)), dim=1)
        concept_prompts = torch.mean(concept_prompts, dim=1) # ensembling

        return concept_prompts
    
    @property
    def device(self):
        return next(self.parameters()).device

    def load_pretrained_state_dict(self, state_dict):
        msg = self.load_state_dict(state_dict, strict=False)
         
         # CLIP's weights may be not saved to save disk space, so we ignore their warning info
        missing_keys = [k for k in msg.missing_keys if not k.startswith('clip.')]

        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", msg.unexpected_keys)
        
        # assert len(msg.unexpected_keys) == 0, msg.unexpected_keys
        # for key in msg.missing_keys:
        #     assert key.startswith('clip.')
        return msg
    
    def encode_image(self, image):
        assert hasattr(self, 'clip')
        return self.clip.encode_image(image)
    
    def encode_text(self, text):
        assert hasattr(self, 'clip')
        attn_mask = None

        if self.use_all_text_tokens:
            x = self.clip.token_embedding(text).type(self.clip.dtype)  # [batch_size, n_ctx, d_model]

            x = x + self.clip.positional_embedding.type(self.clip.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.clip.ln_final(x).type(self.clip.dtype)

            x = x @ self.clip.text_projection
            # L == 77 here, which is too long, so we truncate x
            eos_positions = text.argmax(dim=-1)
            max_length = eos_positions.max()
            x = x[:, :max_length, :]
            
            attn_mask = text[:, :max_length].gt(0).float()
            return x, attn_mask
        
        if self.n_clip_prompts:
            clip_prompts = self.clip_prompts.repeat(text.size(0), 1, 1) # [batch_size, n_clip_prompts, d_model]
            x = self.clip.token_embedding(text).type(self.clip.dtype)  # [batch_size, n_ctx, d_model]
            x = torch.cat((clip_prompts, x[:, :-self.n_clip_prompts, :]), dim=1) # [batch_size, 77, d_model]


            x = x + self.clip.positional_embedding.type(self.clip.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.clip.ln_final(x).type(self.clip.dtype)

            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1) + self.n_clip_prompts] @ self.clip.text_projection
            return x, attn_mask

        return self.clip.encode_text(text), attn_mask
    
    def _get_clip_image_embs(self, raw_image=None, image=None, clip_image_embs=None):
        device = self.device

        if clip_image_embs is None:
            if image is None:
                assert raw_image is not None, 'Please specify either raw_image or image'
                
                if not isinstance(raw_image, (tuple, list)):
                    raw_image = [raw_image]
                
                image = []
                for img in raw_image:
                    if type(img) is str:
                        img = Image.open(img).convert('RGB')
                    image.append(self.preprocess(img))
                image = torch.stack(image, dim=0)

            if image.dim() == 5:
                # video, (B, T, C, H, W)
                B, T, C, H, W = image.shape
                image = image.view(B * T, C, H, W)
                clip_image_embs = self.encode_image(image.to(device))
                clip_image_embs = clip_image_embs.view(B, T, -1)
            else:
                clip_image_embs = self.encode_image(image.to(device))
        else:
            clip_image_embs = clip_image_embs.to(device)

        if self.normalize:
            clip_image_embs = F.normalize(clip_image_embs, dim=-1)

        return clip_image_embs
    
    @torch.no_grad()
    def get_clip_image_embs(self, raw_image=None, image=None, clip_image_embs=None):
        self.clip.eval()
        return self._get_clip_image_embs(raw_image, image, clip_image_embs)
    
    @torch.no_grad()
    def get_clip_text_embs(self, raw_text=None, text=None, clip_text_embs=None):
        device = self.device
        self.clip.eval()

        attn_mask = None

        if hasattr(self, 'text_model'):
            assert raw_text is not None
            assert clip_text_embs is None
            clip_text_embs = self.text_model.encode(raw_text, convert_to_tensor=True)
        else:
            if clip_text_embs is None:
                if text is None:
                    assert raw_text is not None, 'Please specify either raw_text or text'
                    text = clip.tokenize(raw_text, truncate=True)
                
                clip_text_embs, attn_mask = self.encode_text(text.to(device))
            else:
                clip_text_embs = clip_text_embs.to(device)

        if self.normalize:
            clip_text_embs = F.normalize(clip_text_embs, dim=-1)
        
        if self.training and self.noise_std > 0:
            assert self.normalize
            clip_text_embs = clip_text_embs + (torch.randn(clip_text_embs.shape).to(self.device) * self.noise_std)
            clip_text_embs = F.normalize(clip_text_embs, dim=-1)
        
        return clip_text_embs, attn_mask

    def forward(self, raw_image=None, image=None, clip_image_embs=None, 
                      raw_text=None, text=None, clip_text_embs=None, 
                      raw_related_text=None, lang=None,
                      related_attn_mask=None):

        if self.concepts and (self.n_clip_prompts or self.concept_prompts is None):
            self.concept_prompts = self.prepare_concept_prompts(self.concepts_path)
        
        if self.freeze_visual_backbone:
            clip_image_embs = self.get_clip_image_embs(raw_image, image, clip_image_embs) if not self.adapt else None
        else:
            assert not self.adapt
            clip_image_embs = self._get_clip_image_embs(raw_image, image, clip_image_embs)
        
        # here, raw_related_text == raw_text if `with_related_caption_as_input` is False
        clip_text_embs, encoder_attention_mask = self.get_clip_text_embs(raw_related_text, text, clip_text_embs) if self.adapt else (None, None)
        if related_attn_mask is not None:
            encoder_attention_mask = related_attn_mask.to(self.device)

        prompt_feats, encoder_hidden_states = None, None

        if self.clip_projecter is not None:
            encoder_hidden_states, *other_outputs = self.clip_projecter(
                clip_text_embs if self.adapt else clip_image_embs, 
                'text' if self.adapt else 'image',
                concept_prompts=self.concept_prompts,
            )

        if self.prompt_learner is not None:
            if self.prefixer_flag:
                assert len(other_outputs) >= 1
                prompt_feats = self.prompt_learner(other_outputs[0])
            else:
                prompt_feats = self.prompt_learner(clip_text_embs if self.adapt else clip_image_embs)

        assert raw_text is not None

        text = self.tokenizer(
            raw_text, 
            padding='longest', 
            truncation=True, 
            max_length=self.max_tokens, 
            return_tensors="pt",
        ).to(self.device)

        # [CLS] + tokens + [SEP]
        # [CLS] as the begin-of-sentence token
        # [SEP] as the end-of-sentence token
        input_ids = text.input_ids
        attention_mask = text.attention_mask
        token_type_ids = lang.to(self.device) if lang is not None else None

        labels = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        loss = self.decoder(input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            prompt_feats=prompt_feats,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            labels=labels,
                            return_dict=True,
                            reduction='mean',
                            ).loss

        return loss
    
    def generate(self, image, clip_image_embs=None, lang=None, sample=False, num_beams=1, max_length=30, min_length=5, top_p=0.9,
                 repetition_penalty=1.0, num_return_sequences=1, greedy=False, only_with_prompts=False, index=None, DEBUG=False):
        
        if self.concepts and self.concept_prompts is None:
            self.concept_prompts = self.prepare_concept_prompts(self.concepts_path)
        
        clip_image_embs = self.get_clip_image_embs(image=image, clip_image_embs=clip_image_embs)

        batch_size = clip_image_embs.size(0)

        prompt = [self.prompt] * batch_size

        prompt_feats, encoder_hidden_states = None, None

        if self.clip_projecter is not None:
            encoder_hidden_states, *other_outputs = self.clip_projecter(clip_image_embs, 'image', only_with_prompts=only_with_prompts, index=index, concept_prompts=self.concept_prompts)
        
        if self.prompt_learner is not None:
            if self.prefixer_flag:
                assert len(other_outputs) >= 1
                prompt_feats = self.prompt_learner(other_outputs[0])

                if DEBUG:
                    assert batch_size == 1
                    retrieval_ids = other_outputs[1][0].cpu().tolist()
                    print(self.raw_concepts[retrieval_ids])
            else:
                prompt_feats = self.prompt_learner(clip_image_embs)

        if lang is None:
            lang = configs.main_lang # English
        if type(lang) in [str, int]:
            if type(lang) is str:
                lang = configs.lang2code[lang]
            token_type_ids = (clip_image_embs.new_zeros(batch_size, 1) + lang).long()
        else:
            token_type_ids = lang.to(self.device)

        if num_beams > 1:
            assert (sample is False) and (num_return_sequences == 1)
            if prompt_feats is not None:
                prompt_feats = self._repeat_wisely(prompt_feats, num_beams)
            if encoder_hidden_states is not None:
                encoder_hidden_states = self._repeat_wisely(encoder_hidden_states, num_beams)

        if num_return_sequences > 1:
            assert (sample is True) and (num_beams == 1)
            if prompt_feats is not None:
                prompt_feats = self._repeat_wisely(prompt_feats, num_beams)
            if encoder_hidden_states is not None:
                encoder_hidden_states = self._repeat_wisely(encoder_hidden_states, num_beams)
            prompt = [self.prompt] * (batch_size * num_return_sequences)

        model_kwargs = {
            "prompt_feats": prompt_feats, 
            "encoder_hidden_states": encoder_hidden_states,
            "token_type_ids": token_type_ids,
        }

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        # For BERT-like LMs, input_ids will be [CLS] if the manual prompt is empty
        input_ids = input_ids[:, :-1]

        def _get_captions(caption_ids):
            captions = []
            for i, output in enumerate(caption_ids):
                caption = self.tokenizer.decode(output, skip_special_tokens=True)
                captions.append(caption[len(self.prompt):])
            return captions

        if greedy:
            # greedy generation from OSCAR
            assert (num_beams == 1) and (num_return_sequences == 1)
            outputs, logprobs = self.decoder._generate_no_beam_search(input_ids=input_ids, cur_len=input_ids.shape[1], max_length=max_length,
                                          do_sample=False, temperature=1,
                                          top_k=0, top_p=1, repetition_penalty=repetition_penalty,
                                          pad_token_id=self.tokenizer.pad_token_id, eos_token_ids=[self.tokenizer.sep_token_id],
                                          batch_size=batch_size, **model_kwargs)

            return _get_captions(outputs)

        elif sample:
            # sampling from OSCAR
            outputs, logprobs = self.decoder._generate_no_beam_search(input_ids=input_ids, cur_len=input_ids.shape[1], max_length=max_length,
                                          do_sample=True, temperature=1,
                                          top_k=0, top_p=1, repetition_penalty=repetition_penalty,
                                          pad_token_id=self.tokenizer.pad_token_id, eos_token_ids=[self.tokenizer.sep_token_id],
                                          batch_size=batch_size, **model_kwargs)

            # outputs: (bs x num_return_sequences, max_length)
            # logprobs: (bs x num_return_sequences,)

            return _get_captions(outputs), logprobs

        else:
            # beam search from huggingface
            outputs = self.decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)

            return _get_captions(outputs)

    def _repeat_wisely(self, tensor, num_beams, dim=0):
        if not hasattr(self, "should_repeat"):
            self.should_repeat = True
            version = [int(item) for item in transformers.__version__.split('.')]
            if version[0] > 4 or (version[0] == 4 and version[1] >=27):
                # after 4.27.0, we should not repeat encoder's outputs by `num_beams` * `num_return_sequences` times
                self.should_repeat = False
        
        if self.should_repeat:
            tensor = tensor.repeat_interleave(num_beams, dim=dim)
        
        return tensor
