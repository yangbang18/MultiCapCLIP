import os
import torch
import pickle
import random
import configs
import pandas as pd
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from typing import List


def get_related_caption_ids(text_embs: torch.FloatTensor, all_captions: list, topk: int=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_embs = F.normalize(text_embs.to(device), dim=-1)

    # check uniqueness of captions
    record = set()
    valid_ids = []
    valid_text_embs = []
    for i, caption in enumerate(all_captions):
        if caption not in record:
            valid_ids.append(i)
            valid_text_embs.append(text_embs[i])
        record.add(caption)

    valid_ids = torch.LongTensor(valid_ids)
    valid_text_embs = torch.stack(valid_text_embs, dim=0)

    all_ids = []
    for i in tqdm(range(len(text_embs)), desc='getting indexes of related captions'):
        ids = torch.cosine_similarity(text_embs[i:i+1], valid_text_embs).topk(topk)[1]
        ids = valid_ids[ids.cpu()].tolist()
        all_ids.append(ids)

    return all_ids


def save_related_caption_ids(all_ids: list, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    all_ids = [','.join([str(i) for i in item]) for item in all_ids]

    print('Save related captions ids to', save_path)
    with open(save_path, 'w') as f:
        f.write('\n'.join(all_ids))


class AdaptDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, 
                       num_samples: int=-1, 
                       with_related_caption_as_input: bool=False, 
                       related_caption_ids_path: str=None,
                       related_caption_topk: int=-1,
                       related_caption_prob: float=1.0,
                       num_related_caption: int=1,
                       src_lang: str=configs.main_lang,
                       auto_generate_related_caption_ids: bool=True,
                       train_langs: List[str]=[configs.main_lang],
                       train_en_prob: float=None,
                       hdf5_path: str=None,
                       parallel_text_paths: List[str]=None,
                       ):
        assert os.path.exists(path), path
        print("### load data from", path)
        
        self.file_format = os.path.basename(path).split('.')[-1]
        assert self.file_format in ['txt', 'tsv', 'pkl']

        if self.file_format == 'txt':
            self.data = open(path, 'r').read().strip().split('\n')
        elif self.file_format == 'tsv':
            self.data = pd.read_csv(path, sep='\t', lineterminator='\n')
        elif self.file_format == 'pkl':
            self.data = pickle.load(open(path, 'rb'))

        if hdf5_path:
            import h5py
            print(f'loading hdf5 file from {hdf5_path}')
            self.hdf5_data = h5py.File(hdf5_path, 'r')
            assert len(self.hdf5_data) == len(self.data), f'hdf5 data size: {len(self.hdf5_data)}, {self.file_format} data size: {len(self.data)}'
        else:
            self.hdf5_data = None

        if parallel_text_paths:
            print('### load parallel text from', parallel_text_paths)
            all_lang = []
            all_data = []

            for p in parallel_text_paths:
                assert p.endswith('.txt')
                lang = os.path.basename(p).split('_')[0]
                assert lang in configs.lang2code
                data = open(p, 'r').read().strip().split('\n')
                assert len(data) == len(self.data)
                all_lang.append(lang)
                all_data.append(data)

            parallel_data = []
            for i in range(len(self.data)):
                item = {}
                for lang, data in zip(all_lang, all_data):
                    item[lang] = data[i]
                parallel_data.append(item)

            self.parallel_data = parallel_data
        
        if num_samples == -1:
            print(f'### Use all {len(self.data)} samples')
        else:
            assert num_samples > 0
            print(f'### Use the first {num_samples} samples')
            self.data = self.data[:num_samples]
        
        if with_related_caption_as_input:
            assert related_caption_ids_path is not None

            if os.path.exists(related_caption_ids_path):
                # load indexes of related captions from the off-the-shelf file
                data = open(related_caption_ids_path, 'r').read().strip().split('\n')
                self.related_caption_ids = [
                    list(map(lambda x: int(x), item.split(','))) 
                    for item in data
                ]
            else:
                print('Fail to load related caption ids from', related_caption_ids_path)
                assert auto_generate_related_caption_ids
                assert self.file_format == 'pkl', 'To obtain indexes of related captions, you should use a pickle file that stores text embeddings'

                text_embs = torch.FloatTensor([item['text_emb'] for item in self.data]) # (n_captions, embed_dim)
                all_captions = [item.get('caption', item.get(configs.main_lang)) for item in self.data]

                self.related_caption_ids = get_related_caption_ids(text_embs, all_captions)
                save_related_caption_ids(self.related_caption_ids, related_caption_ids_path)

            assert len(self.related_caption_ids) == len(self.data), f'{len(self.related_caption_ids)} {len(self.data)}'
        
        self.with_related_caption_as_input = with_related_caption_as_input
        self.related_caption_topk = related_caption_topk
        self.related_caption_prob = related_caption_prob

        if type(num_related_caption) is int:
            self.num_related_caption = (num_related_caption, num_related_caption)
        else:
            assert type(num_related_caption) is list
            assert len(num_related_caption) == 2, num_related_caption
            self.num_related_caption = sorted(num_related_caption)
        
        assert self.num_related_caption[0] >= 1
        
        self.src_lang = src_lang
        if self.src_lang != configs.main_lang:
            print(f'Warning: you are using {src_lang} as the source language!')

        self.train_langs = train_langs
        self.train_en_prob = train_en_prob
        print('### train langs:', self.train_langs)
        if self.train_en_prob is not None:
            print('### train_en_prob:', self.train_en_prob)
    
    def get_line(self, index):
        if self.file_format == 'tsv':
            line = self.data.iloc[index].to_dict()
        elif self.file_format == 'txt':
            line = {'caption': self.data[index]}
        elif self.file_format == 'pkl':
            line = self.data[index]

        if 'caption' not in line:
            assert configs.main_lang in line
            line['caption'] = line[configs.main_lang]

        if self.hdf5_data is not None and 'text_emb' not in line:
            line['text_emb'] = np.asarray(self.hdf5_data[str(index)])
        
        if hasattr(self, 'parallel_data'):
            line.update(self.parallel_data[index])

        return line

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.get_line(index)

        out = {}
        if self.train_en_prob is None:
            lang = random.choice(self.train_langs)
        else:
            other_langs = [l for l in self.train_langs if l != 'en']
            if random.random() < self.train_en_prob:
                lang = 'en'
            else:
                lang = random.choice(other_langs)

        out['src_caption'] = line.get(self.src_lang, line['caption'])   # English caption as the CLIP's encoder input
        out['caption'] = line.get(lang, line['caption'])                    # Any caption as the target output
        if lang != 'en':
            assert lang in line
        out['lang'] = torch.LongTensor([configs.lang2code[lang]])
        
        if self.with_related_caption_as_input:

            n1, n2 = self.num_related_caption
            if n2 != 1:
                assert self.file_format == 'pkl', "only support a pickle file now"
                ids = self.related_caption_ids[index] if self.related_caption_topk == -1 \
                    else self.related_caption_ids[index][:self.related_caption_topk]
                n = random.randint(n1, n2)
                n = min(n, len(ids))
                n_pad = n2 - n

                # related_indexes = sorted(random.sample(ids, n))
                assert n > 1
                related_indexes = [ids[0]] + sorted(random.sample(ids, n - 1))

                text_embs = [torch.FloatTensor(self.data[related_index]['text_emb']) for related_index in related_indexes]
                text_embs = text_embs + [text_embs[-1]] * n_pad
                out['clip_text_embs'] = torch.stack(text_embs, dim=0)
                out['related_attn_mask'] = torch.LongTensor([1] * n + [0] * n_pad)
                return out

            if random.random() < self.related_caption_prob:
                if self.related_caption_topk == -1:
                    related_index = random.choice(self.related_caption_ids[index])
                else:
                    related_index = random.choice(self.related_caption_ids[index][:self.related_caption_topk])
                line = self.get_line(related_index) # override the original line
            out['src_caption'] = line['caption']

        # Note that the line below may come from related_index
        text_emb = line.get('text_emb', None)
        if text_emb is not None:
            text_emb = torch.FloatTensor(text_emb)
            out['clip_text_embs'] = text_emb

        return out
