import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basic import BasicContainer


class MLP(BasicContainer):
    def __init__(self, hidden_size, mlp_ratio=2):
        super().__init__()
        intermediate_size = int(hidden_size * mlp_ratio)

        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.LayerNorm(intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size)
        )
        self.apply(self._init_weights)
    
    def forward(self, feats):
        return self.net(feats)


class NaiveProjecter(BasicContainer):
    def __init__(self, embed_dim, hidden_size, hidden_dropout_prob=0.1, average_pooling=False) -> None:
        super().__init__()
        self.net = nn.Linear(embed_dim, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.average_pooling = average_pooling
        self.apply(self._init_weights)
    
    def _preprocess(self, embs):
        if embs.dim() == 2:
            embs = embs.unsqueeze(1)
        if self.average_pooling:
            embs = embs.mean(1, keepdims=True)
        return embs

    def _forward(self, embs, *args, **kwargs):
        out = self.net(embs)
        out = self.LayerNorm(out)
        out = self.dropout(out)
        return (out, )

    def forward(self, embs, *args, **kwargs):
        out = self._preprocess(embs)
        out = self._forward(out, *args, **kwargs)
        return out


class ProjecterWithConceptPrompts(NaiveProjecter):
    def __init__(self, embed_dim, hidden_size, hidden_dropout_prob=0.1, average_pooling=False, n_concept_prompts=8, n_virtual_prompts=0, exclude=False) -> None:
        super().__init__(embed_dim, hidden_size, hidden_dropout_prob, average_pooling)
        self.n_concept_prompts = n_concept_prompts
        print(f'n_concept_prompts = {n_concept_prompts}, n_virtual_prompts = {n_virtual_prompts} in the projector')
        if n_virtual_prompts > 0:
            self.virtual_prompts = nn.Parameter(torch.FloatTensor(1, n_virtual_prompts, embed_dim))
            nn.init.trunc_normal_(self.virtual_prompts, std=0.02)
        self.exclude = exclude

    def _forward(self, embs, emb_type=None, only_with_prompts=False, index=None, concept_prompts=None, **kwargs):
        assert emb_type is not None, "Please pass `emb_type` (\"image\" or \"text\") to the projector"
        assert emb_type in ['image', 'text']
        
        # note that concept_prompts has been l2-normalized
        assert concept_prompts is not None

        B, T, D = embs.shape
        embs = embs.view(B * T, D)
        db = concept_prompts.to(embs.device)
        sims = embs @ db.t() # (B * T, dict_size)
        sims = sims.view(B, T, -1).mean(1)
        #sims = sims.view(B, T, -1).max(1)[0]
        
        _, retrieval_ids = sims.topk(self.n_concept_prompts, dim=1, sorted=True, largest=True)
        prompts = db.unsqueeze(0).repeat(B, 1, 1).gather(dim=1, index=retrieval_ids.unsqueeze(2).repeat(1, 1, D))
        embs = embs.view(B, T, D)

        if hasattr(self, 'virtual_prompts'):
            virtual_prompts = F.normalize(self.virtual_prompts, dim=-1).repeat(B, 1, 1)
            prompts = torch.cat((prompts, virtual_prompts), dim=1)

        if self.exclude:
            out = embs
        else:
            if only_with_prompts:
                if index is None:
                    out = prompts
                else:
                    out = prompts[:, index:index+1, :]
            else:
                out = torch.cat((embs, prompts), dim=1) # (B, T + prompt_length, D)

        out = self.net(out)
        out = self.LayerNorm(out)
        out = self.dropout(out)

        return (out, prompts, retrieval_ids)


def build_clip_projecter(config: dict, embed_dim: int, hidden_size: int):
    if config.get('add_cross_attention'):
        average_pooling = config.get('average_pooling', False)
        
        if config.get('concepts'):
            return ProjecterWithConceptPrompts(
                embed_dim, hidden_size,
                average_pooling=average_pooling,
                n_concept_prompts=config['n_concept_prompts'],
                n_virtual_prompts=config.get('n_virtual_prompts', 0),
                exclude=config.get('exclude_from_visual_features', False),
            )

        return NaiveProjecter(embed_dim, hidden_size, average_pooling=average_pooling)
    
    return None
