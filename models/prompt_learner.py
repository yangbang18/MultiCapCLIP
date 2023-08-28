import torch.nn as nn
from models.basic import BasicContainer


class NaivePromptLeaner(BasicContainer):
    def __init__(self, embed_dim, hidden_size, hidden_dropout_prob=0.1) -> None:
        super().__init__()
        self.net = nn.Linear(embed_dim, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.apply(self._init_weights)
    
    def forward(self, embs):
        out = self.net(embs)
        out = self.LayerNorm(out)
        out = self.dropout(out)
        if out.dim() == 2:
            out = out.unsqueeze(1)
        return out


class ConceptPromptPrefixer(BasicContainer):
    def __init__(self, embed_dim, hidden_size, n_concept_prompts, hidden_dropout_prob=0.1) -> None:
        super().__init__()
        self.n_concept_prompts = n_concept_prompts
        self.net = nn.Linear(embed_dim, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.apply(self._init_weights)
    
    def forward(self, concept_prompts):
        assert concept_prompts.size(1) == self.n_concept_prompts
        return self.dropout(self.LayerNorm(self.net(concept_prompts)))


def build_prompt_learner(config: dict, embed_dim: int, hidden_size: int):
    if config.get('add_cross_attention'):
        if config.get('ConceptPromptPrefixer'):
            print(f'### Using ConceptPromptPrefixer, with n_concept_prompts = {config["n_concept_prompts"]}')
            return ConceptPromptPrefixer(embed_dim, hidden_size, config['n_concept_prompts'])

        return None
    else:
        print('### Using only the CLIP\'s emb to prompt the decoder')
        return NaivePromptLeaner(embed_dim, hidden_size)
