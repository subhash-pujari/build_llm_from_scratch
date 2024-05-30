import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ConfigGPT:
    vocab_size: int
    context_length: int
    emb_dim: int
    n_layers: int
    n_heads: int
    drop_rate: float
    qkv_bias: bool


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_embed = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.layers = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.final = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        bacth_size, seq_len = in_idx.shape
        token_embeds = self.embed(in_idx)
        pos_embeds = self.pos_embed(torch.arange(seq_len, device=in_idx.device))
        x = token_embeds + pos_embeds
        x = self.dropout(x)
        x = self.layers(x)
        x = self.final_norm(x)
        logits = self.final(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, ep=1e-05) -> None:
        super().__init__()

    def forward(self, x):
        return x
