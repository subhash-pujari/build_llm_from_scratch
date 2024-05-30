import torch
from build_llm.core.transformer import TransformerBlock

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 10,  # Context length
    "emb_dim": 6,  # Embedding dimension
    "n_heads": 3,  # Number of attention heads
    "n_layers": 2,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False,  # Query-Key-Value bias
}

torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)
