import torch.nn as nn
import torch


class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T  # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


class SelfAttention_v2(nn.Module):
    """Self-attention mechanism with learnable parameters where the parameters are initialized
    with random values following an initialization scheme."""

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=1)
        context_vec = attn_weights @ values
        return context_vec


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)  # A

        # Create a mask to prevent the model from attending to future tokens. The buffer is used to store the mask tensor and is not fineutated during training.
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )  # B

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # C New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)  # C
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, block_size, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = (
            d_out // num_heads
        )  # Head dimension is d_out divided by the number of heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(
            d_out, d_out
        )  # Use a Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(block_size, block_size), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)  # (b, num_tokens, d_out)
        queries = self.W_query(x)  # (b, num_tokens, d_out)
        values = self.W_value(x)  # (b, num_tokens, d_out)
        keys = keys.view(
            b, num_tokens, self.num_heads, self.head_dim
        )  #  (b, num_tokens, num_heads, head_dim)
        values = values.view(
            b, num_tokens, self.num_heads, self.head_dim
        )  #  (b, num_tokens, num_heads, head_dim)
        queries = queries.view(
            b, num_tokens, self.num_heads, self.head_dim
        )  #  (b, num_tokens, num_heads, head_dim)
        keys = keys.transpose(1, 2)  #  (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)  #  (b, num_heads, num_tokens, head_dim)
        values = values.transpose(1, 2)  #  (b, num_heads, num_tokens, head_dim)
        attn_scores = (
            queries @ keys.transpose(2, 3)
        )  # compute dot product for each head -- pay attention that the token embeddings are multiplied for each head separately

        mask_bool = self.mask.bool()[
            :num_tokens, :num_tokens
        ]  # mask for the attention scores

        attn_scores.masked_fill_(mask_bool, -torch.inf)  # apply the mask
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(
            1, 2
        )  # (b, num_heads, num_tokens, head_dim)

        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )  # (b, num_tokens, d_out) -- reshape the output back to the original dimensions

        context_vec = self.out_proj(
            context_vec
        )  # linear projection of the concatenated heads' embeddings

        return context_vec
