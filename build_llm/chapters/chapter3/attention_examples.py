import torch

from build_llm.gpt.attention import (
    CausalAttention,
    MultiHeadAttention,
    SelfAttention_v2,
)

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with
        [0.77, 0.25, 0.10],  # one
        [0.05, 0.80, 0.55],
    ]  # step
)


d_in = 3
d_out = 2

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))


queries = sa_v2.W_query(inputs)  # A
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=1)
print(attn_weights)
print("attn_weights : ", attn_weights)


context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print("mask : ", mask_simple)


masked_simple = attn_weights * mask_simple
print("masked_simple : ", masked_simple)


row_sums = masked_simple.sum(dim=1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print("masked_simple_norm : ", masked_simple_norm)


mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
print("mask triu: ", mask)

masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print("masked : ", masked)


scaled_dot_attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)
print("scaled_dot_attn_weights: ", scaled_dot_attn_weights)


torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)  # A
example = torch.ones(6, 6)  # B
print("dropout(example): ", dropout(example))


torch.manual_seed(123)
print("dropout(attn_weights): ", dropout(attn_weights))


batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)  # A


torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)


torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
