import tiktoken
import torch

from build_llm.core.gpt import GPTModel, generate_text_simple
from build_llm.core.utils import text_to_token_ids, token_ids_to_text

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
model = GPTModel(GPT_CONFIG_124M)
model.eval()

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"],
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


inputs = torch.tensor(
    [
        [16833, 3626, 6100],  # ["every effort moves",
        [40, 1107, 588],
    ]
)  #  "I really like"]

targets = torch.tensor(
    [
        [3626, 6100, 345],  # [" effort moves you",
        [588, 428, 11311],
    ]
)  #  " really like chocolate"]

with torch.no_grad():  # A
    logits = model(inputs)

probas = torch.softmax(logits, dim=-1)  # Probability of each token in vocabulary

print(probas.shape)

token_ids = torch.argmax(probas, dim=-1, keepdim=True)

print("Token IDs:\n", token_ids)

print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")


text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)


avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)


neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)


print("Logits shape:", logits.shape)
print("Targets shape:", targets.shape)


logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()

print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)

print(loss)
