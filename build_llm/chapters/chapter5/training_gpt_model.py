import tiktoken
import torch

from build_llm.core.dataset import create_dataloader_v1
from build_llm.core.gpt import GPTModel, generate_text_simple
from build_llm.core.loss import calc_loss_loader
from build_llm.core.utils import (
    plot_losses,
    text_to_token_ids,
    token_ids_to_text,
    train_model_simple,
)


GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 10,  # Context length
    "emb_dim": 6,  # Embedding dimension
    "n_heads": 3,  # Number of attention heads
    "n_layers": 2,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False,  # Query-Key-Value bias
}


file_path = "data/the-verdict.txt"


with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()


tokenizer = tiktoken.get_encoding("gpt2")


total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)


train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]


torch.manual_seed(123)


train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
)


print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)


print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)


model = GPTModel(GPT_CONFIG_124M)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # A
model.to(device)
train_loss = calc_loss_loader(train_loader, model, device)  # B
val_loss = calc_loss_loader(val_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)


# torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)  # A

num_epochs = 2

train_losses, val_losses, tokens_seen = train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=1,
    start_context="Every effort moves you",
)


epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


model.to("cpu")
model.eval()

"""
Generate text using the trained model with a simple strategy.
"""


tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"],
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


"""
Save and load the model.
"""

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    "output/model_and_optimizer.pth",
)


checkpoint = torch.load("output/model_and_optimizer.pth")
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train()
