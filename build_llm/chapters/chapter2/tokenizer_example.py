import tiktoken
import re
from build_llm.gpt.dataset import create_dataloader_v1
from build_llm.gpt.tokenizers import SimpleTokenizerV2


with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of character:", len(raw_text))
print(raw_text[:99])


text = "Hello, world. This, is a test."
result = re.split(r"(\s)", text)
print(result)


result = re.split(r"([,.]|\s)", text)
print(result)


result = [item for item in result if item.strip()]
print(result)


text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)


preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))


print(preprocessed[:30])


all_words = sorted(list(set(preprocessed)))
vocab_size = len(all_words)
print(vocab_size)


vocab = {token: integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i > 50:
        break


all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}


tokenizer = SimpleTokenizerV2(vocab)
ids = tokenizer.encode(
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."
)

text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."
# print(ids)

print(tokenizer.decode(tokenizer.encode(text)))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
)
integers = tokenizer.encode(text)
print(integers)

strings = tokenizer.decode(integers)
print(strings)

with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))

enc_sample = enc_text[50:]

context_size = 4  # A
x = enc_sample[:context_size]
y = enc_sample[1 : context_size + 1]
print(f"x: {x}")
print(f"y: {y}")

# encode the text into integers
for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)

# decode the integers back to text
for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))


with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)


max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)
