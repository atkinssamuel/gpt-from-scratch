import torch

from src.data import get_shakespeare
from tatooine.data.tabular import DataBank
from tatooine.ml.text.tokenizers import CharacterTokenizer

db = DataBank()


with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(len(text))
print(text[:1000])


chars = sorted(list(set(text)))

vocab_size = len(chars)

print("".join(chars))
print(vocab_size)


char_tokenizer = CharacterTokenizer(chars)

data = torch.tensor(char_tokenizer.encode(text), dtype=torch.long)

print(data.shape, data.dtype)


n = int(data.shape[0] * 0.9)

train = data[:n]
valid = data[n:]

block_size = 8

print(train[: block_size + 1])

print(get_shakespeare())
