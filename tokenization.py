import torch
import torch.nn as nn

from torch.nn import functional as F
from tatooine.ml.text.tokenizers import CharacterTokenizer
from src.data import Shakespeare

torch.manual_seed(1337)

_data_dir = "data/"

txt = "First Citizen:\nYou are all resolved rather to die than to famish?"

with open(_data_dir + "shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharacterTokenizer(sorted(list(set(text))))
encoded_data = torch.tensor(tokenizer.encode(txt), dtype=torch.long)

shakespeare = Shakespeare(block_size=8, batch_size=4, device="cpu")
X, y = shakespeare.get_batch()


print(f"\n\nRaw text: {txt}")
print(f"\n\nEncoded text: {encoded_data}")
print(f"\n\nX: {X}, \n\ny: {y}")

"""
Raw text: 

First Citizen:
You are all resolved rather to die than to famish?

Encoded text: tensor([18, 47, 56, 57, 58,  1, 15, 47, 58, 47, 64, 43, 52, 10,  0, 37, 53, 59,
         1, 39, 56, 43,  1, 39, 50, 50,  1, 56, 43, 57, 53, 50, 60, 43, 42,  1,
        56, 39, 58, 46, 43, 56,  1, 58, 53,  1, 42, 47, 43,  1, 58, 46, 39, 52,
         1, 58, 53,  1, 44, 39, 51, 47, 57, 46, 12])


X: tensor([[24, 43, 58,  5, 57,  1, 46, 43],
           [44, 53, 56,  1, 58, 46, 39, 58],
           [52, 58,  1, 58, 46, 39, 58,  1],
           [25, 17, 27, 10,  0, 21,  1, 54]]), 

y: tensor([[43, 58,  5, 57,  1, 46, 43, 39],
           [53, 56,  1, 58, 46, 39, 58,  1],
           [58,  1, 58, 46, 39, 58,  1, 46],
           [17, 27, 10,  0, 21,  1, 54, 39]])

n_embd: 16

X[0]: [24, 43, 58,  5, 57,  1, 46, 43]

    24 -> [0.32, 0.1, ...,  0.54] x 16

      [ 0,  1,  2,  3,  4,  5,  6,  7]

    0  -> [0.12, 0.43, ..., 0.89] x 16

y[0]: [43, 58,  5, 57,  1, 46, 43, 39]

ll him, and we'l
l him, and we'll

ld us but the su
eld us but the s
"""
