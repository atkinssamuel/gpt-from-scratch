import torch
import torch.nn as nn

from torch.nn import functional as F

torch.manual_seed(1337)

# batch, block size (# of tokens, AKA time), embedding dim (# channels)
B, T, C = 1, 8, 4

x = torch.randn(B, T, C)

head_size = 16

key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)
q = query(x)
v = value(x)

weights = q @ k.transpose(-2, -1)

tril = torch.tril(torch.ones(T, T))
weights = torch.zeros(T, T)
weights = weights.masked_fill(tril == 0, float("-inf"))

print(v)

print(weights)

weights = F.softmax(weights, dim=-1)

print(weights)

print(weights @ v)
