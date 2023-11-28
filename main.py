import torch

from src.data import Shakespeare
from tatooine.ml.text.models import (
    BigramLanguageModel,
    AttentionIsAllYouNeed,
)
from tatooine.ml.train import train_model


""" Parameters: """
block_size = 8
batch_size = 32
n_embd = 32
n_heads = 4
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

data = Shakespeare(block_size, batch_size, device)

vocab_size = data.vocab_size

bm = AttentionIsAllYouNeed(vocab_size, block_size, n_embd, n_heads, device)

# bm = bm.to(data.params.device)

# gen = bm.generate(
#     torch.tensor(data.tokenizer.encode(" "), dtype=torch.long).reshape(1, -1), 100
# )

# print(data.tokenizer.decode(gen.tolist()[0]))

optimizer = torch.optim.Adam(bm.parameters(), lr=learning_rate)

train_model(bm, data, optimizer)

gen = bm.generate(
    torch.tensor(data.tokenizer.encode(" "), dtype=torch.long).reshape(1, -1), 10000
)
print(data.tokenizer.decode(gen.tolist()[0]))
