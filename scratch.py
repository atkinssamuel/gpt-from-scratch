import torch

from src.data import get_shakespeare
from tatooine.ml.text.models import BigramLanguageModel
from tatooine.ml.train import train_model


data = get_shakespeare()

bm = BigramLanguageModel(data.vocab_size)
bm = bm.to(data.device)

gen = bm.generate(
    torch.tensor(data.tokenizer.encode(" "), dtype=torch.long).reshape(1, -1), 100
)

print(data.tokenizer.decode(gen.tolist()[0]))

optimizer = torch.optim.Adam(bm.parameters(), lr=1e-3)

train_model(bm, data, optimizer)

gen = bm.generate(
    torch.tensor(data.tokenizer.encode(" "), dtype=torch.long).reshape(1, -1), 100
)
print(data.tokenizer.decode(gen.tolist()[0]))
