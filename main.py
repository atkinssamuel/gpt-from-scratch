import torch

from src.data import Shakespeare
from src.llm import (
    AttentionIsAllYouNeed,
)
from tatooine.ml.train import train_model


""" Parameters: """
n_updates = 0
block_size = 256
batch_size = 64
n_embd = 384
n_heads = 6
n_layers = 6
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
loaded_model = "optimal.model"
# loaded_model = None
""""""

data = Shakespeare(block_size, batch_size, device)
bm = AttentionIsAllYouNeed(
    data.vocab_size, block_size, n_embd, n_heads, n_layers, device
)
bm.to(device)

loaded_model = "models/" + loaded_model if loaded_model is not None else None
if loaded_model is not None:
    print("Loading model...")
    bm.load_state_dict(torch.load(loaded_model, map_location=torch.device(device)))

print(f"Training using {device}...")
train_model(
    bm, data, torch.optim.Adam(bm.parameters(), lr=learning_rate), n_updates=n_updates
)

start_token = (
    torch.tensor(data.tokenizer.encode(" "), dtype=torch.long).reshape(1, -1).to(device)
)
gen = bm.generate(start_token, 5000)

print(data.tokenizer.decode(gen.tolist()[0]))
