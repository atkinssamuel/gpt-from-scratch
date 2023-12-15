import torch

from src.data import Shakespeare, get_shakespeare_info
from src.params import params
from src.llm import (
    AttentionIsAllYouNeed,
)
from src.utils import load_model

if __name__ == "__main__":
    device = "cpu"  # 0 for GPU inference
    tokens, vocab_size, tokenizer = get_shakespeare_info()

    model = AttentionIsAllYouNeed(
        vocab_size,
        params["block_size"],
        params["n_embd"],
        params["n_heads"],
        params["n_layers"],
        device,
    )
    load_model(model, device, "optimal.model")

    start_token = (
        torch.tensor(tokenizer.encode(" "), dtype=torch.long).reshape(1, -1).to(device)
    )
    gen = model.generate(start_token, 5000)

    print(tokenizer.decode(gen.tolist()[0]))
