import torch

from src.data import get_data, get_shakespeare_info
from src.params import params
from src.llm import (
    AttentionIsAllYouNeed,
)
from src.utils import load_model

if __name__ == "__main__":
    device = 0
    tokens, vocab_size, tokenizer = get_shakespeare_info()
    n = int(0.9 * tokens.shape[0])
    valid = get_data(tokens, "valid", n, params)

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
