import os
import pickle
import torch

from tatooine.ml.text.tokenizers import CharacterTokenizer
from tatooine.ml.data import ModelData

_data_dir = "data/"


class Shakespeare(ModelData):
    def __init__(self, block_size: int, batch_size: int, device: str) -> None:
        if not os.path.exists(_data_dir + "shakespeare.txt"):
            raise Exception(
                "TextData file '" + _data_dir + "shakespeare.txt' does not exist."
            )

        with open(_data_dir + "shakespeare.txt", "r", encoding="utf-8") as f:
            self.text = f.read()

        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        chars = sorted(list(set(self.text)))

        self.vocab_size = len(chars)
        self.tokenizer = CharacterTokenizer(chars)

        data = torch.tensor(self.tokenizer.encode(self.text), dtype=torch.long)

        n = int(data.shape[0] * 0.9)

        self.train = data[:n]
        self.valid = data[n:]

    def get_batch(self, split: str = "train") -> list[torch.Tensor]:
        data = self.train if split == "train" else self.valid

        # starting indices for each batch (e.g. [231, 6542, 12, 6] for a batch size of 4)
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))

        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])

        x, y = x.to(self.device), y.to(self.device)

        return x, y
