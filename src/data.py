import os
import pickle
import torch

from typing import Tuple, List
from tatooine.ml.text.tokenizers import CharacterTokenizer
from torch.utils.data import Dataset

_data_dir = "data/"


def get_shakespeare_info() -> Tuple[torch.tensor, int]:
    if not os.path.exists(_data_dir + "shakespeare.txt"):
        raise Exception(
            "TextData file '" + _data_dir + "shakespeare.txt' does not exist."
        )

    with open(_data_dir + "shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    tokenizer = CharacterTokenizer(chars)

    return torch.tensor(tokenizer.encode(text), dtype=torch.long), vocab_size, tokenizer


class Shakespeare(Dataset):
    def __init__(
        self, tokens: torch.tensor, block_size: int
    ) -> Tuple[torch.tensor, torch.tensor]:
        self.block_size = block_size
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        return (
            self.tokens[idx : idx + self.block_size],
            self.tokens[idx + 1 : idx + self.block_size + 1],
        )
