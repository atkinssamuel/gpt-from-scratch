import os
import pickle
import torch

from tatooine.ml.text.tokenizers import CharacterTokenizer
from tatooine.ml.data import ModelData

_data_dir = "data/"


class TextData(ModelData):
    def __init__(self, filename: str) -> None:
        if not os.path.exists(_data_dir + filename):
            raise Exception(
                "TextData file '" + _data_dir + filename + "' does not exist."
            )

        with open(_data_dir + filename, "r", encoding="utf-8") as f:
            self.text = f.read()

        self.block_size = 8
        self.batch_size = 4

        chars = sorted(list(set(self.text)))

        self.vocab_size = len(chars)
        self.tokenizer = CharacterTokenizer(chars)

        data = torch.tensor(self.tokenizer.encode(self.text), dtype=torch.long)

        n = int(data.shape[0] * 0.9)

        self.train = data[:n]
        self.valid = data[n:]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_batch(
        self, batch_size: int = None, split: str = "train"
    ) -> list[torch.Tensor]:
        torch.manual_seed(1337)

        batch_size = self.batch_size if batch_size is None else batch_size
        data = self.train if split == "train" else self.valid

        # starting indices for each batch (e.g. [231, 6542, 12, 6] for a batch size of 4)
        ix = torch.randint(len(data) - self.block_size, (batch_size,))

        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])

        x, y = x.to(self.device), y.to(self.device)

        return x, y


def get_shakespeare():
    """
    This function parses and extracts the shakespeare data (if a parsed shakespeare data
    object does not already exist).
    """
    if not os.path.exists(_data_dir + "shakespeare.data"):
        parse_shakespeare()

    with open(_data_dir + "shakespeare.data", "rb") as file:
        return pickle.load(file)


def parse_shakespeare():
    """
    Recreates and saves a TextData object using the shakespeare corpus.
    """
    shakespeare = TextData("shakespeare.txt")

    with open(_data_dir + "shakespeare.data", "wb") as file:
        pickle.dump(shakespeare, file)

    return
