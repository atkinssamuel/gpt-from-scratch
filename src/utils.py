import torch.nn as nn
import torch

from typing import Union


def load_model(model: nn.Module, device: Union[str, int], filename: str = None):
    loaded_model = "models/" + filename if filename is not None else None
    if loaded_model is not None:
        print("Loading model...")
        model.load_state_dict(torch.load(loaded_model))
        return model.to(device)
    return model
