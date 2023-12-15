import torch
import os

from typing import Union
from src.utils import load_model
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from src.data import Shakespeare, get_shakespeare_info
from src.llm import (
    AttentionIsAllYouNeed,
)
from tatooine.ml.train import train_model


def ddp_setup(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12358"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def training_thread(rank: Union[int, str], world_size: int, params: dict):
    if world_size != 0:
        ddp_setup(rank, world_size)

    tokens, vocab_size, _ = get_shakespeare_info()

    n = int(0.9 * tokens.shape[0])

    train = Shakespeare(tokens[:n], params["block_size"])
    valid = Shakespeare(tokens[n:], params["block_size"])

    train_loader = DataLoader(
        train,
        batch_size=params["batch_size"],
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(train) if world_size != 0 else None,
    )
    valid_loader = DataLoader(
        valid,
        batch_size=params["batch_size"],
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(valid) if world_size != 0 else None,
    )

    model = AttentionIsAllYouNeed(
        vocab_size,
        params["block_size"],
        params["n_embd"],
        params["n_heads"],
        params["n_layers"],
        rank,
    )

    if params["model_name"] is not None:
        load_model(model, rank, params["model_name"])

    train_model(
        model,
        train_loader,
        valid_loader,
        torch.optim.Adam(model.parameters(), lr=params["learning_rate"]),
        rank,
        n_eval_iters=params["n_eval_iters"],
        n_updates=params["n_updates"],
        checkpoint_iter=params["checkpoint_iter"],
        model_name=params["model_name"]
        if params["model_name"] is not None
        else "optimal.model",
    )

    if world_size != 0:
        destroy_process_group()
