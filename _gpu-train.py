import torch.multiprocessing as mp
import torch

from src.parallel import training_thread
from src.params import params


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(training_thread, args=(world_size, params), nprocs=world_size)
