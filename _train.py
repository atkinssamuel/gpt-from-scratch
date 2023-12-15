from src.parallel import training_thread
from src.params import params


if __name__ == "__main__":
    training_thread("cpu", 0, params)
