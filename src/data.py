import os


_data_dir = "data/"


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
    return
