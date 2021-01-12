import numpy as np
import pandas as pd
from typing import List, Union, Dict, Optional, Tuple


def sample_excluding(n: int,
                     exclude: List[int]) -> int:
    x = np.random.randint(n)
    while x in exclude:
        x = np.random.randint(n)
    return x


def sequential_tt_split(n: int,
                        n_train: int) -> (np.array, np.array):
    indices = np.arange(n)
    n_test = n - n_train
    train_idx = indices[:n_train]
    test_idx = indices[-n_test:]    
    return train_idx, test_idx
