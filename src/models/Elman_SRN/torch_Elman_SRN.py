import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


class ElmanSRN(nn.Module):
    def __init__(self, sizes: list[int] = [8, 20, 8]):
        pass

    def load_numpy(self, ndarray: np.ndarray):
        ndarray = np.array([ord(c) if isinstance(c, str) else c for c in ndarray])
        training_data = torch.from_numpy(ndarray)
        print(training_data)
