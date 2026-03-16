import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ElmanSRN(nn.Module):
    def __init__(self, sizes: list[int] = [8, 20, 8]):
        pass
