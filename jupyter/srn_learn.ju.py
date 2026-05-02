# %% [md]
"""
# Simple recursive network for sequence learning
"""

# %%
import sys
from pathlib import Path
import numpy as np

import torch


# %%
# import local packages
sys.path.append(str(Path().resolve().parents[0]))

from src.utils.logger import Logger, get_logger
from src.utils.args import get_args
from src.data.formater import PandasLoader, Format


# %%
args = get_args()


# %%
logger = Logger()
logger.info("logger set")


# %%
loader = PandasLoader(args["input"], args["format"])

data = loader.get()
print(data)
data = data.to_numpy().T
print(data.shape)
print(data)
data = np.expand_dims(data, axis=-1)
print(data)

print(data.shape)

unique_vals, encoded = np.unique(data.reshape(-1), return_inverse=True)

print(unique_vals)
print(encoded)
print(encoded.max())
print(unique_vals.shape)
encoded = encoded.reshape(data.shape)
print(encoded)


# %%


class SRN:
    def __init__(self):
        pass

    def forward(self):
        pass
