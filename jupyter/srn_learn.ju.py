# %% [md]
"""
# Simple recursive network for sequence learning
"""

# %%
import sys
from pathlib import Path

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
test = data.to_numpy().T
print(test.shape)
print(test)
