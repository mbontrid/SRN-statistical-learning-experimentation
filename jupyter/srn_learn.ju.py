# %% [md]
"""
# Simple recursive network for sequence learning
"""

# %%
import sys
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike
import torch
from typing import Callable, Sequence
from tqdm import trange


# import local packages
sys.path.append(str(Path().resolve().parents[0]))

from src.utils.logger import setup_logger, add_log_level
from src.utils.args import get_args
from src.data.formater import PandasLoader

# %%
args = get_args()
logger_level = "TRACE"

# %%
add_log_level("TRACE", 5)
logger = setup_logger(name="logger", level=logger_level)
logger.info("logger set to info level")
logger.trace("TRACE logger activated")


# %%
loader = PandasLoader(args["input"], args["format"])
responses = loader.get()
logger.info(f"subjects response per trial: \n {responses}")

# converting to array of shape (subjects, responses)
responses = responses.to_numpy().T
logger.debug(f"responses after transposition= {responses}")

# get set of responses values and their order of appearance(giving them a numerical id) restoring the value from it's encoded index is easy.
unique_vals, encoded = np.unique(responses.reshape(-1), return_inverse=True)
logger.info(f"unique values (set)={unique_vals}")
logger.info(f"encoded values={encoded}")
encoded = encoded.reshape(responses.shape)
logger.debug(f"final encoded shape (subjests, responses)={encoded.shape}")


# %%
def make_uniform_tensor(
    extremum: tuple[float, float], shape: Sequence[int], grad: bool
):
    t = torch.empty(*shape)
    t.uniform_(*extremum)
    if grad:
        t.requires_grad_()
    return t


# %% [md]
"""
![test](https://web.stanford.edu/group/pdplab/pdphandbook/srn_net.png)

A SRN is a simplified RNN. The output of the hidden layer is fed back as input to the hidden layer at the
next time step. The output of the hidden layer is also used to compute the output of the network.

![SRN basic architecture](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.researchgate.net%2Fpublication%2F361380872%2Ffigure%2Ffig1%2FAS%3A1169080920875058%401655742020827%2FSchematic-diagram-of-Elman-network-structure-in-simple-recurrent-neural-network.jpg&f=1&nofb=1&ipt=1d5a36f3ef48ba69e88b9f96a9176f2e1ed8232b2019b9d3f7f7335e1ee85f1d)
```mermaid
graph TB;
hidden --> context
input --> hidden
context --> hidden
hidden --> output
```
"""


# %%
class SRN_subject:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        activation: list[Callable] = [torch.nn.Tanh],
        lr: float = 0.05,
        initial_w_unif: tuple[float, float] = (-0.1, 0.1),
        loss_fn: Callable = torch.nn.MSELoss(),
    ):

        self.lr = lr
        self.activation = activation
        self.loss_fn = loss_fn

        self.Wxh = make_uniform_tensor(
            extremum=initial_w_unif,
            shape=[hidden_size, input_size + hidden_size],
            grad=True,
        )

        self.Why = make_uniform_tensor(
            extremum=initial_w_unif, shape=[output_size, hidden_size], grad=True
        )

        self.context = torch.zeros(hidden_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        # input of hidden layer: input concatenated with previous output of the hidden layer
        x_cat_context = torch.cat([x, self.context], dim=0)

        h = activation[0](self.Wxh @ x_cat_context)
        self.context = h.detach()  # detach the context from the computational graph to prevent backprop through time
        y = activation[1](self.Why @ h)
        return y

    def backprop(self, y_pred, y):

        loss = self.loss_fn(y_pred, y)
        loss.backward()

        if self.Wxh.grad is None or self.Why.grad is None:
            e = RuntimeError("gradient missing")
            logger.error(e)
            raise e

        with torch.no_grad():
            self.Wxh -= self.lr * self.Wxh.grad
            self.Why -= self.lr * self.Why.grad
            self.Wxh.grad.zero_()
            self.Why.grad.zero_()

        return loss


# %% [md]
"""
Question:
For the backpropagation, when we learn patern on on a screen, how do we see the cases with no stimulus? making them -1 (opposite to the cell with stimulus 1).
"""


# %% [md]
"""
### hyperparameters
"""

# %%

hidden_size = len(unique_vals)
activation = [torch.nn.Tanh(), torch.nn.Sigmoid()]
lr = 0.1
initial_w_unif = (-0.1, 0.1)
loss_fn = torch.nn.MSELoss()

extremum_grid = (0.0, 1.0)

# %% [md]
"""
## Prediction and backprpagation
"""

# %%
input_size = len(unique_vals)
output_size = len(unique_vals)

for i in trange(encoded.shape[0]):
    srn_subject = SRN_subject(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        activation=activation,
        lr=lr,
        initial_w_unif=initial_w_unif,
        loss_fn=loss_fn,
    )

    for j in range(encoded.shape[1] - 1):
        x = encoded[i, j]
        y_true = encoded[i, j + 1]

        # make a grid with the previous value at the encoded label position.
        x_grid = torch.ones(len(unique_vals)) * extremum_grid[0]
        x_grid[x] = extremum_grid[1]

        # make a grid with the response value at the encoded label position.
        y_grid = torch.ones(len(unique_vals)) * extremum_grid[0]
        y_grid[y_true] = extremum_grid[1]

        y_pred_grid = srn_subject.forward(x_grid)
        loss = srn_subject.backprop(y_pred_grid, y_grid)

        y_true = unique_vals[y_true]
        y_pred = torch.argmax(y_pred_grid)
        y_pred_label = unique_vals[y_pred]
        x_label = unique_vals[x]
        logger.trace(
            f"subject {i}, trial {j}, x={x_label}, y={y_true}, y_pred={y_pred_label}, loss={loss}"
        )
