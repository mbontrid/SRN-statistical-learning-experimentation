# %% [md]
"""
# Simple recursive network for sequence learning
"""

# %% [md]
"""
## Setup
"""

# %%
import sys
from pathlib import Path
import numpy as np
import torch
from typing import Callable, Sequence
from tqdm import trange


# import local packages
sys.path.append(str(Path().resolve().parents[0]))

from src.utils.logger import setup_logger, add_log_level
from src.utils.args import get_args
from src.data.formater import PandasLoader

# %%
# global setup parameters
args = get_args()
# define the verbose level
logger_level = "TRACE"

# %%
# golbal setup
add_log_level("TRACE", 5)
logger = setup_logger(name="logger", level=logger_level)
logger.info("logger set to info level")
logger.trace("TRACE logger activated")

logger.debug(f"arguments dictionnray: {args}")

# %% [md]
"""
## Data setup

"""
# %%
# load data
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
## SRN implementation
![srn elman diagrma](https://web.stanford.edu/group/pdplab/pdphandbook/srn_net.png)

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
    """Basic Elman SRN. Made of two weighted layer, the first one has it's ouptut copied for future concatenation with the next input.

    Attributes:
        lr: learning rate.
        activation: List of activation functions. The function are nessary.
        loss_fn: loss function to use for backpropagation.
        Wxh: Weights of the first layer.
        Why: Weights of the second layer.
        context: After each forward pass, hidden layer is copied in context.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        activation: tuple[Callable, Callable] = (torch.nn.Tanh(), torch.nn.Sigmoid()),
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

        self.bh = torch.zeros(hidden_size, requires_grad=True)
        self.by = torch.zeros(output_size, requires_grad=True)

        self.context = torch.zeros(hidden_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        # input of hidden layer: input concatenated with previous output of the hidden layer
        x_cat_context = torch.cat([x, self.context], dim=0)

        h: torch.Tensor = activation[0](self.Wxh @ x_cat_context + self.bh)
        self.context = h.clone().detach()  # detach the context from the computational graph to prevent backprop through time
        y = activation[1](self.Why @ h + self.by)
        return y

    def backprop(self, y_pred, y):

        loss = self.loss_fn(y_pred, y)
        loss.backward()  # compute gradients

        with torch.no_grad():
            self.Wxh -= self.lr * self.Wxh.grad
            self.Why -= self.lr * self.Why.grad
            self.bh -= self.lr * self.bh.grad
            self.by -= self.lr * self.by.grad

            # reset the gradients.
            self.Wxh.grad.zero_()
            self.Why.grad.zero_()
            self.bh.grad.zero_()
            self.by.grad.zero_()

        return loss


# %% [md]
"""
Question:
For the backpropagation, when we learn patern on on a screen, how do we see the cases with no stimulus? making them -1 (opposite to the cell with stimulus 1).
"""


# %% [md]
"""
### hyperparameters
Hyperparameters of the srn.
"""

# %%

hidden_size = len(unique_vals)
activation = (torch.nn.Tanh(), torch.nn.Sigmoid())
lr = 0.05
# extremum of the uniform distribution for the initial weights.
initial_w_unif = (-0.1, 0.1)
loss_fn = torch.nn.MSELoss()

# lower val is cell with no stimulus, higher val is cell with stimulus for each steps
extremum_grid = (
    0.0,
    1.0,
)

# how many subjects to compute.
subject_to_test = 0.1

# %% [md]
"""
## Prediction and backprpagation
```mermaid
graph LR;
a --> a_emb --> d
b --> b_emb --> e
c --> c_emb --> f
```
"""

# %%
input_size = len(unique_vals)
output_size = len(unique_vals)


for subject in range(int(encoded.shape[0] * subject_to_test)):
    # instantiate a new srn for each subject.
    srn_subject = SRN_subject(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        activation=activation,
        lr=lr,
        initial_w_unif=initial_w_unif,
        loss_fn=loss_fn,
    )

    for trial in range(encoded.shape[1] - 1):
        # the next value is the true prediction of the input value.
        x = encoded[subject, trial]
        y_true = encoded[subject, trial + 1]

        # make a grid with stimulus at the encoded label position.
        x_grid = torch.ones(len(unique_vals)) * extremum_grid[0]
        x_grid[x] = extremum_grid[1]

        # make a grid with the expected stimulus at the encoded label position.
        y_grid = torch.ones(len(unique_vals)) * extremum_grid[0]
        y_grid[y_true] = extremum_grid[1]

        y_pred_grid = srn_subject.forward(x_grid)
        loss = srn_subject.backprop(y_pred_grid, y_grid)

        y_true_label = unique_vals[y_true]  # get the expected letter
        # get the encoded index of the most expected stimulus.
        y_pred = torch.argmax(y_pred_grid)
        y_pred_label = unique_vals[y_pred]  # get the predicted letter
        x_label = unique_vals[x]  # get the input letter

        if logger_level == "TRACE":
            # only print trials with no embeddings (random values)
            if y_true_label in ["d", "e", "f"]:
                logger.trace(
                    f"subject {subject}, trial {trial}, x={x_label}, y={y_true_label}, y_pred={y_pred_label}, correct={y_true_label == y_pred_label}, loss={loss}"
                )
