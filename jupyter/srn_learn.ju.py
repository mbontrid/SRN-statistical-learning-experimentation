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
logger.info(f"responses after transposition= {responses}")
# responses = np.expand_dims(responses, axis=-1)
# logger.trace(f"response after expansions on last dim={responses}")
# get set of responses values and their order of appearance(giving them a numerical id)
unique_vals, encoded = np.unique(responses.reshape(-1), return_inverse=True)
logger.info(f"unique values (set)={unique_vals}")
logger.info(f"encoded values={encoded}")
encoded = encoded.reshape(responses.shape)
logger.debug(f"final encoded shape (subjests, responses)={encoded.shape}")


# %%
def make_uniform_tensor(
    extremum: tuple[float, float], shape: Sequence[int], grad: bool
):
    return extremum[0] + (extremum[1] - extremum[0]) * torch.rand(
        size=shape, requires_grad=grad
    )


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
        learn_rate: float = 0.05,
        initial_w_unif: tuple[float, float] = (-0.1, 0.1)
    ):

        self.lr = learn_rate
        self.activation = activation

        self.Wxh = make_uniform_tensor(
            extremum=initial_w_unif, shape=[hidden_size, input_size + hidden_size], grad=True
        ).requires_grad_()
        
        self.Why = make_uniform_tensor(
            extremum=initial_w_unif, shape=[output_size, hidden_size], grad=True
        ).requires_grad_()

        self.context = torch.zeros(hidden_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        # input of hidden layer: input concatenated with previous output of the hidden layer
        x_cat_context = torch.cat([x, self.context], dim=0)

        h = activation[0](self.Wxh @ x_cat_context)
        self.context = h
        y = activation[1](self.Why @ h)
        return y

    def backprop(self, y_pred, y):

        if self.Wxh.grad is None:
            e = RuntimeError("Wxh grad is None")
            logger.error(e)
            raise e
        if self.Why.grad is None:
            e = RuntimeError("Why grad is None")
            logger.error(e)
            raise e

        loss = torch.mean((y_pred - y) ** 2)
        loss.backward()

        with torch.no_grad():
            self.Wxh -= self.lr * self.Wxh.grad
            self.Why -= self.lr * self.Why.grad
            self.Wxh.grad.zero_()
            self.Why.grad.zero_()

        return loss



# %%[md]
```
For the backpropagation, when we learn patern on on a screen
```

# %%

input_size = len(unique_vals)
hidden_size = len(unique_vals)
output_size = len(unique_vals)

activation = [torch.nn.Tanh(), torch.nn.Softmax()]
lr = 0.1


for i in trange(encoded.shape[0]):
    srn_subject = SRN_subject(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        activation=activation,
        learn_rate=lr,
    )

    for j in range(encoded.shape[1] - 1):
        x = encoded[i, j]
        y = encoded[i, j + 1]

        # make a grid with the previous value at the encoded label position.
        x_grid = torch.ones(len(unique_vals)) * -1
        x_grid[x] = 1

        # make a grid with the response value at the encoded label position.
        y_grid = torch.ones(len(unique_vals)) * -1
        y_grid[y] = 1

        y_pred_grid = srn_subject.forward(x_grid)
        loss = srn_subject.backprop(y_pred_grid, y_grid)
        y_pred = torch.argmax(y_pred_grid)
        logger.debug(
            f"subject {i}, trial {j}, x={x}, y={y}, y_pred={y_pred}, loss={loss}"
        )


# %%

class SRN(AbstractNNModel):
    def __init__(self, params):
        self.allowed_parameters = {
            "vocab_size": int,
            "hidden_size": int,
            "lr": float,
            "mu": float,
            "clearval": float,
            "epochs": int,
        }
        self.init_model(params)

    def validate_parameters(self, params):
        for parameter, value in params.items():
            if parameter not in self.allowed_parameters.keys():
                raise ParameterNotAllowedException(parameter)
            expected_type = self.allowed_parameters[parameter]
            if not isinstance(value, expected_type):
                raise WrongParameterTypeException(
                    parameter, value.type(), expected_type
                )
        for parameter in self.allowed_parameters:
            if parameter not in params.key():
                raise MissingParameterException(parameter)
        return True

    def init_model(self, params):

        self.validate_parameters(params)
        self.input_size = params["vocab_size"]
        self.hidden_size = params["hidden_size"]
        self.output_size = params["vocab_size"]

        self.lr = params["lr"]
        self.mu = params["mu"]
        self.clearval = params["clearval"]

        self.Wxh = np.random.uniform(-0.1, 0.1, (self.hidden_size, self.output_size))
        self.Whh = np.random.uniform(-0.1, 0.1, (self.hidden_size, self.hidden_size))
        self.Why = np.random.uniform(-0.1, 0.1, (self.output_size, self.hidden_size))

        self.bh = np.zeros(self.hidden_size)
        self.by = np.zeros(self.output_size)

        self.reset_context()

    def reset_context(self):
        self.context = np.ones(self.hidden_size) * self.clearval

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    def forward(self, x):

        h = np.tanh(self.Wxh @ x + self.Whh @ self.context + self.bh)

        y = self.softmax(self.Why @ h + self.by)

        return h, y

    def train_step(self, x, target):

        h, y = self.forward(x)

        dy = y - target

        dWhy = np.outer(dy, h)
        dby = dy

        dh = self.Why.T @ dy
        dh_raw = (1 - h**2) * dh

        dWxh = np.outer(dh_raw, x)
        dWhh = np.outer(dh_raw, self.context)
        dbh = dh_raw

        self.Why -= self.lr * dWhy
        self.by -= self.lr * dby

        self.Wxh -= self.lr * dWxh
        self.Whh -= self.lr * dWhh
        self.bh -= self.lr * dbh

        self.context = h + self.mu * self.context

        loss = -np.sum(target * np.log(y + 1e-12))
        return loss

