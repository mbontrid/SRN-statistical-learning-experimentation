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
from typing import Callable, Sequence
from tqdm import trange

import numpy as np
import torch
import matplotlib.pyplot as plt

# import local packages
sys.path.append(str(Path().resolve().parents[0]))

from src.data.formater import PandasLoader
from src.data.utils import rand_key_emb_value
from src.utils.args import get_args
from src.utils.logger import add_log_level, setup_logger

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
dict_unique_vals = {
    label: encod for label, encod in zip(unique_vals, range(len(unique_vals)))
}
logger.info(f"unique values (set)={unique_vals}")
logger.info(f"encoded values={encoded}")
logger.info(f"number of labels={len(unique_vals)}")
encoded = encoded.reshape(responses.shape)
logger.debug(f"final encoded shape (subjects, responses)={encoded.shape}")


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

        h: torch.Tensor = self.activation[0](self.Wxh @ x_cat_context + self.bh)
        self.context = h.detach()  # detach the context from the computational graph to prevent backprop through time
        y = self.activation[1](self.Why @ h + self.by)
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


# %%
def predict_grid(model: SRN_subject, x: int, y_true: int, unique_vals):
    # make a grid with stimulus at the encoded label position.
    x_grid = torch.ones(len(unique_vals)) * grid_min
    x_grid[x] = grid_max

    #  make a grid with the expected stimulus at the encoded label position.
    y_true_grid = torch.ones(len(unique_vals)) * grid_min
    y_true_grid[y_true] = grid_max

    y_pred_grid = model.forward(x_grid)
    return x_grid, y_true_grid, y_pred_grid


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
lr = 0.7
# extremum of the uniform distribution for the initial weights.
initial_w_unif = (-0.1, 0.1)
loss_fn = torch.nn.MSELoss()

# lower val is cell with no stimulus, higher val is cell with stimulus for each steps
grid_max = 1.0
grid_min = 0.0

# how many subjects to compute.
test_population_ratio = 0.05

epochs = 1

input_size = len(unique_vals)
output_size = len(unique_vals)

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


for subject in range(int(encoded.shape[0] * test_population_ratio)):
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

    for epoch in range(epochs):
        true_predictable = 0
        true_predictable_prediction = 0

        for step in range(encoded.shape[1] - 1):
            x = encoded[subject, step]
            y_true = encoded[subject, step + 1]

            x_grid, y_grid, y_pred_grid = predict_grid(
                model=srn_subject,
                x=x,
                y_true=y_true,
                unique_vals=unique_vals,
            )

            loss = srn_subject.backprop(y_pred_grid, y_grid)

            y_true_label = unique_vals[y_true]  # get the expected letter
            # get the encoded index of the most expected stimulus.
            y_pred = torch.argmax(y_pred_grid)
            y_pred_label = unique_vals[y_pred]  # get the predicted letter
            x_label = unique_vals[x]  # get the input letter

            if y_true_label in ["d", "e", "f"]:
                logger.trace(
                    f"subject {subject}, trial {step}, x={x_label}, y={y_true_label}, y_pred={y_pred_label}, correct={y_true_label == y_pred_label}, loss={loss}",
                )
                true_predictable += 1
                if y_true_label == y_pred_label:
                    true_predictable_prediction += 1

        logger.debug(
            f"subject={subject}, epoch={epoch}, mean_true={true_predictable_prediction / true_predictable if true_predictable > 0 else 0}"
        )

# %% [md]
"""
As we can see, the result si not satiffying. To assert the capablity of the snr to learn, we will use a longer sequence generated on the stack.

### Data generator
"""

# %%

lr = 0.5
hidden_size = len(unique_vals)
generate_size = 100000
true_predictable = []
loss_evolution = []

srn_subject = SRN_subject(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    activation=activation,
    lr=lr,
    initial_w_unif=initial_w_unif,
    loss_fn=loss_fn,
)


seq_stimulus = {
    dict_unique_vals.get("a"): dict_unique_vals.get("d"),
    dict_unique_vals.get("b"): dict_unique_vals.get("e"),
    dict_unique_vals.get("c"): dict_unique_vals.get("f"),
}
embedings = tuple(
    set(encoded.reshape(-1).tolist())
    - set(seq_stimulus.keys())
    - set(seq_stimulus.values())
)

sequ_generator = rand_key_emb_value(
    seq_stimulus=seq_stimulus, embeddings=embedings, size=generate_size
)


# %% [md]
"""
The sequ_generator can generate a simulation of data.
"""
# %%
y_true = next(sequ_generator)
for step, new_value in enumerate(sequ_generator):
    x = y_true
    y_true = new_value

    x_grid, y_grid, y_pred_grid = predict_grid(
        model=srn_subject,
        x=x,
        y_true=y_true,
        unique_vals=unique_vals,
    )
    loss = srn_subject.backprop(y_pred_grid, y_grid)

    y_true_label = unique_vals[y_true]  # get the expected letter
    # get the encoded index of the most expected stimulus.
    y_pred = torch.argmax(y_pred_grid)
    y_pred_label = unique_vals[y_pred]  # get the predicted letter
    x_label = unique_vals[x]  # get the input letter

    loss_evolution.append(loss)
    if y_true_label in ["d", "e", "f"]:
        print(
            f"step={step}, x={x_label}, y={y_true_label}, y_pred={y_pred_label}, correct={y_true_label == y_pred_label}, loss={loss}",
            end="\r",
        )
        if y_true_label == y_pred_label:
            true_predictable.append(True)
        else:
            true_predictable.append(False)


# %% [md]
"""
Visibly, with enough simulation steps, the srn can predict a predictable value without error.
For a->d, b->e, c->f and 24 embeddings, the number of possible 3 tuples is:
$$\text{3 steps possibilities} = 3 * 24 = 72 $$
So there is a chance that the srn simply overfit on all the possiblities if there is too much steps.
"""
# %%
