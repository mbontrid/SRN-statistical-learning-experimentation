import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging


class ElmanSRN:
    def __init__(self, sizes: list[int] = [1, 2, 1]):
        self.hidden_size = sizes[1]
        self.input_size = sizes[0]
        self.output_size = sizes[2]
        self.rnn = RNN(self.input_size, self.hidden_size, self.output_size)

    def load_numpy(self, ndarray: np.ndarray):
        ndarray = np.array([ord(c) if isinstance(c, str) else c for c in ndarray])
        print("full numpy: ", ndarray)
        input = []
        target = []
        for i in range(ndarray.shape[0] - 1):
            input.append(ndarray[i])
            target.append(ndarray[i + 1])
        input = torch.from_numpy(np.array(input))
        target = torch.from_numpy(np.array(target))

        if torch.accelerator.is_available():
            print("loading tensor in accelerator...")
            input = input.to(torch.accelerator.current_accelerator())
            target = target.to(torch.accelerator.current_accelerator())
            print("loading done !")

        print("tensors size:", input.size())
        print("first input: ", input[0])
        print("first target: ", target[0])

    def training(self):
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.rnn.parameters(), lr=0.1, momentum=0.9)

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.rnn.parameters(), lr=0.1, momentum=0.9)

        for iter in range(10):
            running_loss = 0
            hidden = self.rnn.init_hidden()
            for i in range(input.size(0)):
                optimizer.zero_grad()
                output, hidden = self.rnn(input[i].reshape(1, 5), hidden.detach())
                loss = criterion(output, target[i].reshape(1, 5))
                loss.backward(retain_graph=True)
                running_loss += loss.item()
                optimizer.step()
            print(
                "iter ",
                str(iter),
                " average loss on iteration :",
                str(running_loss / input.size(0)),
            )


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn1 = nn.RNNCell(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, input, hidden):
        hidden = self.rnn1(input, hidden)
        output = self.sig(self.linear(hidden))
        return (
            output,
            hidden,
        )  # we return both output and hidden state, as both will be needed for the next step

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size, dtype=torch.double)
