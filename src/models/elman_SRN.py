import numpy as np


class generic_network:
    def __init__(self, dimension: list[int] = [8, 8, 5, 8, 8]):
        input_size = dimension[0]
        hidden_size = dimension[1:-1]
        output_size = dimension[-1]
        print(input_size, hidden_size, output_size)


class Elman_SRN:
    def __init__(self, sizes: list[int] = [8, 8, 8]):
        self.input_array = np.ones(sizes[0])
        self.first_hidden_array = np.random.rand(sizes[1] + sizes[0])
        self.output_array = np.ones(sizes[2])

    def forward(self, sequence: tuple[int]):
        pass
