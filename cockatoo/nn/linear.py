import numpy as np

from .module import Module
from ..tensor import Tensor

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Tensor(np.random.randn(in_features, out_features) * 0.01, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

        self.add_parameter("weight", self.weight)
        self.add_parameter("bias", self.bias)

    def forward(self, x):
        return x @ self.weight + self.bias
