import torch
from torch import nn
from torch import functional as F
from torch.nn.parameter import Parameter
import numpy as np

x = np.array([[0.2,0.4,0.4],
              [0.3,0.3,0.3],
              [0.7,0.0,0.0]])

# reshape softmax to 2d so np.dot gives matrix multiplication

def softmax(x):
    """Compute the softmax of vector x."""
    exps = np.exp(x)
    return exps / np.sum(exps)


soft_max = softmax(x)


def softmax_grad(softmax):
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)



