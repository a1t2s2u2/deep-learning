"""
rim - NumPyで作る深層学習フレームワーク

使用例:
    from rim import Tensor
    from rim.core import relu, sigmoid

    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    y = relu(x)
    loss = y.sum()
    loss.backward()
    print(x.grad)
"""

__version__ = '0.1.0'

from .core import (
    Tensor,
    Function,
    Context,
    relu,
    sigmoid,
    tanh,
    exp,
    log,
    softmax,
)

__all__ = [
    'Tensor',
    'Function',
    'Context',
    'relu',
    'sigmoid',
    'tanh',
    'exp',
    'log',
    'softmax',
]
