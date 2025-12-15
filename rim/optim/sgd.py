"""
確率的勾配降下法（SGD）
"""

from typing import Iterable
from ..core.parameter import Parameter
from .optimizer import Optimizer


class SGD(Optimizer):
    """
    確率的勾配降下法（Stochastic Gradient Descent）

    パラメータ更新式:
        θ = θ - lr * grad
    """

    def __init__(self, params: Iterable[Parameter], lr: float = 0.01):
        """
        SGDの初期化

        Args:
            params: 最適化対象のパラメータ
            lr: 学習率
        """
        super().__init__(params, lr)

    def step(self):
        """パラメータを更新"""
        for param in self.params:
            if param.grad is not None:
                # θ = θ - lr * grad
                param.data = param.data - self.lr * param.grad.data
