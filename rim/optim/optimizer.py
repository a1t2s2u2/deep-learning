"""
最適化器の基底クラス
"""

from typing import Iterable
from ..core.parameter import Parameter


class Optimizer:
    """
    全ての最適化器の基底クラス
    """

    def __init__(self, params: Iterable[Parameter], lr: float):
        """
        最適化器の初期化

        Args:
            params: 最適化対象のパラメータ
            lr: 学習率
        """
        self.params = list(params)
        self.lr = lr

    def step(self):
        """
        パラメータを更新

        サブクラスで実装が必要
        """
        raise NotImplementedError

    def zero_grad(self):
        """全パラメータの勾配をゼロクリア"""
        for param in self.params:
            param.zero_grad()
