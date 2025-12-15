"""
Adam最適化器
"""

from typing import Iterable
import numpy as np
from ..core.parameter import Parameter
from .optimizer import Optimizer


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) 最適化器

    モメンタムとRMSPropを組み合わせた適応的学習率手法
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8
    ):
        """
        Adamの初期化

        Args:
            params: 最適化対象のパラメータ
            lr: 学習率
            betas: (beta1, beta2) モメンタムとRMSPropの減衰率
            eps: 数値安定性のための小さな値
        """
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0  # タイムステップ

        # 1次と2次モーメントの初期化
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        """パラメータを更新"""
        self.t += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.data

            # 1次モーメント（移動平均）の更新
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # 2次モーメント（二乗の移動平均）の更新
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # バイアス補正
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # パラメータ更新
            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
