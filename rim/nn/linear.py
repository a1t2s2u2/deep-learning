"""
全結合層（Linear layer）の実装
"""

import math
from ..core.tensor import Tensor
from ..core.parameter import Parameter
from .module import Module


class Linear(Module):
    """
    全結合層（線形変換）

    y = xW^T + b を計算

    Attributes:
        in_features: 入力の特徴数
        out_features: 出力の特徴数
        weight: 重み行列 (out_features, in_features)
        bias: バイアスベクトル (out_features,)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Linear層の初期化

        Args:
            in_features: 入力の特徴数
            out_features: 出力の特徴数
            bias: バイアスを使用するか
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 重みの初期化（Heの初期化）
        k = 1.0 / math.sqrt(in_features)
        self.weight = Parameter(
            Tensor.rand(out_features, in_features) * 2 * k - k
        )

        if bias:
            self.bias = Parameter(
                Tensor.rand(out_features) * 2 * k - k
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        順伝播

        Args:
            x: 入力テンソル (..., in_features)

        Returns:
            出力テンソル (..., out_features)
        """
        # y = xW^T
        output = x @ self.weight.T

        # バイアスを加算
        if self.bias is not None:
            output = output + self.bias

        return output

    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
