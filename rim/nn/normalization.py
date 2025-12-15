"""
正規化層の実装
"""

import numpy as np
from ..core import Tensor, Parameter
from .module import Module


class BatchNorm2d(Module):
    """2次元Batch Normalization層"""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        """
        Parameters
        ----------
        num_features : int
            入力のチャネル数
        eps : float
            数値安定性のための小さな値
        momentum : float
            移動平均の更新率
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # 学習可能なパラメータ
        self.gamma = Parameter(Tensor(np.ones(num_features, dtype=np.float32)))
        self.beta = Parameter(Tensor(np.zeros(num_features, dtype=np.float32)))

        # 移動平均（学習時に更新、推論時に使用）
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

    def forward(self, x: Tensor) -> Tensor:
        """
        順伝播

        Parameters
        ----------
        x : Tensor
            入力 (batch_size, channels, height, width)

        Returns
        -------
        Tensor
            出力 (batch_size, channels, height, width)
        """
        if self.training:
            # 訓練モード: バッチ統計を使用
            batch_mean = np.mean(x.data, axis=(0, 2, 3), keepdims=False)
            batch_var = np.var(x.data, axis=(0, 2, 3), keepdims=False)

            # 移動平均を更新
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            mean = batch_mean
            var = batch_var
        else:
            # 推論モード: 移動平均を使用
            mean = self.running_mean
            var = self.running_var

        # 正規化
        # (batch, channels, height, width) の形状に合わせてブロードキャスト
        mean = mean.reshape(1, -1, 1, 1)
        var = var.reshape(1, -1, 1, 1)
        gamma = self.gamma.data.reshape(1, -1, 1, 1)
        beta = self.beta.data.reshape(1, -1, 1, 1)

        x_normalized = (x.data - mean) / np.sqrt(var + self.eps)
        out = gamma * x_normalized + beta

        output = Tensor(out, requires_grad=x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad)

        # 逆伝播の実装（訓練時のみ）
        if output.requires_grad and self.training:
            def _backward():
                grad_output = output.grad.data
                batch_size = x.data.shape[0]
                spatial_size = x.data.shape[2] * x.data.shape[3]
                N = batch_size * spatial_size

                # gammaとbetaの勾配
                grad_gamma = np.sum(grad_output * x_normalized, axis=(0, 2, 3))
                grad_beta = np.sum(grad_output, axis=(0, 2, 3))

                if self.gamma.grad is None:
                    self.gamma.grad = Tensor(grad_gamma)
                else:
                    self.gamma.grad.data += grad_gamma

                if self.beta.grad is None:
                    self.beta.grad = Tensor(grad_beta)
                else:
                    self.beta.grad.data += grad_beta

                # 入力の勾配
                if x.requires_grad:
                    grad_x_normalized = grad_output * gamma

                    # バッチ正規化の逆伝播（複雑な計算）
                    std_inv = 1.0 / np.sqrt(var + self.eps)

                    grad_var = np.sum(
                        grad_x_normalized * (x.data - mean) * -0.5 * (std_inv ** 3),
                        axis=(0, 2, 3), keepdims=True
                    )

                    grad_mean = np.sum(
                        grad_x_normalized * -std_inv,
                        axis=(0, 2, 3), keepdims=True
                    ) + grad_var * np.sum(-2.0 * (x.data - mean), axis=(0, 2, 3), keepdims=True) / N

                    grad_x = (
                        grad_x_normalized * std_inv +
                        grad_var * 2.0 * (x.data - mean) / N +
                        grad_mean / N
                    )

                    if x.grad is None:
                        x.grad = Tensor(grad_x)
                    else:
                        x.grad.data += grad_x

            output._backward = _backward

        return output

    def __repr__(self):
        return f"BatchNorm2d(num_features={self.num_features}, eps={self.eps}, momentum={self.momentum})"
