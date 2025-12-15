"""
プーリング層の実装
"""

import numpy as np
from ..core import Tensor
from .module import Module


class MaxPool2d(Module):
    """2次元Max Pooling層"""

    def __init__(self, kernel_size: int, stride: int = None):
        """
        Parameters
        ----------
        kernel_size : int
            プーリングウィンドウのサイズ
        stride : int
            ストライド（Noneの場合はkernel_sizeと同じ）
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

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
            出力 (batch_size, channels, out_height, out_width)
        """
        batch_size, channels, height, width = x.shape

        # 出力サイズの計算
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1

        # プーリング実行
        out = np.zeros((batch_size, channels, out_height, out_width), dtype=x.data.dtype)
        max_indices = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=np.int32)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                window = x.data[:, :, h_start:h_end, w_start:w_end]
                # 各ウィンドウの最大値を取得
                out[:, :, i, j] = np.max(window, axis=(2, 3))

                # 最大値のインデックスを保存（逆伝播用）
                window_reshaped = window.reshape(batch_size, channels, -1)
                max_idx = np.argmax(window_reshaped, axis=2)
                max_indices[:, :, i, j, 0] = max_idx // self.kernel_size  # 行インデックス
                max_indices[:, :, i, j, 1] = max_idx % self.kernel_size   # 列インデックス

        output = Tensor(out, requires_grad=x.requires_grad)

        # 逆伝播の実装
        if output.requires_grad:
            def _backward():
                grad_output = output.grad.data
                grad_x = np.zeros_like(x.data)

                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        w_start = j * self.stride

                        for b in range(batch_size):
                            for c in range(channels):
                                # 最大値だった位置に勾配を伝播
                                max_h = max_indices[b, c, i, j, 0]
                                max_w = max_indices[b, c, i, j, 1]
                                grad_x[b, c, h_start + max_h, w_start + max_w] += grad_output[b, c, i, j]

                if x.grad is None:
                    x.grad = Tensor(grad_x)
                else:
                    x.grad.data += grad_x

            output._backward = _backward

        return output

    def __repr__(self):
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride})"


class AvgPool2d(Module):
    """2次元Average Pooling層"""

    def __init__(self, kernel_size: int, stride: int = None):
        """
        Parameters
        ----------
        kernel_size : int
            プーリングウィンドウのサイズ
        stride : int
            ストライド（Noneの場合はkernel_sizeと同じ）
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

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
            出力 (batch_size, channels, out_height, out_width)
        """
        batch_size, channels, height, width = x.shape

        # 出力サイズの計算
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1

        # プーリング実行
        out = np.zeros((batch_size, channels, out_height, out_width), dtype=x.data.dtype)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                window = x.data[:, :, h_start:h_end, w_start:w_end]
                # 各ウィンドウの平均値を取得
                out[:, :, i, j] = np.mean(window, axis=(2, 3))

        output = Tensor(out, requires_grad=x.requires_grad)

        # 逆伝播の実装
        if output.requires_grad:
            def _backward():
                grad_output = output.grad.data
                grad_x = np.zeros_like(x.data)

                # 平均なので勾配を均等に分配
                grad_per_element = 1.0 / (self.kernel_size * self.kernel_size)

                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        grad_x[:, :, h_start:h_end, w_start:w_end] += (
                            grad_output[:, :, i:i+1, j:j+1] * grad_per_element
                        )

                if x.grad is None:
                    x.grad = Tensor(grad_x)
                else:
                    x.grad.data += grad_x

            output._backward = _backward

        return output

    def __repr__(self):
        return f"AvgPool2d(kernel_size={self.kernel_size}, stride={self.stride})"
