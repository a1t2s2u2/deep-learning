"""
畳み込み層の実装
"""

import numpy as np
from ..core import Tensor, Parameter
from .module import Module


class Conv2d(Module):
    """2次元畳み込み層"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True
    ):
        """
        Parameters
        ----------
        in_channels : int
            入力チャネル数
        out_channels : int
            出力チャネル数
        kernel_size : int
            カーネルサイズ（正方形を想定）
        stride : int
            ストライド
        padding : int
            パディング
        bias : bool
            バイアスを使用するか
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He初期化
        k = np.sqrt(1.0 / (in_channels * kernel_size * kernel_size))
        self.weight = Parameter(
            Tensor(
                np.random.randn(
                    out_channels, in_channels, kernel_size, kernel_size
                ).astype(np.float32) * k
            )
        )

        if bias:
            self.bias = Parameter(Tensor(np.zeros(out_channels, dtype=np.float32)))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        順伝播

        Parameters
        ----------
        x : Tensor
            入力 (batch_size, in_channels, height, width)

        Returns
        -------
        Tensor
            出力 (batch_size, out_channels, out_height, out_width)
        """
        # im2colを使った畳み込み実装
        batch_size, in_channels, height, width = x.shape

        # パディング
        if self.padding > 0:
            x_padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
        else:
            x_padded = x.data

        # 出力サイズの計算
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # im2col: 画像をカラム形式に変換
        col = self._im2col(x_padded, self.kernel_size, self.stride)
        # col shape: (batch_size * out_height * out_width, in_channels * kernel_size * kernel_size)

        # 重みを行列形式に変換
        col_weight = self.weight.data.reshape(self.out_channels, -1).T
        # col_weight shape: (in_channels * kernel_size * kernel_size, out_channels)

        # 行列積で畳み込み計算
        out = np.dot(col, col_weight)
        # out shape: (batch_size * out_height * out_width, out_channels)

        # バイアスを加算
        if self.bias is not None:
            out += self.bias.data

        # 形状を戻す
        out = out.reshape(batch_size, out_height, out_width, self.out_channels)
        out = out.transpose(0, 3, 1, 2)  # (batch, out_channels, out_height, out_width)

        # Tensorとして返す（逆伝播用の計算グラフを構築）
        output = Tensor(out, requires_grad=x.requires_grad or self.weight.requires_grad)

        # 逆伝播の実装
        if output.requires_grad:
            def _backward():
                grad_output = output.grad.data
                # grad_output shape: (batch_size, out_channels, out_height, out_width)

                # 勾配を列形式に変換
                grad_output_col = grad_output.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

                # 重みの勾配
                grad_weight = np.dot(col.T, grad_output_col)
                grad_weight = grad_weight.T.reshape(self.weight.shape)

                if self.weight.grad is None:
                    self.weight.grad = Tensor(grad_weight)
                else:
                    self.weight.grad.data += grad_weight

                # バイアスの勾配
                if self.bias is not None:
                    grad_bias = np.sum(grad_output_col, axis=0)
                    if self.bias.grad is None:
                        self.bias.grad = Tensor(grad_bias)
                    else:
                        self.bias.grad.data += grad_bias

                # 入力の勾配
                if x.requires_grad:
                    grad_col = np.dot(grad_output_col, col_weight.T)
                    grad_x = self._col2im(grad_col, x_padded.shape, self.kernel_size, self.stride)

                    # パディングを除去
                    if self.padding > 0:
                        grad_x = grad_x[:, :, self.padding:-self.padding, self.padding:-self.padding]

                    if x.grad is None:
                        x.grad = Tensor(grad_x)
                    else:
                        x.grad.data += grad_x

            output._backward = _backward

        return output

    def _im2col(self, x: np.ndarray, kernel_size: int, stride: int) -> np.ndarray:
        """
        画像をカラム形式に変換

        Parameters
        ----------
        x : np.ndarray
            入力画像 (batch_size, in_channels, height, width)
        kernel_size : int
            カーネルサイズ
        stride : int
            ストライド

        Returns
        -------
        np.ndarray
            カラム形式 (batch_size * out_height * out_width, in_channels * kernel_size * kernel_size)
        """
        batch_size, in_channels, height, width = x.shape
        out_height = (height - kernel_size) // stride + 1
        out_width = (width - kernel_size) // stride + 1

        col = np.zeros(
            (batch_size, in_channels, kernel_size, kernel_size, out_height, out_width),
            dtype=x.dtype
        )

        for y in range(kernel_size):
            y_max = y + stride * out_height
            for x_pos in range(kernel_size):
                x_max = x_pos + stride * out_width
                col[:, :, y, x_pos, :, :] = x[:, :, y:y_max:stride, x_pos:x_max:stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(-1, in_channels * kernel_size * kernel_size)
        return col

    def _col2im(self, col: np.ndarray, x_shape: tuple, kernel_size: int, stride: int) -> np.ndarray:
        """
        カラム形式を画像に戻す

        Parameters
        ----------
        col : np.ndarray
            カラム形式
        x_shape : tuple
            元の画像の形状
        kernel_size : int
            カーネルサイズ
        stride : int
            ストライド

        Returns
        -------
        np.ndarray
            画像形式
        """
        batch_size, in_channels, height, width = x_shape
        out_height = (height - kernel_size) // stride + 1
        out_width = (width - kernel_size) // stride + 1

        col = col.reshape(batch_size, out_height, out_width, in_channels, kernel_size, kernel_size)
        col = col.transpose(0, 3, 4, 5, 1, 2)

        x = np.zeros(x_shape, dtype=col.dtype)

        for y in range(kernel_size):
            y_max = y + stride * out_height
            for x_pos in range(kernel_size):
                x_max = x_pos + stride * out_width
                x[:, :, y:y_max:stride, x_pos:x_max:stride] += col[:, :, y, x_pos, :, :]

        return x

    def __repr__(self):
        return (f"Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})")
