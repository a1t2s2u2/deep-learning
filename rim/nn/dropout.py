"""
Dropout層の実装
"""

import numpy as np
from ..core import Tensor
from .module import Module


class Dropout(Module):
    """Dropout層"""

    def __init__(self, p: float = 0.5):
        """
        Parameters
        ----------
        p : float
            ドロップアウト率（0から1の間）
        """
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        """
        順伝播

        Parameters
        ----------
        x : Tensor
            入力

        Returns
        -------
        Tensor
            出力
        """
        if not self.training or self.p == 0:
            # 推論モードまたはp=0の場合はそのまま返す
            return x

        # 訓練モード: ランダムにドロップアウト
        # Inverted Dropoutを使用（スケーリングを訓練時に行う）
        mask = np.random.binomial(1, 1 - self.p, size=x.shape).astype(np.float32)
        out = x.data * mask / (1 - self.p)

        output = Tensor(out, requires_grad=x.requires_grad)

        # 逆伝播の実装
        if output.requires_grad:
            def _backward():
                grad_output = output.grad.data
                grad_x = grad_output * mask / (1 - self.p)

                if x.grad is None:
                    x.grad = Tensor(grad_x)
                else:
                    x.grad.data += grad_x

            output._backward = _backward

        return output

    def __repr__(self):
        return f"Dropout(p={self.p})"
