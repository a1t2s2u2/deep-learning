"""
平均二乗誤差（Mean Squared Error）損失
"""

from ..core.tensor import Tensor
from ..nn.module import Module


class MSELoss(Module):
    """
    平均二乗誤差損失

    L = mean((y_pred - y_true)^2)
    """

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        損失を計算

        Args:
            y_pred: 予測値
            y_true: 真の値

        Returns:
            損失（スカラー）
        """
        diff = y_pred - y_true
        loss = (diff * diff).mean()
        return loss
