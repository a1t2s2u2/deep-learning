"""
活性化関数のModuleラッパー
"""

from ..core.tensor import Tensor
from ..core import activations as F
from .module import Module


class ReLU(Module):
    """ReLU活性化関数のModuleラッパー"""

    def forward(self, x: Tensor) -> Tensor:
        """順伝播"""
        return F.relu(x)


class Sigmoid(Module):
    """Sigmoid活性化関数のModuleラッパー"""

    def forward(self, x: Tensor) -> Tensor:
        """順伝播"""
        return F.sigmoid(x)


class Tanh(Module):
    """Tanh活性化関数のModuleラッパー"""

    def forward(self, x: Tensor) -> Tensor:
        """順伝播"""
        return F.tanh(x)


class Softmax(Module):
    """Softmax活性化関数のModuleラッパー"""

    def __init__(self, axis: int = -1):
        """
        Softmaxの初期化

        Args:
            axis: Softmaxを適用する軸
        """
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        """順伝播"""
        return F.softmax(x, axis=self.axis)

    def __repr__(self) -> str:
        return f"Softmax(axis={self.axis})"
