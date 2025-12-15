"""
損失関数モジュール
"""

from .mse import MSELoss
from .cross_entropy import CrossEntropyLoss

__all__ = [
    "MSELoss",
    "CrossEntropyLoss",
]
