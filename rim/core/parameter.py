"""
学習可能なパラメータを表すクラス
"""

from .tensor import Tensor


class Parameter(Tensor):
    """
    学習可能なパラメータを表すクラス

    Tensorを継承し、常にrequires_grad=Trueとして扱われる
    モデルのパラメータ（重みやバイアス）を表現するために使用
    """

    def __init__(self, data, requires_grad=True):
        """
        Parameterの初期化

        Args:
            data: パラメータのデータ
            requires_grad: 勾配計算の要否（デフォルトTrue）
        """
        super().__init__(data, requires_grad=requires_grad)

    def __repr__(self) -> str:
        return f"Parameter({self.data})"
