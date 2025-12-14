"""
自動微分用の関数基底クラス
全ての微分可能な演算はFunctionを継承する
"""

from typing import Tuple, Any
from .tensor import Tensor


class Function:
    """
    全ての微分可能な関数の基底クラス

    自動微分をサポートする演算は、このクラスを継承し
    forward()とbackward()メソッドを実装する
    """

    @staticmethod
    def forward(ctx: 'Context', *args, **kwargs) -> Any:
        """
        順伝播の計算を実行

        Args:
            ctx: 逆伝播用に情報を保存するContextオブジェクト
            *args: 入力引数
            **kwargs: キーワード引数

        Returns:
            順伝播の計算結果
        """
        raise NotImplementedError

    @staticmethod
    def backward(ctx: 'Context', grad_output: Tensor) -> Tuple[Tensor, ...]:
        """
        逆伝播の計算を実行

        Args:
            ctx: forward()で保存した情報を含むContextオブジェクト
            grad_output: 出力に対する勾配

        Returns:
            各入力に対する勾配のタプル
        """
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        """
        関数を適用（順伝播 + 逆伝播の設定）

        ユーザーが呼び出すエントリーポイント
        """
        ctx = Context()
        output = cls.forward(ctx, *args, **kwargs)

        # いずれかの入力がrequires_grad=Trueなら逆伝播関数を設定
        requires_grad = any(
            isinstance(arg, Tensor) and arg.requires_grad for arg in args
        )

        if requires_grad and isinstance(output, Tensor):
            # 逆伝播関数を設定
            def _backward():
                if output.grad is not None:
                    grads = cls.backward(ctx, output.grad)
                    if not isinstance(grads, tuple):
                        grads = (grads,)

                    # 入力に勾配を蓄積
                    tensor_idx = 0
                    for arg in args:
                        if isinstance(arg, Tensor) and arg.requires_grad:
                            if tensor_idx < len(grads) and grads[tensor_idx] is not None:
                                if arg.grad is None:
                                    arg.grad = grads[tensor_idx]
                                else:
                                    arg.grad = arg.grad + grads[tensor_idx]
                            tensor_idx += 1

            output._backward = _backward

        return output


class Context:
    """
    順伝播時の情報を保存し、逆伝播で使用するためのコンテキストオブジェクト
    PyTorchのctxオブジェクトと同様
    """

    def __init__(self):
        self.saved_tensors = []
        self.saved_values = {}

    def save_for_backward(self, *tensors):
        """逆伝播用にテンソルを保存"""
        self.saved_tensors = tensors

    def save_value(self, key: str, value: Any):
        """逆伝播用に任意の値を保存"""
        self.saved_values[key] = value
