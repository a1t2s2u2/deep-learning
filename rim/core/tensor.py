"""
NumPyを使用したTensor実装
自動微分機能を持つ多次元配列
"""

from typing import Union, Tuple, Optional, List
import numpy as np


class Tensor:
    """
    自動微分をサポートする多次元配列クラス

    Attributes:
        data: NumPy配列
        requires_grad: 勾配計算の要否
        grad: 勾配テンソル
        _prev: 親テンソルの集合（計算グラフ用）
        _op: このテンソルを生成した演算
        _backward: 逆伝播関数
    """

    def __init__(
        self,
        data: Union[np.ndarray, list, float, int, 'Tensor'],
        requires_grad: bool = False,
        _children: Tuple['Tensor', ...] = (),
        _op: str = ''
    ):
        """
        Tensorの初期化

        Args:
            data: データ（NumPy配列、リスト、スカラー、または他のTensor）
            requires_grad: 勾配計算を行うかどうか
            _children: 親テンソル（自動微分用）
            _op: 演算名
        """
        if isinstance(data, Tensor):
            self.data = data.data.copy()
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        else:
            self.data = np.array(data, dtype=np.float32)

        self.requires_grad = requires_grad
        self.grad: Optional['Tensor'] = None

        # 計算グラフ
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    @property
    def shape(self) -> Tuple[int, ...]:
        """テンソルの形状"""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """次元数"""
        return self.data.ndim

    @property
    def size(self) -> int:
        """全要素数"""
        return self.data.size

    def item(self) -> float:
        """スカラー値を取得（1要素のテンソルのみ）"""
        return self.data.item()

    def zero_grad(self):
        """勾配をゼロクリア"""
        self.grad = None

    def backward(self, grad: Optional['Tensor'] = None):
        """
        逆伝播で勾配を計算

        Args:
            grad: 上流からの勾配（Noneの場合は1で初期化）
        """
        if not self.requires_grad:
            return

        # トポロジカルソート
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # 勾配の初期化
        if grad is None:
            self.grad = Tensor(np.ones_like(self.data))
        else:
            self.grad = grad

        # 逆向きに勾配を伝播
        for node in reversed(topo):
            node._backward()

    # ========== 演算 ==========

    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """加算"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op='+'
        )

        def _backward():
            if self.requires_grad:
                # ブロードキャストの逆操作
                grad = out.grad.data
                # 次元を合わせる
                ndims_added = grad.ndim - self.data.ndim
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                # 各軸でブロードキャストされた分をsumで戻す
                for i, dim in enumerate(self.data.shape):
                    if dim == 1:
                        grad = np.sum(grad, axis=i, keepdims=True)

                if self.grad is None:
                    self.grad = Tensor(grad)
                else:
                    self.grad = Tensor(self.grad.data + grad)

            if other.requires_grad:
                grad = out.grad.data
                ndims_added = grad.ndim - other.data.ndim
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(other.data.shape):
                    if dim == 1:
                        grad = np.sum(grad, axis=i, keepdims=True)

                if other.grad is None:
                    other.grad = Tensor(grad)
                else:
                    other.grad = Tensor(other.grad.data + grad)

        out._backward = _backward
        return out

    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """乗算"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op='*'
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad.data * other.data
                ndims_added = grad.ndim - self.data.ndim
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(self.data.shape):
                    if dim == 1:
                        grad = np.sum(grad, axis=i, keepdims=True)

                if self.grad is None:
                    self.grad = Tensor(grad)
                else:
                    self.grad = Tensor(self.grad.data + grad)

            if other.requires_grad:
                grad = out.grad.data * self.data
                ndims_added = grad.ndim - other.data.ndim
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(other.data.shape):
                    if dim == 1:
                        grad = np.sum(grad, axis=i, keepdims=True)

                if other.grad is None:
                    other.grad = Tensor(grad)
                else:
                    other.grad = Tensor(other.grad.data + grad)

        out._backward = _backward
        return out

    def __pow__(self, power: Union[int, float]) -> 'Tensor':
        """べき乗"""
        out = Tensor(
            self.data ** power,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f'**{power}'
        )

        def _backward():
            if self.requires_grad:
                grad = power * (self.data ** (power - 1)) * out.grad.data
                if self.grad is None:
                    self.grad = Tensor(grad)
                else:
                    self.grad = Tensor(self.grad.data + grad)

        out._backward = _backward
        return out

    def __neg__(self) -> 'Tensor':
        """負数"""
        return self * -1

    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """減算"""
        return self + (-other)

    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """除算"""
        return self * (other ** -1)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Tensor(other) - self

    def __rtruediv__(self, other):
        return Tensor(other) / self

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """行列積"""
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op='@'
        )

        def _backward():
            if self.requires_grad:
                # dL/dA = dL/dOut @ B^T
                grad = out.grad.data @ other.data.T
                if self.grad is None:
                    self.grad = Tensor(grad)
                else:
                    self.grad = Tensor(self.grad.data + grad)

            if other.requires_grad:
                # dL/dB = A^T @ dL/dOut
                grad = self.data.T @ out.grad.data
                if other.grad is None:
                    other.grad = Tensor(grad)
                else:
                    other.grad = Tensor(other.grad.data + grad)

        out._backward = _backward
        return out

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """総和"""
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='sum'
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad.data
                if not keepdims and axis is not None:
                    # 軸を復元
                    if isinstance(axis, int):
                        grad = np.expand_dims(grad, axis=axis)
                    else:
                        for ax in sorted(axis):
                            grad = np.expand_dims(grad, axis=ax)

                # ブロードキャストして元の形状に戻す
                grad = np.broadcast_to(grad, self.data.shape)

                if self.grad is None:
                    self.grad = Tensor(grad)
                else:
                    self.grad = Tensor(self.grad.data + grad)

        out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """平均"""
        n = self.data.size if axis is None else np.prod([self.data.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))])
        return self.sum(axis=axis, keepdims=keepdims) / n

    def reshape(self, *shape) -> 'Tensor':
        """形状変更"""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]

        out = Tensor(
            self.data.reshape(shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='reshape'
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad.data.reshape(self.data.shape)
                if self.grad is None:
                    self.grad = Tensor(grad)
                else:
                    self.grad = Tensor(self.grad.data + grad)

        out._backward = _backward
        return out

    def transpose(self, *axes) -> 'Tensor':
        """転置"""
        out = Tensor(
            self.data.transpose(*axes) if axes else self.data.T,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='T'
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad.data.transpose(*axes) if axes else out.grad.data.T
                if self.grad is None:
                    self.grad = Tensor(grad)
                else:
                    self.grad = Tensor(self.grad.data + grad)

        out._backward = _backward
        return out

    @property
    def T(self) -> 'Tensor':
        """転置"""
        return self.transpose()

    @staticmethod
    def zeros(*shape) -> 'Tensor':
        """ゼロテンソル"""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return Tensor(np.zeros(shape, dtype=np.float32))

    @staticmethod
    def ones(*shape) -> 'Tensor':
        """1で埋めたテンソル"""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return Tensor(np.ones(shape, dtype=np.float32))

    @staticmethod
    def randn(*shape) -> 'Tensor':
        """標準正規分布からサンプリング"""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return Tensor(np.random.randn(*shape).astype(np.float32))

    @staticmethod
    def rand(*shape) -> 'Tensor':
        """[0, 1)の一様分布からサンプリング"""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return Tensor(np.random.rand(*shape).astype(np.float32))

    def __repr__(self) -> str:
        grad_str = f", grad_fn=<{self._op}>" if self._op else ""
        return f"Tensor({self.data}{grad_str})"

    def __str__(self) -> str:
        return f"Tensor({self.data})"
