"""
ニューラルネットワークのモジュール基底クラス
"""

from typing import Iterator, List
from ..core.parameter import Parameter


class Module:
    """
    全てのニューラルネットワークモジュールの基底クラス

    PyTorchのnn.Moduleと同様の機能を提供
    """

    def __init__(self):
        """Moduleの初期化"""
        self._parameters = {}
        self._modules = {}
        self.training = True

    def forward(self, *args, **kwargs):
        """
        順伝播を定義

        サブクラスで必ずオーバーライドする必要がある
        """
        raise NotImplementedError("forward() must be implemented")

    def __call__(self, *args, **kwargs):
        """
        モジュールを関数のように呼び出し可能にする

        内部でforward()を呼び出す
        """
        return self.forward(*args, **kwargs)

    def parameters(self) -> Iterator[Parameter]:
        """
        全てのパラメータを返すイテレータ

        Returns:
            パラメータのイテレータ
        """
        for param in self._parameters.values():
            yield param

        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self) -> Iterator[tuple]:
        """
        パラメータと名前のペアを返すイテレータ

        Returns:
            (名前, パラメータ)のタプルのイテレータ
        """
        for name, param in self._parameters.items():
            yield name, param

        for module_name, module in self._modules.items():
            for name, param in module.named_parameters():
                yield f"{module_name}.{name}", param

    def zero_grad(self):
        """全パラメータの勾配をゼロクリア"""
        for param in self.parameters():
            param.zero_grad()

    def train(self):
        """訓練モードに設定"""
        self.training = True
        for module in self._modules.values():
            module.train()

    def eval(self):
        """評価モードに設定"""
        self.training = False
        for module in self._modules.values():
            module.eval()

    def __setattr__(self, name: str, value):
        """
        属性設定時にParameterとModuleを自動登録

        Args:
            name: 属性名
            value: 属性値
        """
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value

        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
