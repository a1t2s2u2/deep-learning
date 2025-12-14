"""
基本的な演算関数
活性化関数や数学関数など
"""

import numpy as np
from .tensor import Tensor


def relu(x: Tensor) -> Tensor:
    """
    ReLU活性化関数
    ReLU(x) = max(0, x)
    """
    out = Tensor(
        np.maximum(0, x.data),
        requires_grad=x.requires_grad,
        _children=(x,),
        _op='ReLU'
    )

    def _backward():
        if x.requires_grad:
            grad = out.grad.data * (x.data > 0)
            if x.grad is None:
                x.grad = Tensor(grad)
            else:
                x.grad = Tensor(x.grad.data + grad)

    out._backward = _backward
    return out


def sigmoid(x: Tensor) -> Tensor:
    """
    Sigmoid活性化関数
    σ(x) = 1 / (1 + e^(-x))
    """
    sig = 1 / (1 + np.exp(-x.data))
    out = Tensor(
        sig,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op='Sigmoid'
    )

    def _backward():
        if x.requires_grad:
            # sigmoid の微分: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
            grad = out.grad.data * sig * (1 - sig)
            if x.grad is None:
                x.grad = Tensor(grad)
            else:
                x.grad = Tensor(x.grad.data + grad)

    out._backward = _backward
    return out


def tanh(x: Tensor) -> Tensor:
    """
    Tanh活性化関数
    tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    """
    tanh_val = np.tanh(x.data)
    out = Tensor(
        tanh_val,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op='Tanh'
    )

    def _backward():
        if x.requires_grad:
            # tanh の微分: d/dx tanh(x) = 1 - tanh^2(x)
            grad = out.grad.data * (1 - tanh_val ** 2)
            if x.grad is None:
                x.grad = Tensor(grad)
            else:
                x.grad = Tensor(x.grad.data + grad)

    out._backward = _backward
    return out


def exp(x: Tensor) -> Tensor:
    """
    指数関数
    exp(x) = e^x
    """
    exp_val = np.exp(x.data)
    out = Tensor(
        exp_val,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op='Exp'
    )

    def _backward():
        if x.requires_grad:
            # 指数関数の微分: d/dx e^x = e^x
            grad = out.grad.data * exp_val
            if x.grad is None:
                x.grad = Tensor(grad)
            else:
                x.grad = Tensor(x.grad.data + grad)

    out._backward = _backward
    return out


def log(x: Tensor) -> Tensor:
    """
    自然対数
    log(x) = ln(x)
    """
    out = Tensor(
        np.log(x.data),
        requires_grad=x.requires_grad,
        _children=(x,),
        _op='Log'
    )

    def _backward():
        if x.requires_grad:
            # 対数の微分: d/dx ln(x) = 1/x
            grad = out.grad.data / x.data
            if x.grad is None:
                x.grad = Tensor(grad)
            else:
                x.grad = Tensor(x.grad.data + grad)

    out._backward = _backward
    return out


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """
    Softmax関数
    softmax(x)_i = exp(x_i) / sum(exp(x_j))

    数値安定性のためmax値を引く
    """
    # 数値安定性のため最大値を引く
    x_max = np.max(x.data, axis=axis, keepdims=True)
    exp_x = np.exp(x.data - x_max)
    softmax_val = exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    out = Tensor(
        softmax_val,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op='Softmax'
    )

    def _backward():
        if x.requires_grad:
            # Softmaxの勾配計算（ヤコビ行列を使用）
            # 通常はCrossEntropyLossと組み合わせて使うため、そこで最適化される
            grad = out.grad.data
            s = softmax_val

            # ヤコビ行列による勾配計算
            grad_input = s * (grad - np.sum(grad * s, axis=axis, keepdims=True))

            if x.grad is None:
                x.grad = Tensor(grad_input)
            else:
                x.grad = Tensor(x.grad.data + grad_input)

    out._backward = _backward
    return out
