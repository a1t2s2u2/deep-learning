"""
クロスエントロピー損失
"""

import numpy as np
from ..core.tensor import Tensor
from ..core.activations import softmax, log
from ..nn.module import Module


class CrossEntropyLoss(Module):
    """
    クロスエントロピー損失

    内部でSoftmaxを適用してからlog lossを計算
    数値安定性のため、log-softmaxを使用
    """

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        損失を計算

        Args:
            y_pred: ロジット（Softmax前の値） (batch_size, num_classes)
            y_true: ラベルインデックス (batch_size,) または one-hot (batch_size, num_classes)

        Returns:
            損失（スカラー）
        """
        batch_size = y_pred.shape[0]
        num_classes = y_pred.shape[1]

        # Log-Softmaxを計算（数値安定性のため）
        # log(softmax(x)) = x - log(sum(exp(x)))
        max_logits = np.max(y_pred.data, axis=1, keepdims=True)
        shifted_logits = y_pred.data - max_logits
        log_sum_exp = np.log(np.sum(np.exp(shifted_logits), axis=1, keepdims=True))
        log_softmax = shifted_logits - log_sum_exp

        log_softmax_tensor = Tensor(
            log_softmax,
            requires_grad=y_pred.requires_grad,
            _children=(y_pred,),
            _op='log_softmax'
        )

        # ラベルがインデックス形式の場合、one-hotに変換
        if y_true.ndim == 1:
            # one-hot encoding
            y_true_onehot = np.zeros((batch_size, num_classes), dtype=np.float32)
            y_true_onehot[np.arange(batch_size), y_true.data.astype(int)] = 1.0
            y_true = Tensor(y_true_onehot)

        # クロスエントロピー損失: -mean(sum(y_true * log_softmax))
        loss_val = -np.mean(np.sum(y_true.data * log_softmax, axis=1))
        loss = Tensor(
            loss_val,
            requires_grad=y_pred.requires_grad,
            _children=(log_softmax_tensor, y_true),
            _op='cross_entropy'
        )

        # 逆伝播の定義
        def _backward():
            if y_pred.requires_grad:
                # クロスエントロピーの勾配: (softmax(y_pred) - y_true) / batch_size
                softmax_val = np.exp(log_softmax)
                grad = (softmax_val - y_true.data) / batch_size

                if y_pred.grad is None:
                    y_pred.grad = Tensor(grad)
                else:
                    y_pred.grad = Tensor(y_pred.grad.data + grad)

        loss._backward = _backward
        return loss
