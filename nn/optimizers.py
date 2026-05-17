"""
负责参数更新。
目前只有 SGD (Stochastic Gradient Descent)。
"""

from __future__ import annotations


class SGD:
    def __init__(self, lr: float = 0.01):
        """
        lr: learning rate，学习率
        """
        self.lr = lr

    def step(self, params_and_grads):
        """
        params_and_grads:
            [
                (param1, grad1),
                (param2, grad2),
                ...
            ]

        原地更新：
            param = param - lr * grad
        """
        for param, grad in params_and_grads:
            if grad is None:
                continue

            param[...] -= self.lr * grad