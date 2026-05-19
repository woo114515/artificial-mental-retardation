"""
负责参数更新。
目前只有 SGD (Stochastic Gradient Descent)。
"""

from __future__ import annotations

import numpy as np

class Optimizer:
    def __init__(self, lr:float = 0.01):
        pass

    def step(self):
        pass

class SGD(Optimizer):
    """
    lr: learning rate，学习率
    param = param - lr * grad
    """
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, params_and_grads):
        """
        params_and_grads:
            [
                (param1, grad1),
                (param2, grad2),
                ...
            ]

        """
        for param, grad in params_and_grads:
            if grad is None:
                continue

            param[...] -= self.lr * grad
class Momentum(Optimizer):
    '''
    v = momentum * v - lr * grad
    param = param + v
    '''
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}

    def step(self, params_and_grads):
        for param, grad in params_and_grads:
            if grad is None:
                continue

            key = id(param)

            if key not in self.velocities:
                self.velocities[key] = np.zeros_like(param)

            v = self.velocities[key]
            v[...] = self.momentum * v - self.lr * grad
            param[...] += v

class Adam(Optimizer):
    '''
    m: 一阶矩估计，类似 Momentum，记录梯度的指数滑动平均
    v: 二阶矩估计，记录梯度平方的指数滑动平均
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad^2

    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)

    param = param - lr * m_hat / (sqrt(v_hat) + eps)
    '''
    def __init__(self, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = {}
        self.v = {}
        self.eps = eps
        self.t = 0

    def step(self, params_and_grads):
        self.t += 1

        for param, grad in params_and_grads:
            if grad is None:
                continue

            key = id(param)

            if key not in self.m:
                self.m[key] = np.zeros_like(param)
                self.v[key] = np.zeros_like(param)
            
            self.m[key][...] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key][...] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            param[...] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


    