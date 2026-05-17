'''
激活函数。目前只有 ReLU。
'''

from __future__ import annotations

import numpy as np


class ReLU:
    def __init__(self):
        self.x = None

    def forward(self, x):
        """
        x: 任意 shape 的 numpy array
        return: max(0, x)
        """
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        """
        dout: 上游传来的梯度，shape 与 x 相同
        return: dout * (x > 0)
        """
        if self.x is None:
            raise RuntimeError("Cannot call backward before forward.")

        return dout * (self.x > 0)

    def params_and_grads(self):
        """
        ReLU 没有可训练参数。
        """
        return []