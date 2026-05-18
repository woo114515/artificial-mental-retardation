'''
激活函数。目前只有 ReLU。
'''

from __future__ import annotations

import numpy as np


class ReLU:
    def __init__(self):
        self.z = None

    def forward(self, z):
        """
        z: 任意 shape 的 numpy array
        return: max(0, z)
        """
        self.z = z
        return np.maximum(0, z)

    def backward(self, dout):
        """
        dout: 上游传来的梯度，shape 与 z 相同
        return: dout * (z > 0)
        """
        if self.z is None:
            raise RuntimeError("Cannot call backward before forward.")

        return dout * (self.z > 0)

    def params_and_grads(self):
        """
        ReLU 没有可训练参数。
        """
        return []