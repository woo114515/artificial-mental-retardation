'''
激活函数。目前只有 ReLU。
'''

from __future__ import annotations

import numpy as np

from .layer import layer

class ReLU(layer):
    def __init__(self):
        super().__init__()

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
    
class Sigmoid(layer):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        """
        z: 任意 shape 的 numpy array
        """
        self.z = z
        self.out = 1 / (1 + np.exp(-z))
        return self.out

    def backward(self, dout):
        """
        dout: 上游传来的梯度，shape 与 z 相同
        """
        if self.z is None:
            raise RuntimeError("Cannot call backward before forward.")

        return dout * self.out * (1 - self.out)

    def params_and_grads(self):
        """
        Sigmod 没有可训练参数。
        """
        return []
