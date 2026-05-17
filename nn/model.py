"""
负责把多个层组合成一个模型。

Sequential:
    按顺序执行 forward
    按反方向执行 backward
    收集所有可训练参数和梯度
"""


class Sequential:
    def __init__(self, layers):
        """
        layers: list of layers

        例如：
        [
            Linear(784, 256),
            ReLU(),
            Linear(256, 128),
            ReLU(),
            Linear(128, 10)
        ]
        """
        self.layers = layers

    def forward(self, X):
        """
        按顺序调用每一层的 forward。

        X: 输入数据
        return: 最后一层输出 logits
        """
        out = X

        for layer in self.layers:
            out = layer.forward(out)

        return out

    def backward(self, dout):
        """
        按反方向调用每一层的 backward。

        dout: loss 对模型输出的梯度
        return: loss 对模型输入的梯度
        """
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout

    def params_and_grads(self):
        """
        收集所有 layer 的可训练参数和对应梯度。

        return:
            [
                (W1, dW1),
                (b1, db1),
                ...
            ]
        """
        result = []

        for layer in self.layers:
            result.extend(layer.params_and_grads())

        return result