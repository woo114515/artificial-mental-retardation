# 神经网络数学原理与反向传播推导

本项目使用 NumPy 从零实现神经网络。代码中统一采用 row-major 数据约定，即每一行表示一个样本：

```text
X.shape = (B, D)
y.shape = (B,)
logits.shape = (B, C)
```

其中 `B` 为 batch size，`D` 为输入维度，`C` 为类别数。以 MNIST 为例，`D=784`，`C=10`。

## 1. 前向传播

全连接层的前向传播为：

\[
Z = XW + b
\]

其中：

| 符号 | Shape |
|---|---|
| \(X\) | \((B, D_{in})\) |
| \(W\) | \((D_{in}, D_{out})\) |
| \(b\) | \((D_{out},)\) |
| \(Z\) | \((B, D_{out})\) |

隐藏层输出再经过激活函数：

\[
A = g(Z)
\]

本项目主要使用 ReLU 和 Sigmoid：

\[
\operatorname{ReLU}(z)=\max(0,z)
\]

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

激活函数的作用是引入非线性，否则多层线性层叠加后仍然等价于一个线性模型。

## 2. Softmax Cross Entropy

最后一层输出 logits：

\[
S.shape=(B,C)
\]

Softmax 将 logits 转换为类别概率：

\[
P_{ij}=\frac{e^{S_{ij}}}{\sum_{k=1}^{C}e^{S_{ik}}}
\]

实际实现中，为了避免指数溢出，会先减去每一行的最大值：

\[
\tilde{S}_{ij}=S_{ij}-\max_k S_{ik}
\]

交叉熵损失为：

\[
L=-\frac{1}{B}\sum_{i=1}^{B}\log P_{i,y_i}
\]

其中 \(y_i\) 是第 \(i\) 个样本的真实类别。本项目标签保持为整数形式，不使用 one-hot 作为输入格式。

## 3. Softmax Cross Entropy 梯度

Softmax 与交叉熵组合后，损失对 logits 的梯度可以简化为：

\[
\frac{\partial L}{\partial S}=\frac{P-Y}{B}
\]

其中 \(Y\) 是标签对应的 one-hot 矩阵。代码中没有显式构造完整 \(Y\)，而是直接在正确类别位置减 1：

```python
dlogits = probs.copy()
dlogits[np.arange(batch_size), labels] -= 1
dlogits /= batch_size
```

因此：

\[
dS.shape=(B,C)
\]

该梯度会作为最后一层 Linear 的上游梯度继续反向传播。

## 4. Linear 层反向传播

Linear 层前向传播为：

\[
Z=XW+b
\]

假设上一层传来的梯度为：

\[
dZ=\frac{\partial L}{\partial Z}
\]

则 Linear 层需要计算三个梯度：

\[
dW=\frac{\partial L}{\partial W},\quad
db=\frac{\partial L}{\partial b},\quad
dX=\frac{\partial L}{\partial X}
\]

矩阵形式为：

\[
dW=X^T dZ
\]

\[
db=\sum_{i=1}^{B} dZ_i
\]

\[
dX=dZ W^T
\]

形状对应关系如下：

| 梯度 | Shape |
|---|---|
| \(dZ\) | \((B, D_{out})\) |
| \(dW\) | \((D_{in}, D_{out})\) |
| \(db\) | \((D_{out},)\) |
| \(dX\) | \((B, D_{in})\) |

对应代码为：

```python
self.dW = self.X.T @ dout
self.db = np.sum(dout, axis=0)
self.dX = dout @ self.W.T
```

这也是本项目手写反向传播的核心。

## 5. 单隐藏层 MLP 的 Shape 流程

以基线模型为例：

```text
784 -> 128 -> 10
```

前向传播 shape 如下：

| 步骤 | Shape |
|---|---|
| \(X\) | \((B, 784)\) |
| \(W_1\) | \((784, 128)\) |
| \(Z_1=XW_1+b_1\) | \((B, 128)\) |
| \(A_1=\operatorname{ReLU}(Z_1)\) | \((B, 128)\) |
| \(W_2\) | \((128, 10)\) |
| \(S=A_1W_2+b_2\) | \((B, 10)\) |
| \(P=\operatorname{softmax}(S)\) | \((B, 10)\) |
| \(L\) | scalar |

反向传播从：

\[
dS=\frac{P-Y}{B}
\]

开始，然后依次经过第二个 Linear 层、激活函数和第一个 Linear 层。`Sequential.backward()` 按前向传播的相反顺序调用每一层的 `backward()`，从而实现链式法则。

## 6. 参数更新

反向传播完成后，每个可训练层会提供参数和梯度：

```text
(W, dW)
(b, db)
```

优化器根据这些梯度更新参数。

SGD：

\[
\theta := \theta - \eta \nabla_\theta L
\]

Momentum：

\[
v := \mu v - \eta \nabla_\theta L
\]

\[
\theta := \theta + v
\]

Adam：

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

\[
\theta:=\theta-\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
\]

其中 \(g_t\) 表示当前梯度，\(\eta\) 是学习率。

## 7. 小结

本项目的训练过程可以概括为：

1. 前向传播计算 logits 和 loss；
2. Softmax Cross Entropy 给出初始梯度 \(dS=(P-Y)/B\)；
3. 各层根据链式法则反向计算梯度；
4. 优化器使用梯度更新参数；
5. 重复 mini-batch 训练，逐步降低损失并提高准确率。

通过手写这些过程，可以更直观地理解深度学习框架中 `loss.backward()` 背后的矩阵运算。
