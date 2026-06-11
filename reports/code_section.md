# 工程实现与代码结构

本项目以“从零实现神经网络”为目标，整体采用模块化设计。代码按照数据处理、神经网络核心组件、训练评估工具和实验脚本进行拆分，尽量模仿深度学习框架中常见的 `Layer / Loss / Optimizer / Model` 组织方式，同时保持实现简单，便于理解反向传播的实际过程。

## 1. 项目结构

主要目录如下：

```text
data/          数据读取、预处理、mini-batch 生成
nn/            神经网络核心模块
utils/         训练辅助工具
experiments/   超参数实验与扩展实验
tests/         单元测试
reports/       报告与实验分析
```

核心入口脚本包括：

| 文件 | 作用 |
|---|---|
| `train.py` | 主训练脚本，支持 MLP / CNN、不同数据集和不同优化器 |
| `evaluate.py` | 加载 checkpoint，对测试集进行整体评估并输出指标 |
| `visualize_predictions.py` | 可视化测试集预测结果 |
| `demo_handwritten_digit.py` | 手写数字 demo，用于展示模型推理效果 |
| `config.py` | 集中管理模型、数据集和训练超参数 |

## 2. 数据模块

`data/` 目录负责数据读取和格式统一。目前支持：

```text
mnist
fashionmnist
cifar10
```

三个数据模块都暴露统一变量：

```python
X_train, y_train
X_val, y_val
X_test, y_test
```

其中：

```text
X.shape = (num_samples, num_features)
y.shape = (num_samples,)
```

MNIST 和 Fashion-MNIST 原始图像大小为 `28 x 28`，展平后输入维度为 `784`；CIFAR-10 是三通道彩色图像，大小为 `3 x 32 x 32`，展平后输入维度为 `3072`。

`data.dataloader.create_mini_batches()` 用于训练阶段 mini-batch 划分和 shuffle：

```python
for X_batch, y_batch in create_mini_batches(X_train, y_train, batch_size):
    ...
```

对于 CNN，`data.transforms.to_nchw_images()` 会将展平图像恢复为：

```text
(num_samples, channels, height, width)
```

这样 MLP 和 CNN 可以共用同一套数据读取接口，只在进入模型前进行形状转换。

## 3. 神经网络核心模块

`nn/` 目录实现神经网络的核心数学组件。

### 3.1 Layer 抽象

项目中的层都遵循相同接口：

```python
forward(X)
backward(dout)
params_and_grads()
```

其中：

- `forward()` 负责前向传播；
- `backward()` 负责反向传播，并返回传给前一层的梯度；
- `params_and_grads()` 返回可训练参数及其梯度，供优化器更新。

这种设计使不同层可以被 `Sequential` 统一组织。

### 3.2 Linear 层

`nn.linear.Linear` 实现全连接层：

```text
Z = XW + b
```

其反向传播保存：

```text
dW = X.T @ dZ
db = sum(dZ, axis=0)
dX = dZ @ W.T
```

这是 MLP 的核心模块，也是理解手写反向传播最重要的部分。

### 3.3 激活函数

`nn.activations` 实现：

```text
ReLU
Sigmoid
```

激活函数没有可训练参数，因此 `params_and_grads()` 返回空列表。它们只负责在前向传播中引入非线性，并在反向传播中根据导数筛选或缩放梯度。

### 3.4 损失函数

`nn.losses.SoftmaxCrossEntropyLoss` 负责分类损失计算。它接收：

```text
logits.shape = (batch_size, num_classes)
labels.shape = (batch_size,)
```

前向传播输出 scalar loss，反向传播输出：

```text
dlogits.shape = (batch_size, num_classes)
```

标签使用整数类别，不需要在数据模块中提前转换为 one-hot。

### 3.5 优化器

`nn.optimizers` 实现：

```text
SGD
Momentum
Adam
```

优化器不关心模型结构，只遍历模型提供的 `(param, grad)` 列表并更新参数。这使得同一个优化器可以同时用于 MLP 和 CNN。

### 3.6 Sequential 模型容器

`nn.model.Sequential` 用于按顺序组合多个层：

```python
model = Sequential([
    Linear(784, 128),
    ReLU(),
    Linear(128, 10),
])
```

前向传播时，`Sequential.forward()` 按顺序调用每一层；反向传播时，`Sequential.backward()` 按相反顺序调用每一层。这对应链式法则在代码中的实现。

## 4. CNN 扩展模块

除了 MLP，项目还实现了基础 CNN 组件：

```text
Conv2D
MaxPool2D
Flatten
```

CNN 输入格式为：

```text
(batch_size, channels, height, width)
```

典型结构为：

```text
Conv2D -> ReLU -> MaxPool2D -> Flatten -> Linear -> ReLU -> Linear
```

其中：

- `Conv2D` 用于提取局部空间特征；
- `MaxPool2D` 用于降低空间尺寸并保留局部最大响应；
- `Flatten` 将四维特征图转换为二维矩阵，方便接入全连接层。

CNN 部分主要用于验证自研框架可以从 MLP 扩展到卷积结构。由于本项目使用纯 NumPy 实现，没有使用 GPU 加速、BatchNorm 或复杂数据增强，因此 CNN 实验重点放在结构正确性和训练流程打通，而不是追求工业级准确率。

## 5. 训练流程

`train.py` 是主训练入口，整体流程如下：

1. 根据 `config.py` 选择数据集；
2. 根据 `MODEL_TYPE` 构建 MLP 或 CNN；
3. 创建损失函数和优化器；
4. 使用 mini-batch 训练一个 epoch；
5. 在训练集和验证集上计算 loss / accuracy；
6. 记录训练历史并绘制曲线；
7. 在测试集上评估最终准确率；
8. 保存 checkpoint。

单个 mini-batch 的训练步骤为：

```text
forward -> loss -> backward -> optimizer.step
```

这对应深度学习训练中最基本的闭环。

## 6. 评估、绘图与 checkpoint

`utils.metrics.evaluate()` 用于非 shuffle 的批量评估。它可以只计算 accuracy，也可以在传入 criterion 时同时计算平均 loss。

`utils.plot.plot_history()` 会保存：

```text
loss_curve.png
accuracy_curve.png
```

曲线保存路径会包含关键超参数，方便后续整理实验结果。

`utils.checkpoint` 提供：

```text
save_checkpoint()
load_checkpoint()
```

checkpoint 使用 `.npz` 格式保存模型参数，并额外保存超参数 metadata。训练完成时会保存 `latest_model.npz`，同时也会按照超参数生成命名 checkpoint。若训练过程中使用 `Ctrl+C` 中断，也会保存当前模型，避免长时间训练结果丢失。

## 7. 实验脚本

`experiments/` 目录保存实验脚本和结果。

主要实验包括：

| 脚本或目录 | 内容 |
|---|---|
| `run_experiments.py` | 学习率、优化器、Momentum、Adam 等实验 |
| `liang_run_experiments.py` | batch size、隐藏层宽度/深度、激活函数等实验 |
| `run_dataset_experiments.py` | MNIST、Fashion-MNIST、CIFAR-10 数据接口验证实验 |
| `experiments/*/*/config.json` | 每组实验配置 |
| `experiments/*/*/results.json` | 每组实验指标 |
| `experiments/*/*/loss_curve.png` | loss 曲线 |
| `experiments/*/*/accuracy_curve.png` | accuracy 曲线 |

实验结果以 JSON、CSV 和图片形式保存，便于在报告中引用。

## 8. 测试与可靠性

项目在 `tests/` 目录下为核心模块编写了单元测试，覆盖：

```text
Linear / ReLU / Sigmoid
SoftmaxCrossEntropyLoss
SGD / Momentum / Adam
Sequential
Conv2D / MaxPool2D / Flatten
数据读取与数据变换
metrics / plot / checkpoint
预测可视化与 demo
```

测试的作用是确保每个模块在单独使用时行为正确，并降低后续重构带来的风险。尤其对于手写反向传播，单元测试可以帮助及时发现 shape 错误、梯度方向错误或参数更新错误。

## 9. 实现特点与不足

本项目的主要特点是：

1. 只依赖 NumPy 实现神经网络核心计算；
2. 使用清晰的模块划分，便于理解每一部分职责；
3. 统一 row-major 数据格式，减少维度混乱；
4. 支持 MLP、基础 CNN、多优化器和多数据集接口；
5. 实验结果自动保存，便于复现实验和撰写报告。

不足之处包括：

1. 纯 NumPy CNN 训练速度较慢，没有 GPU 加速；
2. 尚未实现 BatchNorm、Dropout 和 weight decay；
3. CIFAR-10 上简单模型准确率较低，说明更复杂数据集需要更强模型和更多训练技巧；
4. 当前框架是教学性质实现，重点在理解原理，而不是替代成熟深度学习框架。

总体而言，本项目完成了从数据读取、模型搭建、前向传播、反向传播、参数更新到实验分析的完整流程，能够支撑对深度学习基本原理和工程实现的学习。
