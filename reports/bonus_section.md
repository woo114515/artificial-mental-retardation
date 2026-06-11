# 加分项实现说明

本项目在基础 MLP 之外，额外实现了两个扩展方向：

1. 多数据集统一接口；
2. 基础 CNN 结构。

这两个部分的目标不是替代成熟深度学习框架，而是验证自研 NumPy 框架具有一定扩展能力。

## 1. 多数据集统一接口

项目在 `data/` 目录下支持：

```text
MNIST
Fashion-MNIST
CIFAR-10
```

三个数据模块都提供统一变量：

```python
X_train, y_train
X_val, y_val
X_test, y_test
```

并保持统一 row-major 格式：

```text
X.shape = (num_samples, num_features)
y.shape = (num_samples,)
```

不同数据集的输入维度如下：

| 数据集 | 原始图像 | 展平后维度 |
|---|---|---:|
| MNIST | `1 x 28 x 28` | 784 |
| Fashion-MNIST | `1 x 28 x 28` | 784 |
| CIFAR-10 | `3 x 32 x 32` | 3072 |

为了验证接口通用性，项目新增了：

```text
experiments/run_dataset_experiments.py
```

该脚本使用同一套 MLP 训练流程分别接入三个数据集，并保存：

```text
config.json
results.json
loss_curve.png
accuracy_curve.png
```

代表性接口验证结果如下：

| 数据集 | 模型 | Test Accuracy | 说明 |
|---|---|---:|---|
| MNIST | MLP | 0.9315 | 简单模型即可较好收敛 |
| Fashion-MNIST | MLP | 0.8565 | 数据更复杂，准确率低于 MNIST |
| CIFAR-10 | MLP | 0.3170 | 彩色自然图像更复杂，简单 MLP 表现有限 |

该实验主要说明：三种数据集可以通过统一接口进入同一训练流程。CIFAR-10 准确率较低并不代表接口失败，而是说明该数据集需要更强的模型结构和训练策略。

## 2. CNN 实现

在基础 MLP 之外，项目实现了 CNN 所需的核心层：

```text
Conv2D
MaxPool2D
Flatten
```

CNN 使用 NCHW 格式：

```text
X.shape = (batch_size, channels, height, width)
```

对于原本展平的数据，训练脚本会在进入 CNN 前进行形状转换。例如 MNIST 会从：

```text
(batch_size, 784)
```

转换为：

```text
(batch_size, 1, 28, 28)
```

典型 CNN 结构为：

```text
Conv2D -> ReLU -> MaxPool2D -> Flatten -> Linear -> ReLU -> Linear
```

其中：

- `Conv2D` 负责提取局部图像特征；
- `MaxPool2D` 负责降低空间尺寸并保留局部最大响应；
- `Flatten` 负责连接卷积特征和全连接分类器。

在 MNIST 上，CNN 可以正常完成前向传播、反向传播和参数更新，并取得约 98% 左右的验证/测试准确率。由于本项目使用纯 NumPy 实现，没有使用 GPU 加速、BatchNorm、Dropout 或数据增强，因此 CNN 部分主要作为框架扩展能力验证，而不是追求最高准确率。

## 3. 小结

多数据集接口说明项目的数据层具有可扩展性；CNN 实现说明项目的模型层不局限于全连接网络。通过这两个加分项，项目从基础 MLP 扩展到了更接近实际图像分类任务的结构和数据场景。

同时，这些实验也体现出模型复杂度与数据复杂度之间的关系：MNIST 较简单，MLP 已经可以取得较高准确率；Fashion-MNIST 和 CIFAR-10 更复杂，尤其 CIFAR-10 需要更深 CNN 和更多训练技巧才能取得较好效果。
