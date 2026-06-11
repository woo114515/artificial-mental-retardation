# NumPy 手写神经网络

本项目是《人工智能基础》大作业选题二：使用 **Python + NumPy** 从零实现神经网络，完成 MNIST 手写数字分类，并通过实验理解前向传播、反向传播、优化器、模型复杂度和泛化能力。

项目不使用 PyTorch、TensorFlow、JAX 或自动微分框架。核心目标是把神经网络训练流程中的关键组件亲手实现出来。

## 已实现功能

- MLP：`Linear -> Activation -> Linear`
- CNN：`Conv2D -> Activation -> MaxPool2D -> Flatten -> Linear`
- 激活函数：`ReLU`、`Sigmoid`
- 损失函数：`SoftmaxCrossEntropyLoss`
- 优化器：`SGD`、`Momentum`、`Adam`
- 初始化：普通随机初始化、Xavier、He
- 数据集接口：`MNIST`、`Fashion-MNIST`、`CIFAR-10`
- 训练、评估、checkpoint 保存/加载
- loss / accuracy 曲线绘制
- 测试集预测可视化与错误样本可视化
- 浏览器手写数字 demo
- 单元测试与实验报告

MNIST MLP 实验中，代表性配置 `hidden_dims=[256]` 可以达到约 `98.10%` 测试准确率，满足作业要求的 `accuracy >= 95%`。

## 环境安装

推荐 Python 版本：

```text
Python 3.10+
```

创建虚拟环境并安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

依赖保持轻量：

```text
numpy
matplotlib
pytest
```

## 项目结构

```text
data/          数据读取、预处理、mini-batch 生成
nn/            神经网络核心模块
utils/         训练辅助工具
experiments/   超参数实验与扩展实验
tests/         单元测试
reports/       报告与实验分析
```

核心脚本：

| 文件 | 作用 |
|---|---|
| `train.py` | 主训练入口 |
| `evaluate.py` | 加载 checkpoint 并评估测试集 |
| `visualize_predictions.py` | 可视化预测结果 |
| `demo_handwritten_digit.py` | 浏览器手写数字 demo |
| `config.py` | 统一配置模型、数据集和超参数 |

## 数据格式约定

本项目统一采用 row-major 约定：

```text
X.shape = (num_samples, num_features)
y.shape = (num_samples,)
logits.shape = (batch_size, num_classes)
```

标签使用整数类别，不使用 one-hot 作为数据输入格式。

对于 CNN，展平图像会在进入模型前转换为 NCHW：

```text
(num_samples, channels, height, width)
```

不同数据集的输入维度：

| 数据集 | 原始图像 | 展平后维度 |
|---|---|---:|
| MNIST | `1 x 28 x 28` | 784 |
| Fashion-MNIST | `1 x 28 x 28` | 784 |
| CIFAR-10 | `3 x 32 x 32` | 3072 |

## 配置说明

主要超参数集中在 `config.py`：

```python
DATASET = "mnist"          # "mnist", "fashionmnist", "cifar10"
MODEL_TYPE = "mlp"         # "mlp" or "cnn"
HIDDEN_DIMS = [128]
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 0.01
OPTIMIZER = "sgd"          # "sgd", "momentum", "adam"
ACTIVATION = "ReLU"        # "ReLU", "Sigmoid"
WEIGHT_INIT = "he"
RANDOM_SEED = 42
```

CNN 相关参数也在 `config.py` 中：

```python
CNN_OUT_CHANNELS = 8
CNN_KERNEL_SIZE = 3
CNN_STRIDE = 1
CNN_PADDING = 1
POOL_KERNEL_SIZE = 2
POOL_STRIDE = 2
```

训练前可根据实验需要修改这些配置。

## 运行训练

```bash
python train.py
```

训练过程中会输出每个 epoch 的：

```text
train_loss
val_loss
train_acc
val_acc
```

训练结束后会输出测试集准确率，并保存：

```text
experiments/records/.../loss_curve.png
experiments/records/.../accuracy_curve.png
checkpoints/latest_model.npz
```

同时会根据超参数保存一份命名 checkpoint。若训练中途使用 `Ctrl+C` 中断，也会自动保存当前模型。

## 评估 checkpoint

默认评估最新 checkpoint：

```bash
python evaluate.py
```

也可以指定 checkpoint：

```bash
python evaluate.py --checkpoint checkpoints/latest_model.npz
```

评估脚本会输出整体测试指标，并保存若干识别错误样本到：

```text
experiments/predictions/error/
```

## 预测可视化与手写 demo

可视化测试集预测：

```bash
python visualize_predictions.py
```

启动浏览器手写数字 demo：

```bash
python demo_handwritten_digit.py
```

demo 默认加载：

```text
checkpoints/latest_model.npz
```

## 实验脚本

超参数实验：

```bash
python experiments/run_experiments.py
```

多数据集接口验证实验：

```bash
python experiments/run_dataset_experiments.py
```

快速验证多数据集接口：

```bash
python experiments/run_dataset_experiments.py --epochs 1 --train-limit 128 --val-limit 64 --test-limit 64
```

实验结果通常保存为：

```text
config.json
results.json
loss_curve.png
accuracy_curve.png
```

## 测试

运行全部测试：

```bash
python -m pytest tests/
```

测试覆盖：

- `Linear / ReLU / Sigmoid`
- `SoftmaxCrossEntropyLoss`
- `SGD / Momentum / Adam`
- `Sequential`
- `Conv2D / MaxPool2D / Flatten`
- 数据读取与 shape 变换
- metrics、plot、checkpoint
- 预测可视化与 demo

## 实验报告

最终报告主体：

```text
reports/final_report.md
```

拆分章节：

```text
reports/math_section.md
reports/code_section.md
reports/bonus_section.md
reports/experiment_section.md
```

其中 `final_report.md` 已整合数学推导、工程实现、实验分析、复杂度分析和加分项说明。

## 数据说明

MNIST 和 Fashion-MNIST 原始数据位于：

```text
data/raw/mnist/
data/raw/fashionmnist/
```

CIFAR-10 需要将官方 `cifar-10-python.tar.gz` 放到：

```text
data/raw/cifar10/cifar-10-python.tar.gz
```

仓库中保留了下载路径说明：

```text
data/raw/cifar10/downloadpath.md
```

## 结果概览

代表性 MNIST MLP 结果：

| 配置 | Test Accuracy |
|---|---:|
| `hidden_dims=[128]` | 约 97.75% |
| `hidden_dims=[256]` | 约 98.10% |

多数据集接口验证结果：

| 数据集 | 模型 | Test Accuracy | 说明 |
|---|---|---:|---|
| MNIST | MLP | 0.9315 | 简单模型即可较好收敛 |
| Fashion-MNIST | MLP | 0.8565 | 数据更复杂，准确率低于 MNIST |
| CIFAR-10 | MLP | 0.3170 | 简单 MLP 难以处理彩色自然图像 |

CNN 在 MNIST 上可以正常训练并达到约 98% 左右准确率。由于使用纯 NumPy 实现，没有 GPU 加速、BatchNorm、Dropout 或数据增强，CNN 部分主要用于验证框架扩展能力。

## 项目定位

本项目是教学性质的深度学习框架实现，重点在于理解：

- 神经网络前向传播如何计算；
- 反向传播如何逐层传递梯度；
- 优化器如何更新参数；
- 超参数如何影响训练曲线和泛化能力；
- 模型复杂度和数据复杂度之间的关系。

它不是为了替代成熟深度学习框架，而是帮助我们理解这些框架背后的基本矩阵运算和工程组织方式。
