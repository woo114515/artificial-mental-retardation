# NumPy MLP MNIST

本项目使用 **Python + NumPy** 从零实现一个多层感知机（MLP），在 MNIST 手写数字数据集上完成训练与推理，并争取在测试集上达到 **95% 以上准确率**。

本项目不使用 PyTorch、TensorFlow 等自动微分框架。核心目标是手写实现：

- 前向传播
- 反向传播
- Softmax Cross Entropy Loss
- SGD / Momentum / Adam 优化器
- mini-batch 训练循环
- 训练曲线与超参数实验

---

## 1. 项目目标

本项目最终需要完成：

```text
1. 使用 NumPy 实现 MLP
2. 在 MNIST 测试集上达到 accuracy ≥ 95%
3. 实现清晰的 Layer / Loss / Optimizer / Model 模块
4. 完成训练曲线绘制
5. 完成实验报告
6. 报告中包含反向传播数学推导和超参数对比实验
```

主模型建议结构：

```text
784 → 256 → 128 → 10
```

含义：

```text
输入层：784 维，对应 28 × 28 图片展开
隐藏层 1：256 维
隐藏层 2：128 维
输出层：10 维，对应数字 0 到 9
```

---


## 2. 环境要求

推荐 Python 版本：

```text
Python 3.10+
```


### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

项目根目录包含依赖的python库：

```text
requirements.txt
```

暂时不加入：

```text
torch
torchvision
tensorflow
pandas
scikit-learn
```

推荐始终使用：

```bash
python -m pip install ...
```
安装依赖

---

## 3. 各模块职责

详见各文件夹下`README`

### 3.1 `data/`

负责数据读取、预处理和 mini-batch 生成。

### 3.2 `nn/`

负责神经网络核心数学。

### 3.3 `utils/`

负责训练辅助功能。

### 3.4 `experiments/`

负责超参数实验，例如：

### 3.5 `tests/`

负责模块测试。

---

## 4. 数据格式与 Shape 约定

本项目统一采用 **row-major convention**：

```text
X.shape = (num_samples, num_features)
```

也就是说：

```text
每一行是一个样本
每一列是一个特征
```

### 不要把 X 转置成 `(784, N)`

错误格式：

```text
X_train.shape = (784, 55000)
```

### 标签 y 使用整数标签，不使用 one-hot

本项目中，标签统一使用整数形式：

```text
y.shape = (num_samples,)
```

例如：

```text
y = [3, 7, 1, 0, 9]
```

含义：

```text
第 0 个样本是真实数字 3
第 1 个样本是真实数字 7
第 2 个样本是真实数字 1
```

---

## 5. MNIST 数据预处理要求

原始 MNIST 图片 shape：

```text
(num_samples, 28, 28)
```

需要展平成：

```text
(num_samples, 784)
```

详见`data/`




`SoftmaxCrossEntropyLoss.forward()` 接收：

```text
logits.shape = (batch_size, 10)
labels.shape = (batch_size,)
```

其中：

```text
labels[i] ∈ {0, 1, 2, ..., 9}
```

loss 计算公式是：

```text
loss = -mean(log(probability of the correct class))
```

数学形式：

```text
L = -1/n * sum_i log(P[i, y_i])
```

其中：

```text
P = softmax(logits)
```

不要把所有类别概率直接求和。

---

## 6. 当前主模型

第一版 baseline：

```text
784 → 128 → 10
```

正式主模型：

```text
784 → 256 → 128 → 10
```

即：

```text
Linear(784, 256)
ReLU
Linear(256, 128)
ReLU
Linear(128, 10)
SoftmaxCrossEntropyLoss
```

最后一个 Linear 后面不要加 ReLU。

---

## 7. 训练流程

完整训练流程如下：

```text
1. 加载 MNIST
2. 创建 DataLoader
3. 创建模型
4. 创建 loss
5. 创建 optimizer
6. 对每个 epoch:
      对每个 batch:
          logits = model.forward(X_batch)
          loss = criterion.forward(logits, y_batch)
          dlogits = criterion.backward()
          model.backward(dlogits)
          optimizer.step(model.params_and_grads())
      在 train / validation 上评估 accuracy
7. 测试集评估
8. 保存训练曲线
```

---

## 8. 运行测试

详见`test/`。可以使用`pytest`辅助。

## 10. Git 协作规范

见`Git.md`

## 11. 阶段性开发目标

### 阶段 1：`nn/` 主链路跑通

目标：

```text
用随机假数据完成：

model.forward(X)
loss.forward(logits, labels)
loss.backward()
model.backward(dlogits)
optimizer.step(model.params_and_grads())
```

不需要 MNIST。

---

### 阶段 2：`data/` 跑通

目标：

```text
X_train.shape = (55000, 784)
y_train.shape = (55000,)
X_val.shape = (5000, 784)
y_val.shape = (5000,)
X_test.shape = (10000, 784)
y_test.shape = (10000,)
```

---

### 阶段 3：完整训练跑通

目标：

```text
loss 能下降
accuracy 能上升
至少在小模型上跑通训练
```

第一版模型：

```text
784 → 128 → 10
```

---

### 阶段 4：准确率达标

目标：

```text
MNIST test accuracy ≥ 95%
```

推荐模型：

```text
784 → 256 → 128 → 10
```

---

### 阶段 5：实验与报告

完成：

```text
learning rate 对比
batch size 对比
网络深度对比
优化器对比
训练曲线
最终报告
```

---

## 29. 最终报告建议结构（AI写的，看看就好）

报告建议包含：

```text
1. 项目背景与目标
2. MNIST 数据集介绍
3. 数据预处理方法
4. MLP 网络结构
5. 前向传播公式
6. 反向传播推导
7. Softmax Cross Entropy 推导
8. 优化器设计
9. 代码模块设计
10. 实验设置
11. 训练曲线
12. 超参数对比实验
13. 测试集最终准确率
14. 总结与反思
```
