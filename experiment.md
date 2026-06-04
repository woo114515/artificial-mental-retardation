# 超参数实验
---

## 1. 主模型

### 1.1 Model Architecture

主模型采用单隐藏层 MLP：

```text
784 -> 128 -> 10
```

其中：

- 输入维度：784，对应展平后的 MNIST 图片；
- 隐藏层维度：128；
- 输出维度：10，对应数字类别 0 到 9；
- 隐藏层激活函数：ReLU；
- 损失函数：Softmax Cross Entropy；
- 优化器：Mini-batch SGD。

对应`config,py`
```Python
'''
所有的超参数存放在这里。
'''

'''
由数据集确定的参量
'''
INPUT_DIM = 784 # 输入维数
NUM_CLASSES = 10 # 输出维数

'''
实验时需要调节的参量
'''
# 基础参量
HIDDEN_DIMS = [128] # 隐藏层的结构。如[128]代表只有一层128维的隐藏层
BATCH_SIZE = 64 # 详见mini-batch gardient descent
NUM_EPOCHS = 100 # 总训练次数
LEARNING_RATE = 0.01 # 学习率

# 优化器
OPTIMIZER = "sgd" # 可选：'sgd', 'momentum', 'adam'
MOMENTUM = 0.9
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8

# 激活函数
ACTIVATION = "ReLU" # 可选:'ReLU', 'Sigmoid'

# 参数初始化
WEIGHT_INIT = "he" # 初始权重的方式
RANDOM_SEED = 42 # 初始权重的随机种子
```

## 2. 记录指标

每组实验记录以下指标：

| 指标 | 描述 |
|---|---|
| Train Loss | 训练集上的平均损失 |
| Train Accuracy | 训练集准确率 |
| Validation Loss | 验证集上的平均损失 |
| Validation Accuracy | 验证集准确率 |
| Test Accuracy | 测试集最终准确率 |
| Convergence Speed | 收敛速度，记录达到 达到95%，97% 验证准确率所需 epoch 数 |
| Stability | 训练过程中 loss 是否震荡、是否出现梯度爆炸或 `nan` |
| training time | 训练准确率达 97% 模型所需的时间 |

关注：

- loss 是否稳定下降；
- train accuracy 和 validation accuracy 是否差距过大(是否存在明显过拟合；)
- 不同超参数对收敛速度和最终准确率的影响。

## 3. 实验规则

为了保证实验对比公平，每次实验只改变一个主要变量，其他配置尽量保持不变。

固定项：

```python
INPUT_DIM = 784
NUM_CLASSES = 10
RANDOM_SEED = 42
LOSS = "softmax_cross_entropy"
DATASET = "MNIST"
```

除非当前实验专门研究某个变量，否则保持主模型配置不变。

---

## 4. 数据记录 

***注意： 接下来的表格中数据仅供参考，请做出充足数量的实验***

### 4.1 学习率(LEARNING_RATE) [xu]

#### 4.1.1 预期

- `lr = 0.001`：训练稳定，但收敛较慢；
- `lr = 0.01`：较稳定，可能达到较好结果；
- `lr = 0.05`：可能收敛较快；
- `lr = 0.1`：如果实现正确且数据预处理合适，可能收敛很快，但也可能震荡。

#### 4.1.2 结果

| Learning Rate | Train Acc | Val Acc | Test Acc | Final Train Loss | Final Val Loss | Notes |
|---:|---:|---:|---:|---:|---:|---|
| 0.001 |  |  |  |  |  |  |
| 0.01 |  |  |  |  |  |  |
| 0.05 |  |  |  |  |  |  |
| 0.1 |  |  |  |  |  |  |

### 4.2 实验二: Batch Size(BATCH_SIZE) [Liang]

#### 4.2.1 预期

- 较小 batch size 可能训练波动更大；
- 较大 batch size 训练曲线可能更平滑；
- `batch_size = 64` 或 `128` 通常是 MNIST MLP 的合理选择。

#### 4.2.3 结果
| Batch Size | Train Acc | Val Acc | Test Acc | Final Train Loss | Final Val Loss | Notes |
|---:|---:|---:|---:|---:|---:|---|
| 32 |  |  |  |  |  |  |
| 64 |  |  |  |  |  |  |
| 128 |  |  |  |  |  |  |
| 256 |  |  |  |  |  |  |

### 4.3 实验三: 隐藏层宽度(HIDDEN_DIMS) [Liang]

#### 4.3.1 预期

- `[64]`：参数较少，训练快，但表达能力可能有限；
- `[128]`：主模型配置，通常足够达到 95%；
- `[256]`：表达能力更强，可能提升准确率；
- `[512]`：参数更多，但收益可能变小，也可能更容易过拟合。

#### 4.3.2 结果

| Hidden Dims | Train Acc | Val Acc | Test Acc | Final Train Loss | Final Val Loss | Notes |
|---|---:|---:|---:|---:|---:|---|
| [64] |  |  |  |  |  |  |
| [128] |  |  |  |  |  |  |
| [256] |  |  |  |  |  |  |
| [512] |  |  |  |  |  |  |

### 4.4. 实验四：激活函数(ACTIVATION) [Liang]

#### 4.4.1 预期

- ReLU 计算简单，正区间梯度为 1，通常收敛更快；
- Sigmoid 在输入绝对值较大时容易饱和，可能导致梯度变小，训练较慢；
- 对于隐藏层，ReLU 通常比 Sigmoid 更适合作为默认选择。

#### 4.4.2 结果

| Activation | Weight Init | Train Acc | Val Acc | Test Acc | Final Train Loss | Final Val Loss | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| ReLU | He |  |  |  |  |  |  |
| Sigmoid | Xavier |  |  |  |  |  |  |

### 4.5 实验五：优化器 [Xu]

#### 4.5.1 设置

注意：不同优化器可能有不同数量的参数要调

`sgd` 对应 `LEARNING_RATE`

`Momentum` 对应 `LEARNING_RATE` `MOMENTUM`

`Adam` 对应 `LEARNING_RATE` `BETA1` `BETA2`

请做出充足数量的实验

### 8.3 预期

- SGD 原理最简单，但对学习率比较敏感；
- Momentum 可能比 SGD 收敛更快，并减少震荡；
- Adam 通常收敛更稳定，对学习率不那么敏感。

#### 4.5.2 结果

| Optimizer | Arguments | Train Acc | Val Acc | Test Acc | Final Train Loss | Final Val Loss | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| SGD |  |  |  |  |  |  |  |
| Momentum1 |  |  |  |  |  |  |  |
| Momentum2 |  |  |  |  |  |  |  |
| Momentum3 |  |  |  |  |  |  |  |
| Adam1 |  |  |  |  |  |  |  |
| Adam2 |  |  |  |  |  |  |  |
| Adam3 |  |  |  |  |  |  |  |
| Adam4 |  |  |  |  |  |  |  |

### 4.6. 实验六：隐藏层数(HIDDEN_DIMS) [Liang]

#### 4.6.1 预期

更深的网络具有更强表达能力，但也更难训练，并且在 MNIST 这类相对简单的数据集上不一定明显提升效果,且更容易过拟合。

#### 4.6.2 结果

| Hidden Dims | Train Acc | Val Acc | Test Acc | Final Train Loss | Final Val Loss | Notes |
|---|---:|---:|---:|---:|---:|---|
| [128] |  |  |  |  |  |  |
| [256, 128] |  |  |  |  |  |  |
| [256, 128, 64] |  |  |  |  |  |  |

---

## 5. 过拟合

判断是否过拟合，主要观察 train accuracy 和 validation accuracy 的差距。

如果出现以下现象，说明模型可能过拟合：

```text
Train Accuracy 持续上升，但 Validation Accuracy 停止提升甚至下降。
Train Loss 持续下降，但 Validation Loss 上升。
Train Accuracy 明显高于 Validation Accuracy。
```
