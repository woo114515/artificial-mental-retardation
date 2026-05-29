'''
所有的超参数存放在这里。
'''
'''
是否使用CNN
'''
MODEL_TYPE = "cnn"  # or "mlp"
'''
由数据集确定的参量
'''
INPUT_DIM = 784 # 输入维数
NUM_CLASSES = 10 # 输出维数
IMAGE_CHANNELS = 1
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28

'''
实验时需要调节的参量
'''
# 基础参量
HIDDEN_DIMS = [128] # 隐藏层的结构。如[128]代表只有一层128维的隐藏层
BATCH_SIZE = 64 # 详见mini-batch gardient descent
NUM_EPOCHS = 15 # 总训练次数
LEARNING_RATE = 0.01 # 学习率

# CNN结构
CNN_OUT_CHANNELS = 8
CNN_KERNEL_SIZE = 3
CNN_STRIDE = 1
CNN_PADDING = 1
POOL_KERNEL_SIZE = 2
POOL_STRIDE = 2

# 优化器
OPTIMIZER = "momentum" # 可选：'sgd', 'momentum', 'adam'
MOMENTUM = 0.9
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8

# 激活函数
ACTIVATION = "ReLU" # 可选:'ReLU', 'Sigmoid'

# 参数初始化
WEIGHT_INIT = "he" # 初始权重的方式
RANDOM_SEED = 42 # 初始权重的随机种子

# 模型保存
CHECKPOINT_PATH = "checkpoints/latest_model.npz"
