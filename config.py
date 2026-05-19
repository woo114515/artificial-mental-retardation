'''
所有的超参数存放在这里。
'''

# 由数据集确定，不变
INPUT_DIM = 784 # 输入维数
NUM_CLASSES = 10 # 输出维数

# 实验时需要调节的参数
HIDDEN_DIMS = [128] # 隐藏层的结构。如[128]代表只有一层128维的隐藏层
BATCH_SIZE = 64 # 详见mini-batch gardient descent
NUM_EPOCHS = 20 # 总训练次数
LEARNING_RATE = 0.01 # 学习率

WEIGHT_INIT = "he" 
RANDOM_SEED = 42 # 初始变量的随机种子