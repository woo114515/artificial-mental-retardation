'''
下载或读取 MNIST
把图片展平成 784 维
把像素归一化到 0 到 1
划分 train / validation / test
返回标准格式的数据

输出：
    X_train, y_train
    X_val, y_val
    X_test, y_test
'''
import gzip
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# 1. 读取文件
def load_mnist_images(filename):
    """读取图像文件，返回 (数量, 28, 28) 的 numpy 数组"""
    with gzip.open(filename, 'rb') as f:
        # 图像文件头为 16 字节
        magic_number = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')
        
        # 读取剩下的像素数据
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8).reshape(num_images, num_rows, num_cols)
        return images

def load_mnist_labels(filename):
    """读取标签文件，返回 (数量,) 的 1 维 numpy 数组"""
    with gzip.open(filename, 'rb') as f:
        # 标签文件头只有 8 字节（魔数 4 字节 + 标签数量 4 字节）
        magic_number = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        
        # 读取剩下的标签数据 (0-9 的数字)
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels

# 1.1 定义文件名（确保这些文件和代码在同一文件夹）
# .parent 就是项目根目录
project_root = Path(__file__).parent #.parent 

# 定位到数据文件夹
data_dir = project_root / 'data' #如果后面新增raw文件夹，就在加一个data_dir = project_root / 'raw'

# 拼接具体文件
train_labels_file = data_dir / 'train-labels-idx1-ubyte.gz'
train_images_file = data_dir / 'train-images-idx3-ubyte.gz'
test_images_file  = data_dir / 't10k-images-idx3-ubyte.gz'
test_labels_file  = data_dir / 't10k-labels-idx1-ubyte.gz'

# 1.2 读取所有数据
train_images = load_mnist_images(train_images_file)
train_labels = load_mnist_labels(train_labels_file)

test_images = load_mnist_images(test_images_file)
test_labels = load_mnist_labels(test_labels_file)

# 2. 分离出 5000 个验证集 (Validation Set)
# 选取最后 5000 个样本作为验证集
val_images = train_images[-5000:]
val_labels = train_labels[-5000:]

# 剩下的前 55000 个样本作为真正的训练集
train_images_final = train_images[:-5000]
train_labels_final = train_labels[:-5000]

# 3. 展平图像并与标签合并为二维表格
# 3.1 将图像从 (55000, 28, 28) 展平为 (55000, 784)
# -1 代表让 numpy 自动计算这一维度的长度 (28 * 28 = 784)
train_images_flat = train_images_final.reshape(train_images_final.shape[0], -1)
val_images_flat = val_images.reshape(val_images.shape[0], -1)
test_images_flat = test_images.reshape(test_images.shape[0], -1)

# 3.2 将所有数据归一化
train_images_flat = train_images_flat.astype(np.float32) / 255.0
val_images_flat = val_images_flat.astype(np.float32) / 255.0
test_images_flat = test_images_flat.astype(np.float32) / 255.0

# 3.3 将标签从 1 维 (55000,) 变形为 2 维列向量 (55000, 1)
# 这样才能和刚才展平的图像矩阵进行左右拼接
train_labels_2d = train_labels_final.reshape(-1, 1)
val_images_2d = val_labels.reshape(-1, 1)
test_images_2d = test_labels.reshape(-1, 1)

# 3.4 水平拼接 (Horizontal Stack): 左边是 1 列标签，右边是 784 列像素
train_table = np.hstack((train_labels_2d, train_images_flat))
val_table = np.hstack((val_images_2d, val_images_flat))
test_table = np.hstack((test_images_2d, test_images_flat))

# 4. 将二维表格转换为 pandas DataFrame
# 4.1 动态生成 785 个列名
# 第一列叫 'label'，后面跟着 'pixel_0', 'pixel_1' ... 一直到 'pixel_783'
column_names = ['label'] + [f'pixel_{i}' for i in range(784)]

# 4.2 创建 DataFrame
train_df = pd.DataFrame(train_table, columns=column_names)
val_df = pd.DataFrame(val_table, columns=column_names)
test_df = pd.DataFrame(test_table, columns=column_names)

# 5. 划分训练集、验证集、测试集
train_df = np.array(train_df)
val_df = np.array(val_df)
test_df = np.array(test_df)

m, n = train_df.shape
np.random.shuffle(train_df)
np.random.shuffle(val_df)
np.random.shuffle(test_df)

train_data = train_df.T
val_data = val_df.T
test_data = test_df.T

y_train = train_data[0].astype(int)
X_train = train_data[1:n]
y_val = val_data[0].astype(int)
X_val = val_data[1:n]
y_test = test_data[0].astype(int)
X_test = test_data[1:n]