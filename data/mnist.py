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

# 1. 加载数据
def load_mnist_images(filename):
    """读取图像文件，返回 (数量, 28, 28) 的 numpy 数组"""
    with gzip.open(filename, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')
        
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8).reshape(num_images, num_rows, num_cols)
        return images

def load_mnist_labels(filename):
    """读取标签文件，返回 (数量,) 的 1 维 numpy 数组"""
    with gzip.open(filename, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels

# 2. 定位文件并读取
project_root = Path(__file__).parent
data_dir = project_root / 'raw' / 'mnist'

train_images_file = data_dir / 'train-images-idx3-ubyte.gz'
train_labels_file = data_dir / 'train-labels-idx1-ubyte.gz'
test_images_file  = data_dir / 't10k-images-idx3-ubyte.gz'
test_labels_file  = data_dir / 't10k-labels-idx1-ubyte.gz'

train_images = load_mnist_images(train_images_file)
train_labels = load_mnist_labels(train_labels_file)
test_images = load_mnist_images(test_images_file)
test_labels = load_mnist_labels(test_labels_file)

# 3. 分离验证集，展平并归一化 (构建 Row-Major 的 X 和 y)
# X 的目标形状: (样本数, 784) ; y 的目标形状: (样本数,)      

# 3.1 提取并处理训练集 (前 55000 个)
X_train = train_images[:-5000].reshape(55000, -1).astype(np.float32) / 255.0
y_train = train_labels[:-5000]

# 3.2 提取并处理验证集 (后 5000 个)
X_val = train_images[-5000:].reshape(5000, -1).astype(np.float32) / 255.0
y_val = train_labels[-5000:]

# 3.3 处理测试集 (10000 个)
X_test = test_images.reshape(test_images.shape[0], -1).astype(np.float32) / 255.0
y_test = test_labels

# 4. 数据处理 (Shuffle) - 纯 NumPy 方案
def shuffle_data(X, y):
    shuffled_indices = np.random.permutation(len(X))
    return X[shuffled_indices], y[shuffled_indices]

# 执行处理
X_train, y_train = shuffle_data(X_train, y_train)
X_val, y_val = shuffle_data(X_val, y_val)
X_test, y_test = shuffle_data(X_test, y_test)