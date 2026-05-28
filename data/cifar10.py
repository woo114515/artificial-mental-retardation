import tarfile
import pickle
import numpy as np
from pathlib import Path

def load_cifar10_from_targz(tar_path):
    """
    直接从 .tar.gz 压缩包中读取 CIFAR-10 数据到内存
    不生成任何临时解压文件
    """
    X_train_list = []
    y_train_list = []
    X_test, y_test = None, None

    # 使用 tarfile 以读取 gz 的模式打开压缩包
    with tarfile.open(tar_path, 'r:gz') as tar:
        # 遍历压缩包里的所有文件
        for member in tar.getmembers():
            
            # 如果文件名包含 'data_batch'，说明是训练集的一部分 (共 5 个)
            if 'data_batch' in member.name:
                f = tar.extractfile(member)
                batch_dict = pickle.load(f, encoding='bytes')
                X_train_list.append(batch_dict[b'data'])
                y_train_list.append(np.array(batch_dict[b'labels']))
                
            # 如果文件名包含 'test_batch'，说明是测试集
            elif 'test_batch' in member.name:
                f = tar.extractfile(member)
                batch_dict = pickle.load(f, encoding='bytes')
                X_test = batch_dict[b'data']
                y_test = np.array(batch_dict[b'labels'])

    # 将 5 个训练批次列表合并成完整的数组
    X_train_raw = np.vstack(X_train_list)
    y_train_raw = np.concatenate(y_train_list)
    
    return X_train_raw, y_train_raw, X_test, y_test

# 1. 定位压缩包文件
project_root = Path(__file__).parent
cifar_tar_file = project_root / 'raw' / 'cifar10' / 'cifar-10-python.tar.gz' 

# 2. 从压缩包直接加载数据
X_train_raw, y_train_raw, X_test, y_test = load_cifar10_from_targz(cifar_tar_file)

# 3. 归一化与分离验证集
# 将 uint8 转换为 float32 并除以 255.0
X_train_raw = X_train_raw.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# 切分最后 5000 个作为验证集
X_train = X_train_raw[:-5000]
y_train = y_train_raw[:-5000]

X_val = X_train_raw[-5000:]
y_val = y_train_raw[-5000:]

# 4. 数据处理（Shuffle）
def shuffle_data(X, y):
    shuffled_indices = np.random.permutation(len(X))
    return X[shuffled_indices], y[shuffled_indices]

# 执行处理
X_train, y_train = shuffle_data(X_train, y_train)
X_val, y_val = shuffle_data(X_val, y_val)