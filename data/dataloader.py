'''
打乱训练数据
按 batch size 切分数据
每次返回一个 X_batch 和 y_batch
可以暂时不使用
详见 stochastic gradient descent 和 mini-batch gradient descent 的区别
'''
import numpy as np

def create_mini_batches(X, y, batch_size=64):
    """
    参数:
        X: 特征矩阵，形状为 (m, n) 即 (样本数, 特征数)
        y: 标签向量，形状为 (m,)
        batch_size: 每个 batch 包含的样本数，可根据需求调整
    返回:
        一个生成器，每次 yield 一个元组 (X_batch, y_batch)
    """
    m = X.shape[0] # 获取样本总数
    
    # 1. 打乱索引
    indices = np.random.permutation(m)
        
    # 2. 按照 batch_size 步长进行循环切片
    for i in range(0, m, batch_size):
        # 截取当前 batch 的索引序列
        batch_indices = indices[i : i + batch_size]
        
        # 3. 提取对应的 X 和 y
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        
        # 4. yield 吐出当前批次
        yield X_batch, y_batch #用yield而不是return能节省内存