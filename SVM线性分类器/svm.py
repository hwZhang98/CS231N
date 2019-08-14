import numpy as np

def svm_loss_naive(W, X, y, reg):
    """
        使用循环实现SVM loss函数
        输入维度为D，有C类，我们使用N个样本作为一批输入
        输入：
        -W：shape(D, C) 权重矩阵  权重矩阵一列作为一组特征
        -X: shape(N, D) 数据      输入数据为一行一个样本
        -y：shape(N, )  标签
        -reg：float，正则化强度

        return： tuple
        - 存储为float的loss
        - 权重W的梯度，和W大小相同的array
    """
    dW = np.zeros(W.shape)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    print(y[i])
    print(type(y))
    for i in range(num_train):  # 循环次数为数据的条数
        scores = X[i].dot(W)  # scores.shape = (1, C)，将每一行都乘以权重矩阵，得到对应10个分类的得分
        correct_class_score = scores[y[i]]  # 找到正确的得分
        for j in range(num_classes):  # 循环次数为类别个数
            if (j == y[i]):
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin
                # 猜测：那么就要加大正确列的权重系数，所以是减去X对应行的特征值！！因为最后是减dW，所以这边符号是相反的）
                # 相应的，如果margin大于0，说明损失过大，要减小得分较大列的权重系数，所以是加上对应X的特征值
                dW[:, y[i]] += -X[i].T
                dW[:, j] += X[i].T  # 对应的其他列 加上 对应的数据行
    loss /= num_train
    dW /= num_train
    # 加入正则项
    loss += reg * np.sum(W * W)
    dW += reg * W
    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
    """
    结构化的SVM损失函数，使用向量实现
    """
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)  # 首先计算得分  scores.shape = (num_classes, num_characters)
    # 这边用到list，list是一个列表（对应第几个是正确类别）实现对每一行正确分类得分的查找
    correct_class_score = scores[range(num_train), list(y)].reshape(-1,1) # correct_class_score.shape = (num_train,1)
    margins = np.maximum(0, scores - correct_class_score + 1)  # broadcast，把得分减去正确得分，加上delta，比较出最大值
    margins[range(num_train), list(y)] = 0.0  # 继续筛选，正确分类是不需要计算得分差值的
    loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W * W)  # 对margins求和然后求均值，加上正则化项
    coeff_mat = np.zeros((num_train, num_classes))
    coeff_mat[margins > 0] = 1  # 类似matlab，得到margin中大于0的索引，然后给coeff_mat赋值为1
    coeff_mat[range(num_train), list(y)] = 0  # 将正确类系数肯定为1，所以要清0
    coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)  # 给正确类加权重，有几个错误分类那就加几个
    # ，对应朴素的有几个margin>0就加几次
    dW = (X.T).dot(coeff_mat)  # dW为训练集矩阵乘以系数矩阵
    dW = dW / num_train + reg * W  # 正则化
    return loss, dW


