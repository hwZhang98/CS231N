import numpy as np
import svm

# 梯度、损失计算完成之后，对线性分类的模型进行了建立。利用了随机梯度下降。以往的梯度下降运行时直接将所有数据输入
# 然后运算；随机梯度下降简单理解来说，每次从数据集中随机选取mini - batch数据来进行训练。
class LinearClassifier:
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        '''

        :param X: (N,D)维输入图像，N表示图像数，每个图像都是D*1的列向量
        :param y: (N,)维标签，数组的每个数取值为0...k-1
        :param learning_rate:学习率
        :param reg: 正则化系数
        :param num_iters: 迭代次数
        :param batch_size:  每一批尺寸
        :param verbose: 为False时不显示迭代过程
        :return:
        '''
        num_train = X.shape[0]  # 样本数
        dim = X.shape[1]  # 特征维度
        num_classes = np.max(y) + 1  # 类别数，从0开始数，所以加一

        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)  # 初始化W

        # Run stochastic gradient descent(Mini-Batch) to optimize W
        loss_history = []
        for it in range(num_iters):  # 每次随机取batch的数据来进行梯度下降
            X_batch = None
            y_batch = None
            # Sampling with replacement is faster than sampling without replacement.
            batch_idx = np.random.choice(num_train, batch_size, replace=False)
            X_batch = X[batch_idx, :]  # batch_size by D
            y_batch = y[batch_idx]  # 1 by batch_size
            # evaluate loss and gradient
            # 子类调用的是子类的loss方法
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update  梯度下降
            self.W += -learning_rate * grad
            if verbose and it % 100 == 0:
                print('Iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    # 预测
    def predict(self, X):
        scores = X.dot(self.W)  # 计算得分  scores.shape = (num_train,num_classes)
        y_pred = np.argmax(scores, axis=1)  # 返回最大数的索引
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        pass


class LinearSVM(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        return svm.svm_loss_vectorized(self.W, X_batch, y_batch, reg)
