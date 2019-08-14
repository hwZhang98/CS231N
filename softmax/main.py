import numpy as np
import matplotlib.pyplot as plt
from linear_classifier import Softmax
import time
import random
import sys                   #https://blog.csdn.net/stalbo/article/details/79379078
import data_utils            #https://zhuanlan.zhihu.com/p/28291811
import softmax      #这里要先把文件环境加入path，再在文件里建立一个__init__空文件
sys.path.append('..')
x_train,y_train,x_test,y_test=data_utils.load_cifar10('..\斯坦福CS231N\cifar-10-batches-py')

num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500

mask = range(num_training,num_training+num_validation)
x_val = x_train[mask]
y_val = y_train[mask]
mask = range(num_training)
x_train = x_train[mask]
y_train = y_train[mask]
mask = np.random.choice(num_training,num_dev,replace=False)
x_dev = x_train[mask]
y_dev = y_train[mask]
mask = range(num_test)
x_test = x_test[mask]
y_test = y_test[mask]
'''
print(x_train.shape)
print(y_train.shape)
print(x_dev.shape)
print(y_dev.shape)
print(x_test.shape)
print(y_test.shape)
'''
x_train=np.reshape(x_train,(x_train.shape[0],-1))           #三维数据拉成一维向量
x_test=np.reshape(x_test,(x_test.shape[0],-1))
x_val=np.reshape(x_val,(x_val.shape[0],-1))
x_dev=np.reshape(x_dev,(x_dev.shape[0],-1))
#下面进行归一化处理，对每个特征减去平均值来中心化
mean_image = np.mean(x_train,axis=0)
x_train-=mean_image
x_val-=mean_image
x_test-=mean_image
x_dev-=mean_image
#权重矩阵W其实是W和b，因此我们需要x增加一个维度
x_train=np.hstack([x_train,np.ones((x_train.shape[0],1))])
x_val=np.hstack([x_val,np.ones((x_val.shape[0],1))])
x_test=np.hstack([x_test,np.ones((x_test.shape[0],1))])
x_dev=np.hstack([x_dev,np.ones((x_dev.shape[0],1))])
#损失函数和梯度计算
w=np.random.randn(3073,10)*0.0001   #返回(3073,10)尺寸的符合正态分布的随机数组
loss,grad=softmax.softmax_loss_vectorized(w,x_dev,y_dev,0.00001)#也可使用公式计算梯度 svm_loss_vectorized
print('loss is : %f'%loss)
#梯度检验，用公式计算梯度速度很快,但是实现过程中容易出错，为了解决这个问题，需要进行梯度检验
#把分析梯度法的结果和数值梯度法的结果做比较

def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    for i in range(num_checks):
        ix = tuple([random.randrange(m) for m in x.shape])
        oldval = x[ix]
        x[ix] = oldval + h
        fxph = f(x)
        x[ix] = oldval - h
        fxmh = f(x)
        x[ix] = oldval
        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))
#现在我们对加入了正则项的梯度进行检验
loss, grad = softmax.softmax_loss_vectorized(w,x_dev,y_dev,0.0)
f = lambda w:softmax.softmax_loss_vectorized(w,x_dev,y_dev,0.0)[0]
grad_numerical = grad_check_sparse(f,w,grad)

softmax = Softmax()    #创建对象，此时W为空
tic = time.time()
loss_hist = softmax.train(x_train,y_train,learning_rate = 1e-7,reg = 2.5e4,num_iters = 1500,verbose = True)#此时svm对象中有W
toc = time.time()
print('that took %fs' % (toc -tic))
plt.plot(loss_hist)
plt.xlabel('iteration number')
plt.ylabel('loss value')
plt.show()
#训练完成之后，将参数保存，使用参数进行预测，计算准确率
y_train_pred = softmax.predict(x_train)
print('training accuracy: %f'%(np.mean(y_train==y_train_pred)))#返回条件成立的占比
y_val_pred = softmax.predict(x_val)
print('validation accuracy: %f'%(np.mean(y_val==y_val_pred)))

#在拿到一组数据时一般分为训练集，开发集（验证集），测试集。训练和测试集都知道是干吗的，验证集在除了做验证训练结果外
# 还可以做超参数调优，寻找最优模型。遍历每一种参数组合，训练Softmax模型，然后在验证集上测试，寻找验证集上准确率最高的模型
# 交叉验证很费时间，下面的代码总共循环18次，在9700KCPU上满载运行了十多分钟，自己选择是否运行该段代码。

# 超参数调优（交叉验证）
learning_rates = [1.4e-7, 1.5e-7, 1.6e-7]
# for循环的简化写法12个
regularization_strengths = [(1 + i * 0.1) * 1e4 for i in range(-3, 3)] + [(2 + i * 0.1) * 1e4 for i in range(-3, 3)]
results = {}  # 字典
best_val = -1
best_softmax = None
for learning in learning_rates:  # 循环3次
    for regularization in regularization_strengths:  # 循环6次
        softmax = Softmax()
        softmax.train(x_train, y_train, learning_rate=learning, reg=regularization, num_iters=2000)  # 训练
        y_train_pred = softmax.predict(x_train)  # 预测（训练集）
        train_accuracy = np.mean(y_train == y_train_pred)
        print('training accuracy: %f' % train_accuracy)
        y_val_pred = softmax.predict(x_val)  # 预测（验证集）
        val_accuracy = np.mean(y_val == y_val_pred)
        print('validation accuracy: %f' % val_accuracy)

        if val_accuracy > best_val:
            best_val = val_accuracy
            best_softmax = softmax
        results[(learning, regularization)] = (train_accuracy, val_accuracy)
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f ' % (lr, reg, train_accuracy, val_accuracy))
print('best validation accuracy achieved during cross-validation: %f' % best_val)

#在测试集上验证
y_test_pred = best_softmax.predict(x_test)
test_accuracy = np.mean(y_test == y_test_pred)
print('sofmax on raw pixels final test set accuracy: %f' % (test_accuracy, ))