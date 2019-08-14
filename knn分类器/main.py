import numpy as np
from data_utils import load_cifar10
import matplotlib.pyplot as plt
from  knn import KNearestNeighbor
import sys
sys.path.append('..')
x_train,y_train,x_test,y_test=load_cifar10('..\斯坦福CS231N\cifar-10-batches-py')
# 为了验证我们的结果是否正确，我们可以打印输出下：
print('training data shape:',x_train.shape)
print('training labels shape:',y_train.shape)
print('test data shape:',x_test.shape)
print('test labels shape:',y_test.shape)
# 共有50000张训练集，10000张测试集。
# 下面我们从这50000张训练集每一类中随机挑选samples_per_class张图片进行展示，代码如下：代码如下
classes=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
num_claesses=len(classes)
samples_per_class=7
for y ,cls in enumerate(classes):
    idxs=np.flatnonzero(y_train==y)
    idxs=np.random.choice(idxs,samples_per_class,replace=False)
    for i ,idx in enumerate(idxs):
        plt_idx=i*num_claesses+y+1                          #其中一个图片所在的位置
        plt.subplot(samples_per_class,num_claesses,plt_idx) #激活图片位置的画板
        plt.imshow(x_train[idx].astype('uint8'))
        plt.axis('off')
        if i ==0:
            plt.title(cls)
plt.show()

#为了加快我们的训练速度，我们只选取5000张训练集，500张测试集
num_training=5000
mask=range(num_training)
x_train=x_train[mask]
y_train=y_train[mask]
num_test=500
mask=range(num_test)
x_test=x_test[mask]
y_test=y_test[mask]

#至此，数据载入部分已经算是完成了，但是为了欧氏距离的计算，我们把得到的图像数据拉长成行向量
x_train=np.reshape(x_train,(x_train.shape[0],-1))
x_test=np.reshape(x_test,(x_test.shape[0],-1))
print(x_train.shape,x_test.shape)

classifier=KNearestNeighbor()
classifier.train(x_train,y_train)
dists=classifier.compute_distances_two_loops(x_test)            #也可用其他两种方法
y_test_pred = classifier.predict_labels(dists, k=1)
#模型评估也是机器学习中的一个重要概念，这里我们使用准确率作为模型的评价指标，
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
#这三种方法的差别在于它们的计算时间不同，我们来做下比较。比较代码如下：
import time
def time_function(f,*args):
    tic=time.time()
    f(*args)
    toc=time.time()
    return toc-tic

two_loop_time=time_function(classifier.compute_distances_two_loops,x_test)
print('two loops version took %f seconds' % two_loop_time)

one_loop_time=time_function(classifier.compute_distances_one_loop,x_test)
print('one loop version took %f seconds' % one_loop_time)

no_loops_time=time_function(classifier.compute_distances_no_loops,x_test)
print('no loops version took %f seconds' % no_loops_time)

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
x_train_folds = []
y_train_folds = []

y_train = y_train.reshape(-1, 1)
x_train_folds = np.array_split(x_train, num_folds)          #1 表示将数据集平均分成5份；
y_train_folds = np.array_split(y_train, num_folds)

k_to_accuracies = {}                                        #2 以字典形式存储k和accuracy精度值；

for k in k_choices:
    k_to_accuracies.setdefault(k,[])
for i in range(num_folds):              #3 对每个k值，选取一份测试，其余训练，计算准确率
    classifier = KNearestNeighbor()
    x_val_train = np.vstack(x_train_folds[0:i] + x_train_folds[i+1:])#3.1表示除i之外的作为训练集
    y_val_train = np.vstack(y_train_folds[0:i] + y_train_folds[i+1:])
    y_val_train = y_val_train[:,0]
    classifier.train(x_val_train,y_val_train)
    for k in k_choices:
        y_val_pred = classifier.predict(x_train_folds[i], k=k)  #3.2第i份作为测试集并预测
        num_correct = np.sum(y_val_pred == y_train_folds[i][:, 0])
        accuracy = float(num_correct) / len(y_val_pred)
        k_to_accuracies[k] = k_to_accuracies[k] + [accuracy]

for k in sorted(k_to_accuracies):  #4 表示输出每次得到的准确率以及每个k值对应的平均准确率。
    sum_accuracy=0
    for accuracy in k_to_accuracies[k]:
        print('k=%d, accuracy=%f' % (k,accuracy))
        sum_accuracy+=accuracy
    print('the average accuracy is :%f' % (sum_accuracy/5))

# 为了更形象的表示准确率，我们借助matplotlib.pyplot.errorbar函数来均值和偏差对应的趋势线，代码如下：
for k in k_choices:
    accuracies=k_to_accuracies[k]
    plt.scatter([k]*len(accuracies),accuracies)

accuracies_mean=np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std=np.array([np.std(v) for k ,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices,accuracies_mean,yerr=accuracies_std)
plt.title('cross-validation on k')
plt.xlabel('k')
plt.ylabel('cross-validation accuracy')
plt.show()

# 已经通过交叉验证选择了最好的k值，下面我们就要使用最好的k值来完成预测任务
best_k=10
classifier=KNearestNeighbor()
classifier.train(x_train,y_train)
y_test_pred=classifier.predict(x_test,k=best_k)

num_correct=np.sum(y_test_pred==y_test)
accuracy=float(num_correct)/num_test
print('got %d / %d correct => accuracy: %f' % (num_correct,num_test,accuracy))

# 根据准确率，我们可以看到，使用经过交叉验证得到的最好的k值（当然仅限于我们前面所列举的几个），模型的准确率虽有所提升，
# 但还是不够高，我们将在接下来完成cs231n其他作业的同时使用机器学习其他算法提升模型的分类性能。
# 至此，我们便完成了将测试集输入knn模型、输出相应类别以及模型的准确率的工作。
# 以上我们完成了模型构建、数据载入、预测输出工作，根据预测测试集准确率可知，该分类器的准确率并不高
# （相对于卷积神经网络接近100%的准确率来说），在实际中也很少使用knn算法，但是通过这个算法我们可以对图像分
# 类问题的基本过程有个大致的了解。


