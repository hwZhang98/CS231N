import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_cifar10
import sys
from neural_network import TwoLayerNet
from  vis_utils import visualize_grid
sys.path.append('..')
def get_cifar_data(num_training=49000,num_validation=1000,num_test=1000):
    x_train, y_train, x_test, y_test = load_cifar10('..\斯坦福CS231N\cifar-10-batches-py')
    #验证集
    mask = range(num_training, num_training + num_validation)
    x_val = x_train[mask]
    y_val = y_train[mask]
    #训练集
    mask = range(num_training)
    x_train = x_train[mask]
    y_train = y_train[mask]
    #测试集
    mask = range(num_test)
    x_test = x_test[mask]
    y_test = y_test[mask]
    #求平均值，归一化
    mean_image = np.mean(x_train,axis=0)
    x_train -= mean_image
    x_val -= mean_image
    x_test -= mean_image
    #展开
    x_train = x_train.reshape(num_training,-1)
    x_val = x_val.reshape(num_validation, -1)
    x_test = x_test.reshape(num_test, -1)

    return x_train,y_train,x_val,y_val,x_test,y_test
x_train,y_train,x_val,y_val,x_test,y_test = get_cifar_data()
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
print(x_test.shape)
print(y_test.shape)
input_size = 32*32*3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size,hidden_size,num_classes)
stats = net.train(x_train,y_train,x_val,y_val,num_iters=1000,batch_size=200,learning_rate=1e-3,learning_rate_decay=0.8,
                  reg=0.5,verbose=True)
val_acc = (net.predict(x_val)==y_val).mean()  #预测验证集
print('valiadation accuracy:',val_acc)

#准确率不高，我们需要知道中间发生了什么，对参数进行可视化
#loss 和 accuracy 进行可视化
plt.subplot(211)
plt.plot(stats['loss_history'])
plt.title('loss history')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(212)
plt.plot(stats['train_acc_history'],label='train')
plt.plot(stats['val_acc_history'],label='val')
plt.title('classification accuracy history')
plt.xlabel('epoch')
plt.ylabel('classification accuracy')
plt.show()
#权重可视化
def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32,32,3,-1).transpose(3,0,1,2)
    plt.imshow(visualize_grid(W1,padding=3).astype('uint8'))
    plt.axis('off')
    plt.show()

show_net_weights(net)
#超参数调优
hidden_size = [75,100,125]
results = {}
best_val_acc = 0
best_net = None
learning_rates = np.array([0.7,0.8,0.9,1.0,1.1])*1e-3
regularization_strengths = [0.75,1.0,1.25]
print('running')
for hs in hidden_size:
    for lr in learning_rates:
        for reg in regularization_strengths:
            net = TwoLayerNet(input_size,hs,num_classes)
            stats = net.train(x_train,y_train,x_val,y_val,num_iters=1500,batch_size=200,
                              learning_rate=lr,learning_rate_decay=0.95,reg=reg,verbose=False)
            val_acc = (net.predict(x_val)==y_val).mean()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_net = net
            results[(hs,lr,reg)] = val_acc
print('finshed')

for hs,lr,reg in sorted(results):
    val_acc = results[(hs,lr,reg)]
    print('hs %d lr %e reg %e val accuracy: %f'%(hs,lr,reg,val_acc))

print("best validation accuracy achieved during cross_validationg:%f"% best_val_acc)
