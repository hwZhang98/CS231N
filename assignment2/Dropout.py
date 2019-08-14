#!/usr/bin/env python
# coding: utf-8

# # Dropout
# Dropout [1] is a technique for regularizing neural networks by randomly setting
# some output activations to zero during the forward pass. In this exercise you
# will implement a dropout layer and modify your fully-connected network to
# optionally use dropout.
# 
# [1] [Geoffrey E. Hinton et al, "Improving neural networks by preventing
# co-adaptation of feature detectors", arXiv 2012](https://arxiv.org/abs/1207.0580)

# In[ ]:

# https://github.com/L1aoXingyu/cs231n-assignment-solution/blob/master/assignment2/Dropout.ipynb
# As usual, a bit of setup
from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver

# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# In[ ]:


# Load the (preprocessed) CIFAR10 data.

data = get_CIFAR10_data()
for k, v in data.items():
  print('%s: ' % k, v.shape)


# # Dropout forward pass
# In the file `cs231n/layers.py`, implement the forward pass for dropout. Since
# dropout behaves differently during training and testing, make sure to implement
# the operation for both modes.
# 
# Once you have done so, run the cell below to test your implementation.

# In[ ]:


np.random.seed(231)
x = np.random.randn(500, 500) + 10

for p in [0.25, 0.4, 0.7]:
  out, _ = dropout_forward(x, {'mode': 'train', 'p': p})
  out_test, _ = dropout_forward(x, {'mode': 'test', 'p': p})

  print('Running tests with p = ', p)
  print('Mean of input: ', x.mean())
  print('Mean of train-time output: ', out.mean())
  print('Mean of test-time output: ', out_test.mean())
  print('Fraction of train-time output set to zero: ', (out == 0).mean())
  print('Fraction of test-time output set to zero: ', (out_test == 0).mean())
  print()


# # Dropout backward pass
# In the file `cs231n/layers.py`, implement the backward pass for dropout. After
# doing so, run the following cell to numerically gradient-check your implementation.

# In[ ]:


np.random.seed(231)
x = np.random.randn(10, 10) + 10
dout = np.random.randn(*x.shape)

dropout_param = {'mode': 'train', 'p': 0.2, 'seed': 123}
out, cache = dropout_forward(x, dropout_param)
dx = dropout_backward(dout, cache)
dx_num = eval_numerical_gradient_array(lambda xx: dropout_forward(xx, dropout_param)[0], x, dout)

# Error should be around e-10 or less
print('dx relative error: ', rel_error(dx, dx_num))


# ## Inline Question 1:
# What happens if we do not divide the values being passed through inverse dropout
# by `p` in the dropout layer? Why does that happen?
# 
# ## Answer:
# [在测试的时候，dropout 不做任何操作，所以输出的数学期望是其本身，但是在训练的时候
# ，dropout 会改变输入和输出的数学期望，比如输入是x，以概率 p 保留，那么输出的数学
# 期望就是$E(\hat{x}) = p * x + (1 - p) * 0 = px$，因为我们希望在训练和测试的时候
# 数学期望保持一致，那么在训练中forward的时候，需要除以p保证输入和输出的数学期望不变。
# 这不是我写的，是我抄的]
# 

# # Fully-connected nets with Dropout
# In the file `cs231n/classifiers/fc_net.py`, modify your implementation to use
# dropout. Specifically, if the constructor of the network receives a value that
# is not 1 for the `dropout` parameter, then the net should add a dropout layer
# immediately after every ReLU nonlinearity. After doing so, run the following to
# numerically gradient-check your implementation.

# In[ ]:


np.random.seed(231)
N, D, H1, H2, C = 2, 15, 20, 30, 10
X = np.random.randn(N, D)
y = np.random.randint(C, size=(N,))

for dropout in [1, 0.75, 0.5]:
  print('Running check with dropout = ', dropout)
  model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                            weight_scale=5e-2, dtype=np.float64,
                            dropout=dropout, seed=123)

  loss, grads = model.loss(X, y)
  print('Initial loss: ', loss)
  
  # Relative errors should be around e-6 or less; Note that it's fine
  # if for dropout=1 you have W2 error be on the order of e-5.
  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
    print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
  print()


# # Regularization experiment
# As an experiment, we will train a pair of two-layer networks on 500 training
# examples: one will use no dropout, and one will use a keep probability of 0.25.
# We will then visualize the training and validation accuracies of the two networks
# over time.

# In[ ]:


# Train two identical nets, one with dropout and one without
np.random.seed(231)
num_train = 500
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

solvers = {}
dropout_choices = [1, 0.25]
for dropout in dropout_choices:
  model = FullyConnectedNet([500], dropout=dropout)
  print(dropout)

  solver = Solver(model, small_data,
                  num_epochs=25, batch_size=100,
                  update_rule='adam',
                  optim_config={
                    'learning_rate': 5e-4,
                  },
                  verbose=True, print_every=100)
  solver.train()
  solvers[dropout] = solver
  print()


# In[ ]:


# Plot train and validation accuracies of the two models

train_accs = []
val_accs = []
for dropout in dropout_choices:
  solver = solvers[dropout]
  train_accs.append(solver.train_acc_history[-1])
  val_accs.append(solver.val_acc_history[-1])

plt.subplot(3, 1, 1)
for dropout in dropout_choices:
  plt.plot(solvers[dropout].train_acc_history, '-o', label='%.2f dropout' % dropout)
plt.title('Train accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(ncol=2, loc='lower right')
  
plt.subplot(3, 1, 2)
for dropout in dropout_choices:
  plt.plot(solvers[dropout].val_acc_history, '-o', label='%.2f dropout' % dropout)
plt.title('Val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(ncol=2, loc='lower right')

plt.gcf().set_size_inches(15, 15)
plt.show()


# ## Inline Question 2:
# Compare the validation and training accuracies with and without dropout -- what
# do your results suggest about dropout as a regularizer?
# 
# ## Answer:
# [在使用dropout的时候，训练的准确率会比不使用dropout的时候训练的准确率低，同时验证
# 集的准确率会比不使用dropout的时候高一些，这说明了dropout可以作为regularization
# （正则化），减少过拟合]
# 

# ## Inline Question 3:
# Suppose we are training a deep fully-connected network for image classification,
# with dropout after hidden layers (parameterized by keep probability p). If we
# are concerned about overfitting, how should we modify p (if at all) when we
# decide to decrease the size of the hidden layers (that is, the number of nodes
# in each layer)?
# 
# ## Answer:
# [如果我们需要减小hidden layers的尺寸，也就减小隐藏层的神经元个数，那么dropout的
# 保留概率应该加大，比如考虑一个最极端的情况，我们将隐藏层的神经元个数减小到1，
# 那么如果doprou的保留概率仍然特别小，那网络在大多数时候根本没有进行训练]
# 
