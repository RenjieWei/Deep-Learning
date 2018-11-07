import numpy as np
import h5py
import time
import copy
from random import randint
import matplotlib.pyplot as plt

#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32(MNIST_data['x_test'][:] )
y_test = np.int32(np.array( MNIST_data['y_test'][:,0] ))

MNIST_data.close()
#Implementation of stochastic gradient descent algorithm
#number of inputs
n_inputs = 28*28
n_layer1 = 300
#number of outputs
n_outputs = 10

class DNN():
    def __init__(self):     
        #weights
        self.model = {}
        self.model_grads = {}
        self.model['W1'] = np.random.randn(n_inputs, n_layer1) / np.sqrt(n_inputs)
        self.model["b1"] = np.random.randn(n_layer1) / np.sqrt(n_layer1)
        self.model['W2'] = np.random.randn(n_layer1, n_outputs) / np.sqrt(n_layer1)
        self.model["b2"] = np.random.randn(n_outputs) / np.sqrt(n_outputs)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def d_sigmoid(self, z):
        return z * (1 - z)

    def FC_layer(self, X, W, b):
        return X.dot(W) + b

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z))

    def d_softmax(self, gt, o):
        return o - gt

    def forward(self, x):
        z1 = self.FC_layer(x, self.model['W1'],self. model["b1"])
        self.model['a1'] = self.sigmoid(z1)
        z2 = self.FC_layer(self.model['a1'], self.model['W2'], self.model["b2"])
        o = self.softmax(z2)
        return o

    def gd(self, key, LR):
        self.model[key] = self.model[key] - LR*self.model_grads[key]

    def backward(self, X, gt, o, LR):
        d_sm = self.d_softmax(gt, o)
        self.model_grads['W2'] = np.outer(self.model['a1'], d_sm)
        self.model_grads['b2'] = d_sm
        d_l1 = self.d_sigmoid(self.model['a1']) * d_sm.dot(self.model['W2'].T)
        self.model_grads['W1'] = np.outer(X, d_l1)
        self.model_grads['b1'] = d_l1
        for key in ['W2', 'b2', 'W1', 'b1']:
            self.gd(key, LR)

import time
time1 = time.time()
LR = .01
n_epochs = 20

dnn = DNN()
train_acc = []
for epochs in range(n_epochs):
    #Learning rate schedule
    if (epochs > 5):
        LR = 0.001
    if (epochs > 10):
        LR = 0.0001
    if (epochs > 15):
        LR = 0.00001
    total_correct = 0
    print("epoch:", epochs)
    c = 0
    for n in range(len(x_train)):
        if n == len(x_train) // 4:
            print("training: 25%")
        elif n == len(x_train) // 2:
            print("training: 50%")
        elif n == len(x_train) // 4 * 3:
            print("training: 75%")
        n_random = randint(0,len(x_train)-1)
        y = y_train[n_random]
        gt = np.zeros(n_outputs)
        gt[y] = 1.0
        x = x_train[n_random][:]
        p = dnn.forward(x)
        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1
        dnn.backward(x, gt, p, LR)
    acc = total_correct/np.float(len(x_train))
    train_acc.append(acc)
    print("train acc:", acc)
time2 = time.time()
print("time:", time2-time1)

print(train_acc)
# x = list(range(1, n_epochs+1))
# plt.plot(x, train_acc, label="training")
# plt.ylabel("acc")
# plt.xlabel("epoch")
# plt.legend()
# plt.show()

######################################################
#test data
total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    p = dnn.forward(x)
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1
print("test acc:", total_correct/np.float(len(x_test)))
