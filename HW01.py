
# coding: utf-8

# In[65]:


import numpy as np
import h5py
import time
import copy
import math
from random import randint


# In[66]:


#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5','r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()


# In[67]:


#Implementation of stochastic gradient descent algorithm
#number of inputs
num_inputs = 28*28
#number of outputs
num_outputs = 10
#number of hidden layer
num_hidden = 100

model = {}
model['W1'] = np.random.randn(num_hidden,num_inputs) / np.sqrt(num_inputs)#dim H*D
model['b1_1'] = np.random.randn(num_hidden)#dim H
model['b2_1'] = np.random.randn(num_outputs)#dim K
model['C1'] = np.random.randn(num_outputs,num_hidden)#dim K*H
model_grads = copy.deepcopy(model)


# In[68]:


def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ


# In[69]:


#forward
def forward(x,y, model):
    Z = np.dot(model['W1'], x) + model['b1_1']
    H = 1/(np.exp(-Z) + 1)
    U = np.dot(model['C1'],H.reshape(num_hidden,1)) + 
         model['b2_1'].reshape(num_outputs,1)
    p = softmax_function(U)
    return Z,H,U,p


# In[70]:


#backward
def backward(x,y,p,model,model_grads):
    par_u = 1.0*p
    par_u[y] = par_u[y] - 1.0
    model_grads['b2_1'] = par_u
    model_grads['C1'] = np.dot(par_u.reshape(num_outputs,1),
                               H.reshape(1,num_hidden))
    delta = np.dot(np.transpose(model['C1']),par_u)
    deri_sig = np.multiply(H,(1-H))
    model_grads['b1_1'] = np.multiply(delta.reshape(num_hidden,1),
                                      deri_sig.reshape(num_hidden,1))
    model_grads['W1'] = np.dot(model_grads['b1_1'].reshape(num_hidden,1),
                               x.reshape(1,num_inputs))
    return model_grads


# In[71]:


#Learning rate & epochs
LR = .1
num_epochs = 20

for epochs in range(num_epochs):
    #Learning rate schedule
    if (epochs > 5):
        LR = .01
    if (epochs > 10):
        LR = 0.001
    if (epochs > 15):
        LR = 0.0001
    total_correct = 0
    for n in range( len(x_train)):
        n_random = randint(0,len(x_train)-1 )
        y = y_train[n_random]
        x = x_train[n_random][:]
        Z,H,U,p = forward(x, y, model)
        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1
        model_grads = backward(x, y, p, model,model_grads)
       
        model['C1'] = model['C1'] - LR*model_grads['C1']
        model['b2_1'] = model['b2_1'] - LR*model_grads['b2_1'].reshape(num_outputs,)
        model['b1_1'] = model['b1_1'] - LR*model_grads['b1_1'].reshape(num_hidden,)
        model['W1'] = model['W1'] - 
                      LR*model_grads['W1'].reshape(num_hidden,num_inputs)
    print(total_correct/np.float(len(x_train) ) )


# In[72]:


#test data
#print the accuracy on the test data
total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    Z,H,U,p = forward(x, y, model)
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1
print(total_correct/np.float(len(x_test) ) )


# In[2]:



# Test Accuracy = 97.72%
 
#Implementation
#neural network architecture

# Z = Wx+b1
# Hi = σ(Zi), i=0,...,dH −1,（Sigmoidal units）
# U = CH+b2,
# f(x;θ) = Fsoftmax (U)

# Backpropagation algorithm
# In the forward step, the output f(X;θ) and intermediary network 
# values (Z,H, and U) are calculated. In the backward step, the gradient 
# with respect to the parameters θ is calculated.


    

