
# coding: utf-8

# In[28]:


import numpy as np
import h5py
import time
import copy
import math
from random import randint


# In[29]:


#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5','r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()


# In[30]:


#number of inputs
num_inputs = 28
#number of outputs
num_outputs = 10

#number of channels(3-5)
num_channels = 3

#number of Ky/Kx
filter_size = 5

model = {}
model['W'] = np.random.randn(num_outputs,num_inputs-filter_size+1,num_inputs-filter_size+1,num_channels)/num_inputs 
model['b'] = np.random.randn(num_outputs)/num_inputs
model['K'] = np.random.randn(num_channels,filter_size,filter_size)/num_inputs
model_grads = copy.deepcopy(model)


# In[31]:


def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ


# In[32]:


#forward
def forward(x,y, model):
    x = x.reshape(num_inputs,num_inputs)
    Z = np.random.randn(num_channels, num_inputs-filter_size+1, num_inputs-filter_size+1)
    for p in range(num_channels):
        for i in range(num_inputs-filter_size+1):
            for j in range(num_inputs-filter_size+1):
                mul_x = x[i:i+filter_size, j:j+filter_size]
                Z[p][i][j] = np.tensordot(mul_x, model['K'][p], axes=2)
    H = 1/(np.exp(-Z) + 1)
    H = H.reshape(num_inputs-filter_size+1, num_inputs-filter_size+1,num_channels)
    W_H = np.random.randn(num_outputs)
    for k in range(num_outputs):
        W_H[k] = np.sum(np.multiply(model['W'][k],H))
    U = W_H + model['b']
    f = softmax_function(U)
    return Z,H,U,f


# In[33]:


#backward
def backward(x,y,f,model,model_grads):
    x = x.reshape(28,28)
    par_u = 1.0*f
    par_u[y] = par_u[y] - 1.0
    model_grads['b'] = par_u
    delta = np.zeros((num_inputs-filter_size+1,num_inputs-filter_size+1,num_channels))
    for k in range(num_outputs):
        delta = delta + par_u.reshape(num_outputs)[k]*model['W'][k]
  
    
    deri_sig = np.multiply(H,(1-H))
    sig_delta = np.multiply(deri_sig,delta)
    sig_delta = sig_delta.reshape(num_channels,num_inputs-filter_size+1,num_inputs-filter_size+1)
    for p in range(num_channels):
        for i in range(filter_size):
            for j in range(filter_size):
                mul_x = x[i:i+num_inputs-filter_size+1, j:j+num_inputs-filter_size+1]
                model_grads['K'][p][i][j]= np.tensordot(mul_x, sig_delta[p], axes=2)
    for k in range(num_outputs):
        model_grads['W'][k] = par_u.reshape(num_outputs)[k]*H

    return model_grads


# In[34]:


#Learning rate & num_epochs = 8
LR = .1
num_epochs = 8
for epochs in range(num_epochs):

#Learning rate schedule
    if (epochs > 2):
        LR = .01
    if (epochs > 4):
        LR = 0.001
    total_correct = 0
    for n in range( len(x_train)):
        n_random = randint(0,len(x_train)-1 )
        y = y_train[n_random]
        x = x_train[n_random][:]
        Z,H,U,f = forward(x, y, model)
        prediction = np.argmax(f)
        if (prediction == y):
            total_correct += 1
        model_grads = backward(x, y, f, model,model_grads)

        model['b'] = model['b'] - LR*model_grads['b']
        model['W'] = model['W'] - LR*model_grads['W']
        model['K'] = model['K'] - LR*model_grads['K']

    print(total_correct/np.float(len(x_train) ) )


# In[35]:


#print the accuracy on the test data
total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    Z,H,U,f = forward(x, y, model)
    prediction = np.argmax(f)
    if (prediction == y):
        total_correct += 1
print(total_correct/np.float(len(x_test) ) )


# In[1]:


print('Test Accuracy = 96.92%')

print('Implementation:Convolution Neural Network with single hidden') 
print( 'layer and multi-channels')
print('Randomly select a new data sample (X, Y )')
print('Compute the forward step (Z, H, U, ρ)')
print('Calculate the partial derivatives ( ∂ρ/∂U , δ, ∂ρ/∂K )')
print('Update the parameters θ = {K, W, b} with a stochastic gradient descent step')
print('Filter size: 5, number of channels: 3')

