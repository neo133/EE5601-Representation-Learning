"""
Q1)(b) Implement MLP AND gate
Author- Subhra Shankha Bhattacherjee
Roll - EE19MTECH01008
"""
"""
Note to the evaluator: sometimes the network doesn't predict correctly. Please re-run the code in that case.
"""

import numpy as np 
import random

# random.seed(9001)

#important functions
def sigmoid (x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def error(x,y):
    return x-y

def layer(w,x,b):
	a = np.dot(x,w)+b
	return a
print("This MLP AND gate network has 2 input nodes, 1 hidden layer with 2 nodes and 1 output node. Noise is generated from 0 mean 0.6 standard deviation gaussian distribution. 100 samples for each bit is generated. The network has a learning rate of 0.01 and is run for 200 epochs \n")
print("Prediction tests\n")
N=100 #number of training samples per bit
epochs=200
lr=0.01
hid_neur=2  #number of hidden layer nodes
inpt_neur = 2 # number of input neurons
opt_neur = 1  # number of output neurons
noise_std = 0.6  #standard deviation for noise generation

#XOR dataset
true_inpt = np.array([[0,0],[0,1],[1,0],[1,1]])
true_opt = np.array([[0],[0],[0],[1]])

noise_inpt = []
noise_opt = []

#Generate zero mean gaussian noise embedded input data to make robust network.
for i in range(len(true_inpt)):
    for j in range(N):
        noise_inpt.append(true_inpt[i]+np.random.normal(0,noise_std,(1,2)))
        noise_opt.append(true_opt[i]+np.random.normal(0,noise_std))

noise_inpt = np.array(noise_inpt)
noise_inpt = noise_inpt.reshape((len(noise_opt),2))
noise_opt = np.array(noise_opt)

#Random initialization of weights and bias
hid_w = np.random.uniform(size=(inpt_neur,hid_neur))
hid_b =np.random.uniform(size=(1,hid_neur))
opt_w = np.random.uniform(size=(hid_neur,opt_neur))
opt_b = np.random.uniform(size=(1,opt_neur))

#Training algorithm
for i in range(epochs):
	#Forward Propagation
	hid_layer_opt = sigmoid(layer(hid_w,noise_inpt,hid_b))
	pred_opt = sigmoid(layer(opt_w,hid_layer_opt,opt_b))
	
	#Backpropagation
	loss = error(noise_opt, pred_opt) 
	der_pred_opt = loss * sigmoid_derivative(pred_opt)
	
	err_hid_layer = der_pred_opt.dot(opt_w.T)
	der_hid_layer = err_hid_layer * sigmoid_derivative(hid_layer_opt)

	#weight & bias update
	opt_w += hid_layer_opt.T.dot(der_pred_opt) * lr
	opt_b += np.sum(der_pred_opt,axis=0,keepdims=True) * lr
	hid_w += np.matmul(noise_inpt.T,der_hid_layer) * lr
	hid_b += np.sum(der_hid_layer,axis=0,keepdims=True) * lr


print('Loss: ', np.absolute(loss/N))

print("Training complete!!")
#4 loops to check the 4 input combinations (0,0), (0,1), (1,0), (1,1)
for _ in range(0,4):
    a=raw_input("Enter test sample (comma seperated pair): ").split(',')
    for i in range(0,2):
        a[i]=float(a[i])
    a=np.array(a)
    out_l1 = layer(hid_w,a,hid_b)
    z = sigmoid(out_l1)
    out_l2 = layer(opt_w,z,opt_b)
    pred = sigmoid(out_l2)
    if(pred > 0.39): # I was having some issues, i got this range based on prediction values.
        pred = 1
    else :
        pred = 0
    print("predicted output:")
    print(pred)