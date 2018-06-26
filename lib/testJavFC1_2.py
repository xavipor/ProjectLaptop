"""
	Code based on:
		deeplearning.net
		https://github.com/Newmu/
		Automatic Detection of Cerebral Microbleeds From MRI via 3D CNN 2016 Qi Dou, Hao Chen

"""
import numpy as np
import theano.tensor as T
import theano
import time,os
import theano.tensor.nnet.conv3d2d
import scipy.io as sio
import sys
import pdb

import cPickle,h5py
from lib.max_pool import max_pool_3d
from lib.relu import relu
from lib.relu import dropout
from lib.load_mat import load_mat,sharedata
import Bachitazion
floatX = theano.config.floatX

class myConvPool3dLayer(object):
	def __init(self,input,W,b,filter_shape,poolsize=(1,1,1),activation=relu):

		self.input = input
		self.W = theano.shared(numpy.asarray(W,dtype = theano.config.floatX),borrow=True)
		self.b = theano.shared(numpy.asarray(W,dtype = theano.config.floatX),borrow=True)
		conv = T.nnet.conv3d2d.conv3d(
            signals = self.input,
            filters = self.W,
            signals_shape = None,
            filters_shape = filter_shape,
            border_mode = 'valid')

        pooled_out = max_pool_3d(
            input = conv,
            ds = poolsize,
            ignore_border = True)

        #After convolution and pooling if needed (if pooling = (1,1,1)) is like doing nothing

		self.output = activation( pooled_out + self.b.dimshuffle('x','x',0,'x','x'))

		self.params = [self.W, self.b]

class myHiddenLayer(object):
    def __init__(self, input, W, b, activation=relu):
        self.input=input
        self.W = theano.shared(numpy.asarray(W,dtype = theano.config.floatX),borrow=True)
        self.b = theano.shared(numpy.asarray(W,dtype = theano.config.floatX),borrow=True)
        
        lin_output = T.dot(input,self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]

class myLogisticRegression(object):
    def __init__(self, input, W, b):
    	self.input = input
        self.W = theano.shared(numpy.asarray(W,dtype = theano.config.floatX),borrow=True)
        self.b = theano.shared(numpy.asarray(W,dtype = theano.config.floatX),borrow=True)
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.positive_prob = self.p_y_given_x[:,1]
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]



def RMSprop(cost, params, lr=0.01, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def evaluate_myNet(learning_rate=0.01,n_epochs=200,batch_size=64,pathT="/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/AllPatchesWithMicrobleedsTrain/"):

	in_channels = 1
    in_time = 10
    in_width = 16
    in_height = 16

    #define filter shape of the first layer
    flt_channels_L0 = 64
    flt_time_L0 = 3
    flt_width_L0 = 5
    flt_height_L0 = 5
    filter_shape_L0 = (flt_channels_L0,flt_time_L0,in_channels,flt_height_L0,flt_width_L0)
    
    #define filter shape of the second layer
    flt_channels_L1 = 64
    flt_time_L1 = 3
    flt_width_L1 = 3
    flt_height_L1 = 3
    in_channels_L1 = flt_channels_L0
    filter_shape_L1 = (flt_channels_L1,flt_time_L1,in_channels_L1,flt_heights_L1,flt_width_L1)

    #define filter shape of the second layer
    flt_channels_L2 = 64
    flt_time_L2 = 1
    flt_width_L2 = 3
    flt_height_L2 = 3
    in_channels_L2 = flt_channels_L1
    filter_shape_L2 = (flt_channels_L2,flt_time_L2,in_channels_L2,flt_heights_L2,flt_width_L2)


	myBatchCreator = Bachitazion.Bachitazion(batch_size,pathT)

	model_path = '/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/model/fine_tuned_params_step2.pkl'
	f_param = open(model_path,'r')
	params = cPickle.load(f_param)


	params_L0, params_L1, params_L2, params_L3, params_L4 = [param for param in params]
	#( num_of_filters, flt_time, in_channels, flt_height, flt_width)
	W_L0 = (1-0.2)*params_L0[0].eval()#([64,  3,  1,  5,  5])#
	b_L0 = params_L0[1].eval()#[64] 

	W_L1 = (1-0.3)*params_L1[0].eval()#([64,  3, 64,  3,  3])
	b_L1 = params_L1[1].eval()#[64]

	W_L2 = (1-0.3)*params_L1[0].eval()#([64,  1, 64,  3,  3])
	b_L2 = params_L1[1].eval()#[64]
	
	W_L3 = theano.shared((numpy.random.randn((512, 150)) * theano.tensor.sqrt(2 / (512 + 150))).astype(theano.config.floatX))
	b_L3 = theano.shared(numpy.zeros(150))

	W_L4 = theano.shared((numpy.random.randn((150, 2)) * theano.tensor.sqrt(2 / (150 + 2))).astype(theano.config.floatX))
	b_L4 = theano.shared(numpy.zeros(2))

	f_param.close()
    print 'params loaded!'

    #Simbolic variables for the data
    index = T.lscalar() #index for the minibatch.
    x = T.matrix('x')
    y = T.ivector('y')

   #Define the input of the net, in Theano, the number of dimesions is fixed, but the lenght of those, 
   #can change. Different from TF where we need to specify a None if we are not sure about the dimensions. 
   #The shuffle is needed due to how the convolution function accepts the input. 
   layer0_input = x.reshape((batch_size,in_channels,in_time,in_height,in_width)).dimshuffle(0,2,1,3,4)

   layer0 = myConvPool3dLayer(input =layer0_input,W=W_L0,b=b_L0,filter_shape = filter_shape_L0,poolsize=(2,2,2),activation=relu)

   layer1 = myConvPool3dLayer(input =layer0_output,W=W_L1,b=b_L1,filter_shape = filter_shape_L1,poolsize=(1,1,1),activation=relu)

   layer2 = myConvPool3dLayer(input =layer1_output,W=W_L2,b=b_L2,filter_shape = filter_shape_L2,poolsize=(1,1,1),activation=relu)


   layer3_input = layer2.output.flatten(2)
   
   layer3 = myHiddenLayer(input = layer3_input,W=W_l3,b=b_L3,activation = relu)

   layer4_input = dropout(layer3_output,0.3 )

   layer4 = myLogisticRegression(input = layer4_input, W = W_L4,b = b_L4)

   


