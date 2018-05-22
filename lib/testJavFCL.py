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
from lib.load_mat import load_mat,sharedata
floatX = theano.config.floatX


def model (X):
    
	model_path = '../model/fine_tuned_params_step2.pkl'
	f_param = open(model_path,'r')
	params = cPickle.load(f_param)
	print 'params legnth:',len(params)

	params_L0, params_L1, params_L2, params_L3, params_L4 = [param for param in params]

	W_L0 = params_L0[0]#([64,  3,  1,  5,  5])#( num_of_filters, flt_time, in_channels, flt_height, flt_width)
	b_L0 = params_L0[1]#[64] 

	W_L1 = params_L1[0]#([64,  3, 64,  3,  3])
	b_L1 = params_L1[1]#[64]

	W_L2 = params_L1[0]#([64,  1, 64,  3,  3])
	b_L2 = params_L1[1]#[64]
	
	W_L3_aux=params_L2[0]#([150,   2,  64,   2,   2])
	#We need to reshape it into (512x150)

	W_L3_aux.dimshuffle(1,2,3,4,0)
	W_L3.W_L3_aux.reshape(:.150)



	W_L3 = params_L2[0]#(512, 150) 
	b_L3 = params_L2[1]#(500,)

	W_L4 = params_L3[0]#(500, 100)
	b_L4 = params_L3[1]#(100,)

	f_param.close()
    print 'params loaded!'

