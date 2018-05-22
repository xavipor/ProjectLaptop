import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet.conv3d2d
import os.path as path
from lib.relu import relu
from lib.max_pool import max_pool_3d
  
class ConvPoolLayer(object):
    def __init__(self, rng, input, input_shape, filter, filter_shape, base, activation, poolsize, dtype = theano.config.floatX):
                
        self.input = input   
        self.W = filter    
        self.b = base
        # do the convolution --- have flip
        conv_out = theano.tensor.nnet.conv3d2d.conv3d(
            signals = self.input, #( batch_size, time, in_channels, height, width )
            filters = self.W, #( num_of_filters, flt_time, in_channels, flt_height, flt_width)
            signals_shape = input_shape,
            filters_shape = filter_shape,
            border_mode = 'valid') # the conv stride is 1
        
        conv = conv_out + self.b.dimshuffle('x','x',0,'x','x')
        
        if poolsize is None:
            pooled_out = conv
        else:
            pooled_out = max_pool_3d( input = conv, ds = poolsize, ignore_border = True)
        
        # non-linear function
        self.output = ( 
            pooled_out if activation is None 
            else activation(pooled_out)
        )
        
       # store parameters of this layer
        self.params = [self.W, self.b]


class HiddenLayer(object):
    def __init__(self,rng,input,n_in,n_out,W,b,activation=relu):
        self.input=input
        self.W = W
        self.b = b
        
        lin_output = T.dot(input,self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]

   
class LogisticRegression(object):
    def __init__(self,input,shape):
        self.input = input.reshape(shape)
        self.p_y_given_x = T.nnet.softmax(self.input)
        self.y_pred = T.argmax(self.p_y_given_x,axis=1)
