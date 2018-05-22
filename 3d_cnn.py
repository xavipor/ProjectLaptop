import os
import sys
import time
import numpy
import theano
import theano.tensor.nnet.conv3d2d
import theano.tensor as T
import scipy.io as sio
import matplotlib.pyplot as plt
from lib.relu import relu
from lib.max_pool import max_pool_3d
from lib.load_mat import *
import cPickle
floatX = theano.config.floatX


class HiddenLayer(object):
    def __init__(self, input, W, b, activation=relu):
        self.input=input
        self.W = W
        self.b = b
        
        lin_output = T.dot(input,self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]

class ConvPool3dLayer(object):
    def __init__(self, input, W, b, filter_shape, poolsize=(2,2,2),activation=relu):
        
        self.input = input #Lo mismo el shape.eval() aqui tampoco funciona y es del tipo theano.tensor.var.TensorVariable
        #hacen esto para convertir lo que llega en W que es un CudaArray no una shared variable en 
        #una shared variable como era params_L0[0] Osea hacen una cosa para desacerla no muy sense...
        self.W = theano.shared(value = W, borrow = True)
        self.b = theano.shared(value = b, borrow = True)
        
        # conv tampoco funciona el shape y es...<class 'theano.tensor.var.TensorVariable'>
        conv = T.nnet.conv3d2d.conv3d(
            signals = self.input,#( batch_size, time, in_channels, height, width )
            filters = self.W,#( num_of_filters, flt_time, in_channels, flt_height, flt_width)
            signals_shape = None,
            filters_shape = filter_shape,
            border_mode = 'valid')              
        pooled_out = max_pool_3d(
            input = conv,
            ds = poolsize,
            ignore_border = True)
        
        # non-linear function
        #Aqui esto está bien lo que no se es si esta bien hacer lo de sumar la b antes del pooling
        # por supuerto en output tampoco va shape.eval() <class 'theano.tensor.var.TensorVariable'>
        self.output = activation( pooled_out + self.b.dimshuffle('x','x',0,'x','x')) # arg activitaion function
        # store parameters of this layer
        self.params = [self.W, self.b]

class LogisticRegression(object):
    def __init__(self, input, W, b):
        self.W = W
        self.b = b
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.positive_prob = self.p_y_given_x[:,1]
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

def sharedata(data):
    # This function is intended to invert the numpy.ndarray data into theano.shared data
    return theano.shared(numpy.asarray(data,dtype = theano.config.floatX),borrow = True)

def evaluate_cnn3d(learning_rate = 0.03, n_epochs = 30, batch_size = 1):

    results_path = '../result/final_prediction/'        
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        
    model_path = '../model/fine_tuned_params_step2.pkl'
    f_param = open(model_path,'r')
    params = cPickle.load(f_param)
    print 'params legnth:',len(params)

    params_L0, params_L1, params_L2, params_L3, params_L4 = [param for param in params]
    
    W_L0 = params_L0[0].eval() #CudaArray (32, 5, 1, 7, 7) sin embargo params_L0[0] es un theano.sandbox.cuda.var.CudaNdarraySharedVariable' de dimensiones array([32,  5,  1,  7,  7])
    b_L0 = params_L0[1].eval()#aqui params_L0[1] es un array ([32]) cuda sandbox blabla y b_L0 es un (32,n)
    #Misma historia para los otros, ahora voy a poner solo las dimensiones
    W_L1 = params_L1[0].eval()#(64, 3, 32, 5, 5)
    b_L1 = params_L1[1].eval()#(64,)

    W_L2 = params_L2[0].eval()#(2304, 500)
    b_L2 = params_L2[1].eval()#(500,)

    W_L3 = params_L3[0].eval()#(500, 100)
    b_L3 = params_L3[1].eval()#(100,)

    W_L4 = params_L4[0].eval()#(100, 2)
    b_L4 = params_L4[1].eval()#(2,)

    f_param.close()
    print 'params loaded!'

    print 'weights shape:', W_L0.shape, W_L1.shape, W_L2.shape, W_L3.shape, W_L4.shape
    cand_num = theano.shared(np.asarray(1,dtype='int32'))#array([], dtype=int64)  <class 'theano.tensor.sharedvar.TensorSharedVariable'>
    test_set_x = sharedata(data=numpy.ones([1,20*20*16]))#array([1, 6400]) <class 'theano.sandbox.cuda.var.CudaNdarraySharedVariable'>
    print 'prepare data done'
    
    in_channels = 1
    in_time = 16
    in_width = 20
    in_height = 20
    
    x = T.matrix('x')#<class 'theano.tensor.var.TensorVariable'> x.eval() doesnt work undeclared input
    y = T.ivector('y')#<class 'theano.tensor.var.TensorVariable'>
    batch_size = T.iscalar('batch_size')#<class 'theano.tensor.var.TensorVariable'>

    #define filter shape of the first layer
    flt_channels_L0 = 32
    flt_time_L0 = 5
    flt_width_L0 = 7
    flt_height_L0 = 7
    #(32,5,1,7,7)
    filter_shape_L0 = (flt_channels_L0,flt_time_L0,in_channels,flt_height_L0,flt_width_L0)
    
    #define filter shape of the second layer
    flt_channels_L1 = 64
    flt_time_L1 = 3
    flt_width_L1 = 5
    flt_height_L1 = 5
    in_channels_L1 = flt_channels_L0
    #(64, 3, 32, 5, 5)
    filter_shape_L1 = (flt_channels_L1,flt_time_L1,in_channels_L1,flt_heights_L1,flt_width_L1)
    #imShuffle{0,2,1,3,4}.0 esto es lo que es layer0_input y de tipo  <class 'theano.tensor.var.TensorVariable'> como X
    layer0_input = x.reshape((batch_size,in_channels,in_time,in_height,in_width)).dimshuffle(0,2,1,3,4)
    #Sobre layer0_input.shape.eval() no funciona por que dice que no sstá deficinido, que es parte del grafico y tal 
    #Tipo theano.tensor.var.Tensorvariable

    layer0 = ConvPool3dLayer(
        input = layer0_input,
        W = W_L0,
        b = b_L0,
        filter_shape = filter_shape_L0,
        poolsize=(2,2,2))

    layer1 = ConvPool3dLayer(
        input = layer0.output, #<class 'theano.tensor.var.TensorVariable'> pero en principio no es nada es un elemwise
        W = W_L1,
        b = b_L1,
        filter_shape = filter_shape_L1,
        poolsize = (1,1,1))
        
    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(
        input = layer2_input,
        W = W_L2,
        b = b_L2)
    
    layer3 = HiddenLayer(
        input = layer2.output,
        W = W_L3,
        b = b_L3)
    
    layer4 = LogisticRegression(input = layer3.output, W=W_L4, b=b_L4)
        
    test_model = theano.function(
        inputs = [],
        outputs = [layer4.positive_prob,layer3.output],
        givens = {x: test_set_x, batch_size: cand_num})
    
    
    print '...testing...'
    datapath = '../result/test_set_cand/'
    files = os.listdir(datapath)
    n_cases = len(files)
    print 'n_cases:',files
    start_time = time.time()
    for cs in xrange(n_cases):
        case = cs + 1
        set_x = np.array(h5py.File(datapath + str(case) + '_patches.mat')['test_set_x']) #(6400, 317) <type 'numpy.ndarray'>
        set_x = np.transpose(set_x) - np.mean(set_x)   #317x6400 ... Why not STD??
        print 'predicting {0} subject, contains {1} candidates...'.format(case, set_x.shape[0])
        cand_numcd .set_value(set_x.shape[0])#array(317, dtype=int32)
        test_set_x.set_value(set_x.astype(floatX)) #array([ 317, 6400])
        prediction, feature = test_model()    #prediction (317,) 
        sio.savemat(results_path + str(case)+'_prediction.mat',{'prediction':prediction})                 
    end_time = time.time()
    print 'time spent {} seconds.'.format((end_time-start_time)/n_cases)
 
    
if __name__ == '__main__':
    try:
        evaluate_cnn3d()
    except KeyboardInterrupt:
        sys.exit()
