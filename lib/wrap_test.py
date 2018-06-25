import numpy as np
import theano.tensor as T
import theano
import time,os
import theano.tensor.nnet.conv3d2d
import scipy.io as sio
import sys

import cPickle,h5py
from lib.max_pool import max_pool_3d
from lib.relu import relu
from lib.load_mat import load_mat,sharedata
floatX = theano.config.floatX

class ConvPoolLayer(object):
    def __init__(self, input, filter, base, activation, poolsize, dtype = theano.config.floatX):
        
        """
        Allocate a Conv3dLayer with shared variable internal parameters.
      
        :type input: theano.tensor
        :param input: 5D matrix -- (batch_size, time, in_channels, height, width)
        
        :type filter: 
        :param filter: 5D matrix -- (num_of_filters, flt_time, in_channels, flt_height, flt_width)
        
        :type filters_shape: tuple or list of length 5
        :param filter_shape:(number_of_filters, flt_time,in_channels,flt_height,flt_width)
        
        :type base: tuple or list of length number_of_filters
        :param base:(number_of_filters)
        
        :param activation: non-linear activation function, typically relu or tanh 
        
        :poolsize: tuple or list of length 3
        :param poolsize: the pooling stride, typically (2,2,2)              
        """
        
        self.input = input#shared variable el self.input #solo la primera vez. 
        self.W = filter#Theano <class 'theano.tensor.var.TensorVariable'>
        self.b = base #Shared 
        
        # do the 3d convolution --- have flip
        conv_out = theano.tensor.nnet.conv3d2d.conv3d(
            signals = self.input,   
            filters = self.W,  #Tensor Variable ya no es shared anymore       DISTINTO QE EN CNN3D
            signals_shape = None,
            filters_shape = None,
            border_mode = 'valid')  # the convolution stride is 1
        #conv_out = array([  1,  76,  64, 176, 176]) <class 'theano.tensor.var.TensorVariable'>
        #Aqui convout se puede evaluar lul!
        #2 array([ 1, 36, 64, 86, 86])
        #3 array([ 1, 36, 64, 84, 84])
        #4 array([ 1, 35, 150,83, 83])
        #5 array([ 1, 35,  2, 83, 83])



        conv = conv_out + self.b.dimshuffle('x','x',0,'x','x') 
        
        if poolsize is None:
            pooled_out = conv
        else:
            pooled_out = max_pool_3d(input=conv, ds=poolsize, ignore_border=True)
            #array([ 1, 38, 64, 88, 88])
            #<class 'theano.tensor.var.TensorVariable'>

            #2array([ 1, 36, 64, 86, 86])


        # non-linear function
        self.output = ( 
            pooled_out if activation is None 
            else activation(pooled_out)
        )
        
       # store parameters of this layer
        self.params = [self.W, self.b]
        
class LogisticRegression(object):
    def __init__(self,input,x,y,z): #input es theano.tensor.var.TensorVariable'>
        # flatten the input feature volumes into vectors
        #beofre reshape array([ 1, 35,  2, 83, 83])
        #After reshape array([241115,      2])
        self.input = input.reshape((z,2,y,x)).dimshuffle(0,2,3,1).reshape((x*y*z,2))
        # employ softmax to get prediction probabilities
        self.p_y_given_x = T.nnet.softmax(self.input) # array([241115,      2]) <class 'theano.tensor.var.TensorVariable'>

        # reshape back into score volume
        #array([ 2, 83, 83, 35]) <class 'theano.tensor.var.TensorVariable'>
        self.score_map = self.p_y_given_x.reshape((z,y,x,2)).dimshuffle(3,2,1,0)  # dimension is z y x
        #array([241115]) respuesta binaria de donde está el mayor
        self.y_pred = T.argmax(self.p_y_given_x,axis=1)
    
class wrap_3dfcn(object):
    def __init__(self, input, layer_num, maxpool_sizes, activations, dropout_rates,
                para_path, final_size, show_param_label=False):
        """
        This is to efficiently wrap the whole volume with 3D FCN
        
        :type input: theano.tensor
        :param input: 5D matrix -- (batch_size, time, in_channels, height, width)
        
        :type layer_num: int
        :param layer_num: number of layers in the network
        
        :type maxpool_sizes: list
        :param maxpool_sizes: maxpooling sizes of each layer
        
        :param activations: non-linear activation function, typically relu or tanh
        
        :type dropout_rates: list
        :param dropout_rates: dropout rate of each layer
        
        :param para_path: saved model path
        
        :type final_size: list of length 3
        :param final_size: output score volume size -- (final_time, final_height, final_width) 
        """
        
        f = open(para_path,'r') 
        params = cPickle.load(f) 
        if show_param_label:
            print 'params loaded!' 
                      
        self.layers = []
        next_layer_input = input #array([  1,  78,   1, 180, 180]) shared variable
        for layer_counter in range(layer_num):
            W = params[layer_counter*2]
            #1er array([64,  3,  1,  5,  5]) shared
            #2nd array([64,  3, 64,  3,  3])
            #3er array([64,  1, 64,  3,  3])





            #4   array([150,   2,  64,   2,   2])
            #5   array([  2,   1, 150,   1,   1])
            b = params[layer_counter*2+1]
            #1er [64] shared
            #2nd [64]
            #3er 64
            #4 150
            #5 array([2])

            if show_param_label:
                print 'layer number:{0}, size of filter and base: {1} {2}'.format(layer_counter, W.shape.eval(), b.shape.eval())
            next_layer = ConvPoolLayer(
                    input = next_layer_input,#only the fist time is a shared variable, sin embargo luego va teniendo valores.... 
                    filter = W*(1-dropout_rates[layer_counter]),#When multiply goes from shared to theano tensor
                    base = b,
                    activation = activations[layer_counter],
                    poolsize = maxpool_sizes[layer_counter])
            
            self.layers.append(next_layer)
            next_layer_input = next_layer.output# <class 'theano.tensor.var.TensorVariable'>

            layer_counter += 1
        
        final_time, final_height, final_width = final_size
        score_volume_layer = LogisticRegression(
                input = self.layers[-1].output,
                x = final_width,
                y = final_height,
                z = final_time)
                    
        self.score_volume = score_volume_layer.score_map            
                    
                    
def test_wrapper(input_sizes,output_sizes,patch_size,clip_rate,M_layer,layer_num,maxpool_sizes,activations,dropout_rates,
                para_path,save_score_map_path,whole_volume_path,mode):    
                            
    files = os.listdir(whole_volume_path)
    n_cases = len(files)
    print 'Have {} cases to process'.format(n_cases)               
           
    in_height, in_width, in_time = input_sizes
    
    for case_counter in xrange(n_cases):
        print 'Processing case # {} ... '.format(case_counter + 1)
        # cut the whole volume into smaller blocks, otherwise GPU will be out of memory
        dim0_score_start_pos = []
        dim0_score_end_pos = []
        dim0_start_pos = []
        dim0_end_pos = []  
        for part in range(clip_rate[0]):
            dim0_score_start_pos.append(1+part*output_sizes[0]/clip_rate[0])
            dim0_score_end_pos.append((part+1)*output_sizes[0]/clip_rate[0])
            dim0_start_pos.append(2*M_layer*(1+part*output_sizes[0]/clip_rate[0]-1)+1)
            dim0_end_pos.append(2*M_layer*((part+1)*output_sizes[0]/clip_rate[0]-1)+patch_size[0])   
        dim0_pos = zip(dim0_start_pos,dim0_end_pos)
        dim0_score_pos = zip(dim0_score_start_pos,dim0_score_end_pos)

        dim1_score_start_pos = []
        dim1_score_end_pos = []
        dim1_start_pos = []
        dim1_end_pos = []
        for part in range(clip_rate[1]):
            dim1_score_start_pos.append(1+part*output_sizes[1]/clip_rate[1])
            dim1_score_end_pos.append((part+1)*output_sizes[1]/clip_rate[1])
            dim1_start_pos.append(2*M_layer*(1+part*output_sizes[1]/clip_rate[1]-1)+1)
            dim1_end_pos.append(2*M_layer*((part+1)*output_sizes[1]/clip_rate[1]-1)+patch_size[1])   
        dim1_pos = zip(dim1_start_pos,dim1_end_pos)
        dim1_score_pos = zip(dim1_score_start_pos,dim1_score_end_pos)

        dim2_score_start_pos = []
        dim2_score_end_pos = []
        dim2_start_pos = []
        dim2_end_pos = []
        for part in range(clip_rate[2]):
            dim2_score_start_pos.append(1+part*output_sizes[2]/clip_rate[2])
            dim2_score_end_pos.append((part+1)*output_sizes[2]/clip_rate[2])
            dim2_start_pos.append(2*M_layer*(1+part*output_sizes[2]/clip_rate[2]-1)+1)
            dim2_end_pos.append(2*M_layer*((part+1)*output_sizes[2]/clip_rate[2]-1)+patch_size[2])   
        dim2_pos = zip(dim2_start_pos,dim2_end_pos)
        dim2_score_pos = zip(dim2_score_start_pos,dim2_score_end_pos)

        """
        dim0_score_pos [(1, 83), (84, 166), (167, 249)]
        dim0_pos [(1, 180), (167, 346), (333, 512)]

        dim1_pos [(1, 180), (167, 346), (333, 512)]
        dim1_score_pos [(1, 83), (84, 166), (167, 249)]

        dim2_pos [(1, 78), (71, 148)]
        dim2_score_pos [(1, 35), (36, 70)]

        """


        score_mask = np.zeros((2,output_sizes[0],output_sizes[1],output_sizes[2])) #(2, 249, 249, 70)

        data_path = whole_volume_path + str(case_counter+1) + '_' + mode + '.mat'
        data_set = np.transpose(np.array(h5py.File(data_path)['data']))   #(1, 38797312)   
        data_set = data_set - np.mean(data_set) 
        data_set = data_set.reshape((data_set.shape[0],in_time,1,in_height,in_width)) #(1, 148, 1, 512, 512)

        for dim2 in range(clip_rate[2]):
            for dim1 in range(clip_rate[1]):
                for dim0 in range(clip_rate[0]):
                    #sys.stdout.write('.')
                    #(1 , 78, 1 ,180, 180) smallerdata
                    smaller_data = data_set[:,dim2_pos[dim2][0]-1:dim2_pos[dim2][1],:,dim1_pos[dim1][0]-1:dim1_pos[dim1][1],dim0_pos[dim0][0]-1:dim0_pos[dim0][1]]                    
                    wrapper = wrap_3dfcn(input = theano.shared(np.asarray(smaller_data,theano.config.floatX),borrow = True),
                                        layer_num = layer_num,
                                        maxpool_sizes = maxpool_sizes,
                                        activations = activations,
                                        dropout_rates = dropout_rates,
                                        para_path = para_path,
                                        final_size = (dim2_score_end_pos[0], dim0_score_end_pos[0], dim1_score_end_pos[0]))                                       
                    #tal y como está hecho esto sobra totalmente. Ya tenemos el resultado en wrapper
                    test_model = theano.function(inputs = [], outputs = wrapper.score_volume)                      
                    smaller_score = test_model()
                    score_mask[:,dim0_score_pos[dim0][0]-1:dim0_score_pos[dim0][1],dim1_score_pos[dim1][0]-1:dim1_score_pos[dim1][1],dim2_score_pos[dim2][0]-1:dim2_score_pos[dim2][1]] = smaller_score
                    
        result_file_name = save_score_map_path + str(case_counter+1) + '_score_mask.mat'
        print 'The score_mask saved path:', result_file_name
        sio.savemat(result_file_name,{'score_mask':score_mask})
        print 'Case {} wrap over!'.format(case_counter+1)
    
                      
