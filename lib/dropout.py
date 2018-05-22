import numpy as np
import theano.tensor as T
import theano
from relu import relu


def drop(input,p):
    '''
    The drop regularization
    
    :type input: numpy.array
    :param input: layer or weight matrix on which dropout resp. dropconnect is applied
    
    :type p: float or double between 0. and 1.
    :param p: p propobality of dropping out a unit or connection, therefore (1.-p) is the drop rate, typically in range [0.5 0.8]
    '''
    rng = np.random.RandomState(123456)
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1,p=1-p,size=input.shape,dtype=theano.config.floatX)
    return input * mask
        
