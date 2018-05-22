import theano.tensor as T
#from theano.tensor.signal.downsample import DownsampleFactorMax

import numpy as np
def max_pool_3d(input, ds, ignore_border=False):
    """
    Perfrom 3D max-pooling
        
    :type input: theano.tensor
    :param input: input feature volumes
    
    :type ds: tuple of length 3
    :param ds: factor by which to downscale, typically set as (2,2,2)
    
    :param ignore_border: boolean value. Example when True, (7,7,7) input with ds=(2,2,2) will generate a
    (3,3,3) output. (4,4,4) otherwise.
    """
        
    vid_dim = input.ndim
    #Maxpool frame
    frame_shape = input.shape[-2:]

    # count the number of "leading" dimensions, store as dmatrix
    batch_size = T.prod(input.shape[:-2])
    batch_size = T.shape_padright(batch_size,1)
    new_shape = T.cast(T.join(0, batch_size,T.as_tensor([1,]),frame_shape), 'int32')
    
    input_4D = T.reshape(input, new_shape, ndim=4)
    ##op = DownsampleFactorMax((ds[1],ds[2]), ignore_border) #adjust 
    op = T.signal.pool.Pool(ignore_border)
    output = op(input_4D, ws=(ds[1], ds[2]))
    # restore to original shape
    outshape = T.join(0, input.shape[:-2], output.shape[-2:])
    out = T.reshape(output, outshape, ndim=input.ndim)
    
    #Maxpool time 
    # output (time, rows, cols), reshape so that time is in the back
    shufl = (list(range(vid_dim-4)) + list(range(vid_dim-3,vid_dim))+[vid_dim-4])
    input_time = out.dimshuffle(shufl)
    # reset dimensions
    vid_shape = input_time.shape[-2:]
    # count the number of "leading" dimensions, store as dmatrix
    batch_size = T.prod(input_time.shape[:-2])
    batch_size = T.shape_padright(batch_size,1)
    # store as 4D tensor with shape: (batch_size,1,width,time)
    new_shape = T.cast(T.join(0, batch_size,T.as_tensor([1,]),vid_shape), 'int32')
    input_4D_time = T.reshape(input_time, new_shape, ndim=4)
    
    ##op=DownsampleFactorMax((1,ds[0]), ignore_border)
    #op=DownsampleFactorMax((1,ds[0]), ignore_border)
    op = T.signal.pool.Pool(ignore_border)
    outtime = op(input_4D_time,ws=(1,ds[0]))
    # restore to original shape (xxx, rows, cols, time)
    outshape = T.join(0, input_time.shape[:-2], outtime.shape[-2:])
    shufl = (list(range(vid_dim-4)) + [vid_dim-1] + list(range(vid_dim-4,vid_dim-1)))
    return T.reshape(outtime, outshape, ndim=input.ndim).dimshuffle(shufl)
