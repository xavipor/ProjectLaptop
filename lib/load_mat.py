
import scipy.io as sio
import numpy as np
import sys
import theano
import h5py
import datetime
import random

def load_mat(datafile):
    # load data from .mat file into theano.shared datatype

    def rand_dataset(data,label):      
        # this function is to shuffle the data ###
        sample = data.shape[0]
        index = np.array(range(sample))
        np.random.shuffle(index)
        rand_data = data[index]
        rand_label = label[index]
        return rand_data,rand_label

    print 'loading training dataset ...'
    start_time = datetime.datetime.now()
    
    train_data = np.array(h5py.File(datafile+'train_set_whole.mat')['train_set_x'])
    train_data = np.transpose(train_data)   
    train_data = train_data - np.mean(train_data)
    train_label = np.array(h5py.File(datafile+'train_set_whole.mat')['train_set_y'])
    train_label = np.transpose(train_label)
    train_label = train_label[:,0]    
    train_data,train_label = rand_dataset(train_data,train_label)   

    end_time = datetime.datetime.now()
    print "train data type, size:", type(train_data),train_data.shape
    print "train label type, size:",type(train_label),train_label.shape
    print "Used time: ",(end_time-start_time).seconds,'seconds.'
    
    print "loading validation data ..."
    start_time = datetime.datetime.now()
    validation_data = np.array(h5py.File(datafile+'validation_set_whole.mat')['validation_set_x'])
    validation_data = np.transpose(validation_data)
    validation_data = validation_data - np.mean(validation_data)
    validation_label = np.array(h5py.File(datafile+'validation_set_whole.mat')['validation_set_y'])
    validation_label = np.transpose(validation_label)
    validation_label = validation_label[:,0]
    validation_data,validation_label = rand_dataset(validation_data,validation_label)  
    
    end_time = datetime.datetime.now()
    print "validation data type, size:",type(validation_data),validation_data.shape
    print "validation label type,size:",type(validation_label),validation_label.shape
    print "Used time: ",(end_time-start_time).seconds,'seconds.'
    
    print 'Loading data done!'
    return train_data,train_label,validation_data,validation_label
    
def sharedata(data,label,borrow = True):
    # This function is intended to invert the numpy.ndarray data into theano.shared data
    if not label:
        return theano.shared(np.asarray(data,dtype=theano.config.floatX),borrow = True)
    elif label:
        shared_label = theano.shared(np.asarray(data,dtype=theano.config.floatX),borrow = True)
        return theano.tensor.cast(shared_label,'int32')
    else:
        print 'PLease give the flag to label e.g.sharedata(data = trainset, label = False) '
        return 0

       
if __name__ == '__main__':
    try:
        load_mat()
    except KeyboardInterrupt:
        sys.exit()
