import math
import numpy as np
import h5py
import cPickle
import matplotlib.pyplot as plt
import tensorflow as tf
import pdb
from Bachitazion import Bachitazion
import pickle




##data_path = whole_volume_path + str(74) + '.mat'
#data_set = np.transpose(np.array(h5py.File(data_path)['patchFlatten']))
#image =  data_set.reshape((data_set.shape[0],10,16,16,1))
def save_to_file(filename, object):
    """ Save object to file
    """
    f = open(filename + '.pckl', 'wb')
    pickle.dump(object, f)
    f.close()

def createPlaceHolders(n_H,n_W,n_C,n_D,n_Y):
    """
    Create placeholder for the session
    
    Arguments:
        n_H -- Height of the input image 
        n_W -- Width of the input image
        n_D -- Depth of the input image
        n_C -- Channels of the input image
        n_y -- number of classes
    """
    X = X = tf.placeholder(tf.float32,[None,n_D,n_H,n_W,n_C],name="X")
    Y = tf.placeholder(tf.float32,[None,n_Y],name = "Y")
    return X,Y

def initializeWeights(preTrained = True,path='/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/PesosPruebaTf/'):
    if preTrained:
        W_L0 = np.load(path+'W_L0.npy')
        b_L0 = np.load(path+'b_L0.npy')

        W_L1 = np.load(path+'W_L1.npy')
        b_L1 = np.load(path+'b_L1.npy')

        W_L2 = np.load(path+'W_L2.npy')
        b_L2 = np.load(path+'b_L2.npy')

        W_L3 = np.load(path+'W_L3.npy')
        b_L3 = np.load(path+'b_L3.npy')

        W_L4 = np.load(path+'W_L4.npy')
        b_L4 = np.load(path+'b_L4.npy')

        W0 = tf.Variable(W_L0, name="W0",trainable = False)
        W1 = tf.Variable(W_L1, name="W1",trainable = False)
        W2 = tf.Variable(W_L2, name="W2",trainable = False)

        W3 = tf.Variable(W_L3, name="W3")
        W3 = tf.reshape(W3,shape=[512,150])
        W4 = tf.Variable(W_L4, name="W4")
        W4 = tf.reshape(W4,shape=[150,2])

        b0 = tf.Variable(b_L0, name ="b0",trainable = False)
        b1 = tf.Variable(b_L1, name ="b1",trainable = False)
        b2 = tf.Variable(b_L2, name ="b2",trainable = False)
        b3 = tf.Variable(b_L3, name ="b3")
        b4 = tf.Variable(b_L4, name ="b4")      


    #Would be perfect to define another part if it was created from scratch, but still to do     

    parameters={"W0":W0,"b0":b0,"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3,"W4":W4,"b4":b4}

    return parameters

def computeCost(Z5,Y):

    """
    Computes the cost
    
    Arguments:
    Z5 -- output of forward propagation 
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z5,labels=Y))

    return cost




#Let it like this and call this function with **parameters should be fine....
#def forward_propagation(X, W_L0,b_L0,W_L1,b_L1,W_L2,b_L2,W_L3,b_L3,W_L4,b_L4):
def forward_propagation(X,parameters):
    #This could have been done better defininf a Customized Convolution Layer and the same for flatten layer. 
    #I mean, the result is the same but is more structured. 

    W_L0 = parameters["W0"]
    b_L0 = parameters["b0"]

    W_L1 = parameters["W1"]
    b_L1 = parameters["b1"]

    W_L2 = parameters["W2"]
    b_L2 = parameters["b2"]

    W_L3 = parameters["W3"]
    b_L3 = parameters["b3"]

    W_L4 = parameters["W4"]
    b_L4 = parameters["b4"]


    # Retrieve the parameters from the dictionary "parameters" 
    
    ### START CODE HERE ###
    # CONV2D: stride of 1, padding 'SAME
    Z1 =tf.nn.conv3d(X,W_L0,strides=[1,1,1,1,1],padding='VALID') 
    Z1 = tf.nn.bias_add(Z1,b_L0)
    Z1= tf.nn.max_pool3d(Z1,ksize=(1,2,2,2,1),strides=(1,2,2,2,1),padding='VALID')
    A1 = tf.nn.relu(Z1)

    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    Z2 =tf.nn.conv3d(A1,W_L1,strides=[1,1,1,1,1],padding='VALID') 
    Z2 = tf.nn.bias_add(Z2,b_L1)
    A2 = tf.nn.relu(Z2)

    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z3 =tf.nn.conv3d(A2,W_L2,strides=[1,1,1,1,1],padding='VALID') 
    Z3 = tf.nn.bias_add(Z3,b_L2)
    A3 = tf.nn.relu(Z3)
    P3 = tf.contrib.layers.flatten(A3)

    # RELU
    Z4 = tf.add(tf.matmul(P3,W_L3),b_L3)
#   Z4 =tf.nn.conv3d(A3,W_L3,strides=[1,1,1,1,1],padding='VALID') 
#   Z4 = tf.nn.bias_add(Z4,b_L3)
    A4 = tf.nn.relu(Z4)

    #Z5=tf.nn.conv3d(A4,W_L4,strides=[1,1,1,1,1],padding='VALID') 
    #Z5 = tf.nn.bias_add(Z5,b_L4)
    Z5 =tf.add(tf.matmul(A4,W_L4),b_L4)

    return Z5




def train(learning_rate=0.001,num_epochs = 50):

    myBatchGenerator = Bachitazion(sizeOfBatch=128,pathT='/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/data/AllPatchesWithMicrobleedsTrain/')
    err_val={}
    acc_val={}

    X, Y = createPlaceHolders(16,16,1,10,2)
    parameters = initializeWeights()
    Z5 = forward_propagation(X,parameters)
    cost = computeCost(Z5, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
     # Calculate the correct predictions
    correctPrediction = tf.equal(tf.argmax(Z5,1), tf.argmax(Y,1))
    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, "float"))
    #print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
    init = tf.initialize_all_variables()

    with tf.Session() as sess:   
        sess.run(init)
        for epoch in range(num_epochs):

            trainingCost = 0.
            trainingAcc = 0.
            for batch in range(int(myBatchGenerator.number_batchesT)+1):
                miniX,miniY = myBatchGenerator.nextBatch_T()
                _ , batchTrainingCost,batchTrainingAcc = sess.run([optimizer, cost,accuracy], feed_dict={X: miniX, Y: miniY})
                trainingCost += batchTrainingCost
                trainingAcc += batchTrainingAcc
            trainingCost /= myBatchGenerator.number_batchesT
            trainingAcc /= myBatchGenerator.number_batchesT

            print("End of epoch {0:02d}: err(train) = {1:.2f} acc(train) = {2:.2f}".format(epoch+1,trainingCost,trainingAcc))



            evalCost = 0.
            evalAcc = 0.
            for batch in range(int(myBatchGenerator.number_batchesE)+1):
                miniX,miniY = myBatchGenerator.nextBatch_E()
                batchEvalCost,batchEvalAcc = sess.run([cost,accuracy], feed_dict={X: miniX, Y: miniY})
                evalCost += batchEvalCost
                evalAcc += batchEvalAcc
            evalCost /= myBatchGenerator.number_batchesE
            evalAcc /= myBatchGenerator.number_batchesE

            print("                 err(eval) = {1:.2f} acc(eval) = {2:.2f}".format(epoch+1,evalCost,evalAcc))
            err_val[epoch + 1] = evalCost
            acc_val[epoch + 1] = evalAcc
    sess.close()
    save_to_file("FileToPlotCost",err_val)
    save_to_file("FileToPlotEval",acc_val)

train()
