#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 18:38:38 2018

@author: javier
"""
from os import listdir
from random import shuffle
from math import floor
import h5py
import numpy as np 
class Bachitazion(object):
    def __init__(self, sizeOfBatch=128,pathT="/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/AllPatchesWithMicrobleedsTrain/"):
        self.files =shuffle(listdir(pathT))
        self.batchSize = sizeOfBatch
        self.listTrain = listdir(pathT+"Training/") 
        self.listEval = listdir(pathT+"Evaluation/")
        self.counterT = -1
        self.counterE = -1
        self.number_batchesT =floor(len(self.listTrain)/self.batchSize) 
        self.number_batchesE =floor(len(self.listEval)/self.batchSize) 
        self.myShape = [16,16,10]
        self.Path = pathT
    def nextBatch_T (self):
        self.counterT=self.counterT+1
        #Take care, if counterT or counterE is equal to number_batches, the length is goign to be different,
        #But we can save both cases with the following:

        currentBatchSize = len(self.listTrain[self.counterT * self.batchSize:(self.counterT*self.batchSize+self.batchSize)])
        X=np.zeros((currentBatchSize,self.myShape[2],self.myShape[0],self.myShape[1],1))
        Y=np.zeros((currentBatchSize,2))
        
        #If you select a boundary to finish the list bigger than the propper list is fine it will work OK 
        for i,element in enumerate(self.listTrain[self.counterT * self.batchSize:(self.counterT*self.batchSize+self.batchSize)]):
            data_path = self.Path+"Training/"+element
            aux= np.array(h5py.File(data_path)['patchFlatten'])
            patch= aux.reshape(self.myShape)
            patch = patch.transpose(2,0,1)
            if "WO" in element:
                auxY = np.array([1,0])
            else:
                auxY = np.array([0,1])
            
            X[i,:,:,:,0]=patch
            Y[i,:]=auxY
            
        if self.counterT == self.number_batchesT:
            self.counterT=-1
        
        
        return X,Y

    def nextBatch_E (self):
        self.counterE=self.counterE+1
        #Take care, if counterT or counterE is equal to number_batches, the length is goign to be different,
        #But we can save both cases with the following:

        currentBatchSize = len(self.listEval[self.counterE * self.batchSize:(self.counterE*self.batchSize+self.batchSize)])
        X=np.zeros((currentBatchSize,self.myShape[2],self.myShape[0],self.myShape[1],1))
        Y=np.zeros((currentBatchSize,2))
        
        #If you select a boundary to finish the list bigger than the propper list is fine it will work OK 
        for i,element in enumerate(self.listEval[self.counterE * self.batchSize:(self.counterE*self.batchSize+self.batchSize)]):
            data_path = self.Path+"Evaluation/"+element
            aux= np.array(h5py.File(data_path)['patchFlatten'])
            patch= aux.reshape(self.myShape)
            patch = patch.transpose(2,0,1)
            
            if "WO" in element:
                auxY = np.array([1,0])
            else:
                auxY = np.array([0,1])
            
            X[i,:,:,:,0]=patch
            Y[i,:]=auxY
            
        if self.counterE == self.number_batchesE:
            self.counterE=-1
        
        
        return X,Y

