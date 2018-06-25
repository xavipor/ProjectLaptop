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
    def __init__(self, sizeOfBatch=64,pathT="/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/AllPatchesWithMicrobleedsTrain/",percTrainEval=0.7):
        self.files =shuffle(listdir(pathT))
        self.batchSize = sizeOfBatch
        self.listTrain = listdir(pathT+"Training/") 
        self.listEval = listdir(pathT+"Evaluation/")
        self.counterT = 0
        self.number_batchesT =floor(len(self.listTrain)/self.batchSize) 
        self.number_batchesE =floor(len(self.listEval)/self.batchSize) 
        self.myShape = [16,16,10]
    def nextBatch (self):
        X=np.zeros((self.batchSize,self.myShape[2],1,self.myShape[0],self.myShape[1]))
        ###############
        ### OJO AQUI, TENGO QUE AÑADIR QUE SI ES EL ULTIMO COUNTER, NO VALE ESTO, TIENE QUE SER DE EL VALOR QUE SEA AL ULTIMO DE LO QUE QUEDE POR PILLAR
        ### SABES??
        
        
        
        for i,element in enumerate(self.listTrain[self.counterT * self.batchSize:(self.counterT*self.batchSize+self.batchSize)]):
            data_path = self.listTrain+element
            aux= np.array(h5py.File(data_path)['patchFlatten'])
            patch= aux.reshape(self.myShape)
            X[i,:,1,:,:]=patch
            if 
            
        
        
        
        
        return X,Y
    
        