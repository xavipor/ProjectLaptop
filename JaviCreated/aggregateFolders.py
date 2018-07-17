#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 16:30:18 2018

@author: javier
"""
from os import listdir
from os.path import isfile,join 
from os import walk
import os 
import shutil
from random import shuffle

def aggregateFolders(path = "/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS",type="train",perc=0.8):
    """
    Just to create a folder with all the images for training/test and test set. 
    
    """
    resultPathTrain = path+"/AllPatchesWithMicrobleedsTrain/"
    normalOnes = path +"/PatchesWithMicrobleeds/"
    augmentedOnes = path + "/PatchedMicrobleedAugmented/"
    pathWO = path + "/PatchesNoMicrobleeds/"
    
    pathTraining = resultPathTrain + "Training/"
    pathEvaluation = resultPathTrain + "Evaluation/"
    
    counter = 0
    counterWO = 0
    
    if type =="train":
        #onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        folders = [int(x) for x in listdir(normalOnes)]
        folders.sort()
        foldersTrain = folders[:19]
        
        
        if not os.path.exists(resultPathTrain):
            os.makedirs(resultPathTrain)
        
        for folder in foldersTrain:
            auxPathNormal = normalOnes + str(folder) + "/"
            for element in listdir(auxPathNormal):
                shutil.copy(auxPathNormal+element,resultPathTrain + "MB_"+str(counter)+".mat" )
                counter +=1
        for folder in foldersTrain:
            auxPathAugmented = augmentedOnes + str(folder) + "/"
            for element in listdir(auxPathAugmented):
                shutil.copy(auxPathAugmented+element,resultPathTrain + "MB_"+str(counter)+".mat" )
                counter +=1    
        
        ##################################################################################################################
        ##################################################################################################################
        #################  So here we have all the microbleeds in the folder, the next step is to  separate between  #####
        #################                    training and evaluation set, let say 80 -20                             #####
        ##################################################################################################################
        ##################################################################################################################
        
        
        patchesWithMicrobleed = listdir(resultPathTrain)
        shuffle(patchesWithMicrobleed)
        if not os.path.exists(pathTraining):
            os.makedirs(pathTraining) 
            
        if not os.path.exists(pathEvaluation):
            os.makedirs(pathEvaluation)  
        
        #Move to training
        for element in patchesWithMicrobleed[:int(perc*len(patchesWithMicrobleed))]:
            shutil.move(resultPathTrain+element,pathTraining+element)
            
        #Move to evaluation
        for element in patchesWithMicrobleed[int(perc*len(patchesWithMicrobleed)):]:
            shutil.move(resultPathTrain+element,pathEvaluation+element)
        
        
        ##################################################################################################################
        ##################################################################################################################
        #################  Now that the patches with Microbleeds are allocated, it is time of the patches with no    #####
        #################                           Microbleeds                                                      #####
        ##################################################################################################################
        ##################################################################################################################        
        
        
        for f in [pathTraining,pathEvaluation]:
            elementsWO = listdir(pathWO)
            shuffle(elementsWO)
            start=0
            myend=0
            myend= 1.5*len(listdir(f))   ############################################OJOOOOOOOOOOOOO MODIFICACIÓN AQUI, VOY A PONER MAS NO MICRBOLEEDS *1.5 en el len de esta linea
            #We are goign to see the number of microbleeds patches that we already have, and we are going to seek 
            #for the same number in the WOmicrobleeds folder. We are going to shuffle it before move the files. 
            #see that npw I am moving not copying the images. 
            for element in elementsWO[start:myend]:
                shutil.move(pathWO+element,f + "WO_"+str(element))
                counterWO +=1           
                
            
        
        ###
        ##  FALTA IMPLEMENTAR LA MIERDA DE TEST, ES MAS O MENOS LO MISMO, MINIMAS MODIFICACIONES
        ##
        ##
        
        
        return folders
        
    else:
        pass
        
myList = []    
myList=aggregateFolders(path = "/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS",type="train",perc=0.8)
