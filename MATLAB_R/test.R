library(ANTsRCore)
library(ANTsR)
setwd("/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/gt")
pathToSave="/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/gtResampled/"#Change for GroundTruth or not
pathToRead="/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/gt"
myFiles<-list.files(pathToRead)#Change for GroundTruth or not


for ( i in 1:length(myFiles)){
  img <- antsImageRead(myFiles[i])
  myDimensions <- antsGetSpacing(img)
  myVector <-c()
  for (j in 1:length(myDimensions)){
    myVector[j]=myDimensions[j]
    if (j==3){
      myVector[j]=1
      print(myVector)
    }
  }
  if (i<10){
    antsImageWrite(resampleImage(img,myVector,0,0),paste(pathToSave,"0",i,".nii.gz",sep=""))
    print("Size")
    print(antsImageHeaderInfo(paste(pathToSave,"0",i,".nii.gz",sep=""))$dimensions)
  }else{
    antsImageWrite(resampleImage(img,myVector,0,0),paste(pathToSave,i,".nii.gz",sep=""))
    print("Size")
    print(antsImageHeaderInfo(paste(pathToSave,i,".nii.gz",sep=""))$dimensions)
  }
  
  
}


