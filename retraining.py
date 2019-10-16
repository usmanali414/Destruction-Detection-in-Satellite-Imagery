#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 18:53:42 2019

@author: hec
"""
import utility
import numpy as np
import Network
from sklearn.utils import shuffle
import keras
import os

from optparse import OptionParser


parser = OptionParser()
parser.add_option("--model_name", dest="model_name", help="Path to model to which we are going to retrain")
parser.add_option("--trainfeatures_filename", dest="trainfeatures_filename", help="Path to train features file.")
parser.add_option("--epochs", dest="epochs" , type="int" , help="epochs to train model",default=300)
(options, args) = parser.parse_args()
#============================
def thresholding_and_loading_features(modelname,trainfeatures):
    
    print("Loading Features...")
    train_destruct_features,train_non_destruct_features = utility.loadtrainfeatures(trainfeatures)
    
    
    train_non_destruct_features = np.reshape(train_non_destruct_features,(-1,1,1024))
    train_destruct_features = np.reshape(train_destruct_features,(-1,1,1024))
    print(train_non_destruct_features.shape)
    print(train_destruct_features.shape)
    
    
    #Loading model 
    
    modelpath = os.path.join("models",modelname)
    print(modelpath)
    if(os.path.exists(modelpath)):
        print("loading Trained model..")
        model = Network.model_attention()
        model.load_weights(modelpath)
        
        neg = model.predict(train_non_destruct_features)
        pos = model.predict(train_destruct_features)
        neg = np.reshape(neg,(len(neg)))
        pos = np.reshape(pos,(len(pos)))
        
        
        neg1 = np.where(neg>0.20 )
        neg2= np.where(pos<0.20)
        pos1 = np.where((pos>0.60) )
        pre = np.array(train_non_destruct_features)
        post =np.array(train_destruct_features)
        trainpre1 = pre[neg1]
        trainpre2 = post[neg2]
        trainpost = post[pos1]
        trainpre = np.concatenate((trainpre1,trainpre2),axis = 0)
        
    else:
        print("Error: Model not found on given path..")
        return 0
    print("Shape of Destruct Patches after thresholding: " ,trainpost.shape)
    print("Shape of Non-Destruct Patches after thresholding: " ,trainpre.shape)
    return model,trainpre,trainpost


def clonemodel_setting_dropout(model):
    
    model.layers[10].rate = 0.8
    model2 = keras.models.clone_model(model)
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    for i, layer in enumerate(model2.layers):
        layer.name = 'layer_' + str(i)
    
    
    model2.compile(optimizer=adam,
                  loss=Network.custom_loss(model2.layers[5].output), # Call the loss function with the selected layer
                  metrics=['accuracy'])

    
    
    return model2

def retrain_model(model,trainpre,trainpost,batch_size=32,param_epoch=100):
    
    batchsize = batch_size
    halfbatch = int(batchsize/2)
    epochs = param_epoch
    iterations = 300
    alter = 0
    epochloss = []
    epochacc = []
   
    for epoch in range(epochs):
        print("Training Epoch: ",epoch)
        iterloss = []
        iteracc = []
#        trainpre = shuffle(trainpre)
#        trainpost = shuffle(trainpost)
        start = 0
        end = halfbatch
        startpost = 0
        endpost = halfbatch
  
        for iter in range(iterations):
    
            pre1 = trainpre[start:end]
            post1 = trainpost[startpost:endpost+8]
            data = np.concatenate((pre1,post1),axis=0)
    
            label1 = [0]*(halfbatch)
            label2 = [1]*(halfbatch+8) 
            labels = np.concatenate((label1,label2),axis=0)
            data,labels = shuffle(data,labels, random_state=2)
            loss,acc = model.train_on_batch(data,labels)
            iterloss.append(loss)
    
            iteracc.append(acc)
            start = end
            end = start+halfbatch
            startpost = endpost+8
            endpost = startpost+halfbatch
        l1 = sum(iterloss)/len(iterloss)
        a1 = sum(iteracc)/len(iteracc)
        epochloss.append(l1)
        epochacc.append(a1)
        
        print("Training loss and Accuracy => Loss: "+str(l1)+" Acc: "+str(a1))
        if(epoch==200):
            model.save("models/Model2_retrain_AttentionNetwork_"+str(epoch)+".h5")
    return model  
    
  
#============================================================================
model,train_nondestruct,train_destruct = thresholding_and_loading_features(options.model_name,options.trainfeatures_filename)
model = clonemodel_setting_dropout(model)
model.summary()
model = retrain_model(model,train_nondestruct,train_destruct,batch_size=32,param_epoch=options.epochs)

model.save("models/Model2_retrain_AttentionNetwork_"+str(options.epochs)+".h5")









