#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:55:20 2019

@author: hec
"""

import utility
from sklearn.utils import shuffle
import tensorflow as tf
from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Input, Dense, Multiply,Dropout,Lambda,Reshape,LeakyReLU,Activation,BatchNormalization
import Network
import numpy as np

from optparse import OptionParser

import numpy as np
import cv2

parser = OptionParser()
parser.add_option("--trainfeatures_filename", dest="trainfeatures_filename", help="Path to train features file.")
parser.add_option("--epochs", dest="epochs" , type="int" , help="epochs to train model")
(options, args) = parser.parse_args()

if not options.trainfeatures_filename:   # if filename is not given
	parser.error('Error: features file name must be specified. Pass --trainfeatures_filename to command line')



def train_model(model,traindest,train_nondest,valdest,val_nondest,batch=32,param_epoch=500):#
    batchsize = batch
    halfbatch = int(batchsize/2)
    epochs = param_epoch
    iterations = 100
    alter = 0
    epochloss = []
    epochacc = []
    val_losslist = []
    val_acclist = []
    val_label1 = np.array([0]*len(val_nondest))
    val_label2 = np.array([1]*len(valdest))
    
    val_data = np.concatenate((val_nondest,valdest),axis=0)
    val_label = np.concatenate((val_label1,val_label2),axis=0)
    
    for epoch in range(epochs):
        
        print("Training Epoch: ",epoch)
        iterloss = []
        iteracc = []
#        traindest = shuffle(traindest)
#        train_nondest = shuffle(train_nondest)
        start1,start2 = 0,0
        end1,end2 = halfbatch,halfbatch
        total2 = len(traindest)
        for iter in range(iterations):
    
            pre1 = train_nondest[start1:end1]
            if((total2 - end2) < 8):
                start2 = 0
                end2=halfbatch
            post1 = traindest[start2:end2]
            data = np.concatenate((pre1,post1),axis=0)
        
            label1 = [0]*halfbatch
            label2 = [1]*halfbatch
            labels = np.concatenate((label1,label2),axis=0)
        
            loss,acc = model.train_on_batch(data,labels)
            iterloss.append(loss)
        
            iteracc.append(acc)
            start1 = end1
            end1 = start1+halfbatch
            start2 = end2
            end2 = start2+halfbatch
        l1 = sum(iterloss)/len(iterloss)
        a1 = sum(iteracc)/len(iteracc)
        epochloss.append(l1)
        epochacc.append(a1)
        val_loss,val_acc = model.evaluate(val_data,val_label)
        val_losslist.append(val_loss)
        val_acclist.append(val_acc)
        print("Training loss and Accuracy : "+" Loss: "+str(l1)+" Acc: "+str(a1))
        print("Validation loss and Accuracy : "+" Loss: "+str(val_loss)+" Acc: "+str(val_acc))
        
        if( epoch % 100 == 0):
            model.save('models/Model1_AttentionNetwork_'+str(epoch)+'.h5')

    return model,val_losslist,val_acclist



#loading Features
print("Loading Features...")
train_destruct_features,train_non_destruct_features = utility.loadtrainfeatures(options.trainfeatures_filename)

train_destruct_features = np.array(train_destruct_features)
train_destruct_features = np.reshape(train_destruct_features,(-1,9,1024))


train_non_destruct_features = np.array(train_non_destruct_features)
train_non_destruct_features = np.reshape(train_non_destruct_features,(-1,9,1024))

print("Splitting data into training and testing....")
#splitting into training and testing data
train_destruct_features = shuffle(train_destruct_features,random_state=2)
train_non_destruct_features = shuffle(train_non_destruct_features,random_state=2)
print(train_destruct_features.shape)
print(train_non_destruct_features.shape)
val_destruct = train_destruct_features[1700:]
val_non_destruct = train_non_destruct_features[2000:]

train_destruct_features = train_destruct_features[:1700]
train_non_destruct_features = train_non_destruct_features[:2000]
    
#load Attentionmodel      
print("loading attention model....")      
model = Network.model_attention()

#training model
print("Starting Training..!!")
model,val_loss,val_acc = train_model(model,train_destruct_features,train_non_destruct_features,val_destruct,val_non_destruct,batch=32,param_epoch=options.epochs)#,val_destruct,val_non_destruct,batch=32,param_epoch=options.epochs)
print("Training Done!!!")

print("saving model..!!")
model.save("models/Model1_AttentionNetwork_"+str(options.epochs)+".h5")
        
        
        
        
#pred1 = model.predict(testnormalfeatures2)
#  pred2 = model.predict(testdestructfeatures)
#  pred1= np.reshape(pred1,(len(pred1)))
#  pred2= np.reshape(pred2,(len(pred2)))
#  
#  predicted1 = np.where(pred1>0.5,1,0)
#  predicted2 = np.where(pred2>0.5,1,0)
#  
#  test_acc1 = accuracy_score(test_label1, predicted1)
#  test_acc2 = accuracy_score(test_label2, predicted2)
#  
#  print("Testing loss and Accuracy : "+" Acc1: "+str(test_acc1)+" Acc2: "+str(test_acc2))
##  
            
            
#             pre_loss,pre_acc = model.evaluate(val_pre,val_label1)
#        post_loss,post_acc = model.evaluate(val_post,val_label2)
#          #val_loss,val_acc = model.evaluate(val_data,val_label)
#        print('Validation Scores:  Loss '+str(val_loss)+" Acc: "+str(val_acc))
#        print('Validation individual accuracy:  '+str(pre_acc)+" Acc: "+str(post_acc))