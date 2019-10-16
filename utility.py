# -*- coding: utf-8 -*-

from keras.preprocessing import image
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout
import numpy as np
import os
import pickle
from patchify import patchify, unpatchify



#loading denset net model
def load_denseNet():
    
    densenet = keras.applications.densenet.DenseNet121(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    
    fclast = densenet.get_layer("avg_pool").output
    modeldense = Model(inputs=densenet.input, outputs=fclast)
    return modeldense

#Extract patches through sliding window
def get_patches(img,patchsize=224,stride=224):
    imgpatches = patchify(img, (patchsize,patchsize,3), step=stride)
    return imgpatches

def preprocessing(img,patchsize=224):
    temp = np.array(img).astype('float16')
    temp = temp/255.
    temp = np.reshape(temp,(1,patchsize,patchsize,3))
    return temp

def loadtrainfeatures(filename):
    
    picklepath= os.path.join("features",filename)


    with open(picklepath , 'rb') as handle:
        train_destruct_features = pickle.load(handle)
        train_non_destruct_features = pickle.load(handle)
    return train_destruct_features,train_non_destruct_features

#function to return patches list
def convert_patches_to_list(patches):
  patcheslist = []
  for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
      patcheslist.append(patches[i][j][0])
      
  return patcheslist

#Function to return preprocessed patches list
def preprocessing_patcheslist(patcheslist):

  #Converting into numpy and preprocessing
  patcheslist = np.array(patcheslist)
  patcheslist = patcheslist.astype('float16')
  patcheslist = patcheslist/255.
  
  return patcheslist