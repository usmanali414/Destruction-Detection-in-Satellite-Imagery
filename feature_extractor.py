# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:05:32 2019

@author: M.Usman Ali
"""

import cv2
import glob
import utility

import pandas as pd
import numpy as np
import imageio
import os
import pickle


patchsize = 224
stride = 224
#Loading DenseNet model
model_densenet = utility.load_denseNet()
model_densenet.summary()
#Reading images from train folder
train_destructed_images= []
train_destructed_folder = os.path.join("Data","train","destructed","*.jpg")
train_destructed_images.extend(glob.glob(train_destructed_folder))


train_destructed_features = []
print("Started extracted patches of Destructed Class images.")
total_images = len(train_destructed_images)
for i in range(total_images):
    
    img = cv2.imread(train_destructed_images[i])
    imgpatches = utility.get_patches(img,patchsize,stride)
    tempimgfeatures = []
    
    for j in range(imgpatches.shape[0]):
        for k in range(imgpatches.shape[1]):
            
            temp = imgpatches[j][k][0]
            temp = utility.preprocessing(temp,patchsize)
            temp = model_densenet.predict(temp,steps=1)
            tempimgfeatures.append(temp)
    
    print("Extracting features of image "+str(i)+"/"+str(total_images))
    train_destructed_features.append(tempimgfeatures)

print("Features Extracted succesfully!")
train_destructed_features = np.array(train_destructed_features)
print(train_destructed_features.shape)
train_destructed_features = np.reshape(train_destructed_features,(-1,9,1024))





train_non_destructed_images=[]
train_non_destructed_folder = os.path.join("Data","train","non_destructed","*.jpg")
train_non_destructed_images.extend(glob.glob(train_non_destructed_folder))
train_non_destructed_features = []
print("Started extracted patches of Non Destructed Class images.")
total_images = len(train_non_destructed_images)
for i in range(total_images):
    
    img = cv2.imread(train_non_destructed_images[i])
    imgpatches = utility.get_patches(img,patchsize,stride)
    tempimgfeatures = []
    
    for j in range(imgpatches.shape[0]):
        for k in range(imgpatches.shape[1]):
            
            temp = imgpatches[j][k][0]
            temp = utility.preprocessing(temp,patchsize)
            temp = model_densenet.predict(temp,steps=1)
            tempimgfeatures.append(temp)
    
    print("Extracting features of image "+str(i)+"/"+str(total_images))
    train_non_destructed_features.append(tempimgfeatures)

print("Features Extracted succesfully!")
train_non_destructed_features = np.array(train_non_destructed_features)
print(train_non_destructed_features.shape)
train_non_destructed_features = np.reshape(train_non_destructed_features,(-1,9,1024))


#print("Now saving features...")
#
#picklepath= os.path.join("features","trainfeatures.pickle")
#with open(picklepath, 'wb') as handle:
#    pickle.dump(train_destructed_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
#    pickle.dump(train_non_destructed_features, handle, protocol=pickle.HIGHEST_PROTOCOL)