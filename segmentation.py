#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:05:48 2019

@author: hec
"""
from optparse import OptionParser
import numpy as np
import cv2
import results
import Network
from imageio import imread, imsave
import crf
import utility
import os


parser = OptionParser()
parser.add_option("--model_name", dest="model_name", help="Path to Model to calculate accuracy of test data.")
parser.add_option("--test_path", dest="test_path", help="Path to test data.")
parser.add_option("--apply_CRF", dest="apply_CRF", help="Mention if you want to apply CRF 'yes' or 'no' ")
parser.add_option("--patch_stride", dest="patch_stride",type="int", help="mention stride of window to extract patches and features.", default=64)
(options, args) = parser.parse_args()


model_densenet = utility.load_denseNet()

modelpath = os.path.join("models",options.model_name)

if(os.path.exists(modelpath)):
    
    print("loading Trained model..")
    model = Network.model_attention()
    model.load_weights(modelpath)
    path_to_test = options.test_path
    masks,nameslist = results.predictmask(path_to_test=path_to_test,denseModel=model_densenet,model=model,patch_size=224,window_stride=64)
    modelnamedir = options.model_name.split('.')[0]
    maskdir = os.path.join("predicted_masks",modelnamedir)
    if not (os.path.exists(maskdir)):
        print("Making new directory for masks")
        os.mkdir(maskdir)
    for i in range(len(masks)):
        imgname = nameslist[i].split("/")[-1]
        maskname = imgname.split(".")[0]
        mask = np.where(masks[i]>0.6,255,0)
        cv2.imwrite(os.path.join(maskdir,maskname+".png"),mask)
        
    if (options.apply_CRF == "yes"):
        
        print("Applying Crf..." )
        crfdir = crf.CRF(masks,nameslist)
        print("masks generated.Check below directories")
        print(crfdir)
        print(maskdir)
    else:
        print("masks generated.Check below directories")
        print(maskdir)
else:
    print("Error: Model path not found")