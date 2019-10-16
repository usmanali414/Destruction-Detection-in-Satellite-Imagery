#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:25:17 2019

@author: hec
"""
import numpy as np
import scipy
import os
import pydensecrf.densecrf as dcrf
import cv2
import imageio
import cv2
import Network
from skimage.io import imread, imsave
import scipy
import matplotlib
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian,unary_from_softmax



def CRF(masklist,imageslist):
    
    for i in range(len(imageslist)):
        imagename = imageslist[i].split("/")[-1]
        imagename = imagename.split(".")[0]
        img = cv2.imread(imageslist[i])
        b = masklist[i]
        imsave("temp.png",b)
        anno_rgb = imageio.imread("temp.png").astype(np.uint32)
        min_val = np.min(anno_rgb.ravel())
        max_val = np.max(anno_rgb.ravel())
        if((max_val - min_val) == 0):
            out = (anno_rgb.astype('float') - min_val) / 1
        else:
            out = (anno_rgb.astype('float') - min_val) / (max_val - min_val)
        labels = np.zeros((2, img.shape[0], img.shape[1]))
        labels[1, :, :] = out
        labels[0, :, :] = 1 - out
        
        colors = [0, 255]
        colorize = np.empty((len(colors), 1), np.uint8)
        colorize[:,0] = colors
    
        n_labels = 2
        
        crf = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)
        
        U = unary_from_softmax(labels)
        crf.setUnaryEnergy(U)
        feats = create_pairwise_bilateral(sdims=(100, 100), schan=(13, 13, 13),
        								img=img, chdim=2)
        crf.addPairwiseEnergy(feats, compat=10,
        					  kernel=dcrf.FULL_KERNEL,
        					  normalization=dcrf.NORMALIZE_SYMMETRIC)  
        
        Q = crf.inference(20)
        
        MAP = np.argmax(Q, axis=0)
        MAP = colorize[MAP]
        crfmaskdir = os.path.join("predicted_masks","crf_masks")
        if not (os.path.exists(crfmaskdir)):
            os.mkdir(crfmaskdir)
        cv2.imwrite(crfmaskdir+'/'+imagename+".png", MAP.reshape(anno_rgb.shape))
    return crfmaskdir
                    