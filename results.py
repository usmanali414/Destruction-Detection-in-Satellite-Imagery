#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 10:12:28 2019

@author: hec
"""
import numpy as np
from patchify import patchify, unpatchify
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
import utility
import os
import glob
import cv2

def patch_accuracy_prf(model,modelretrain,test_non_destruct,test_destruct):
    print("=============================")
    print("Patch Level Results")
    print("=============================")
    nondest_acc1,dest_acc1,over_acc1,Precision1,Recall1,fmeasure1 = patch_model_prediction(model,test_non_destruct,test_destruct)
    nondest_acc2,dest_acc2,over_acc2,Precision2,Recall2,fmeasure2 = patch_model_prediction(modelretrain,test_non_destruct,test_destruct)
    over_acc3,Precision3,Recall3,fmeasure3 = CRF_patch_accuracy()
    print("                           Model1 retrain_model CRF")
    #print("Non-Destruction class Accuracy: ",nondest_acc1,nondest_acc2)
    #print("Destruction class Accuracy:     ",dest_acc1,dest_acc2)
    print("Overall Accuracy:               ",over_acc1,over_acc2,over_acc3)
    print("Precision:                      ",Precision1,Precision2,Precision3)
    print("Recall:                         ",Recall1,Recall2,Recall3)
    print("F1-Score:                       ",fmeasure1,fmeasure2,fmeasure3)
    
    #print("precision: ",Precision)
    #print("Recall: ",Recall)
    #print("F-measure: ",fmeasure)
    #print("=============================")
    
def patch_model_prediction(model,test_nondestruct,testdestruct):
    pred = model.predict(test_nondestruct)
    pred= np.reshape(pred,(len(pred)))
   
    predicted = np.where(pred>0.5,1,0)
    label1 = np.array([0]*len(test_nondestruct))
    nondest_acc = round(accuracy_score(label1, predicted),2)
    #print('Non-Destruction class Accuracy:', )
    
    pred = model.predict(testdestruct)
    pred= np.reshape(pred,(len(pred)))
    
    predicted = np.where(pred>0.5,1,0)
    label2 = np.array([1]*len(testdestruct))
    dest_acc = round(accuracy_score(label2, predicted),2)
    #print('Destruction class Accuracy:', )
    
    
    
    testdata = np.concatenate((test_nondestruct,testdestruct),axis=0)
    testlabel = np.concatenate((label1,label2),axis=0)
    pred = model.predict(testdata)
    
    pred= np.reshape(pred,(len(pred)))
   
    predicted = np.where(pred>0.5,1,0)
    over_acc = round(accuracy_score(testlabel, predicted),2)
    #print('overall Accuracy Score :', )
    results = confusion_matrix(testlabel, predicted)
    TN = results[0][0]
    FP = results[0][1]
    FN = results[1][0]
    TP = results[1][1]
    #print(TN,FP,FN,TP)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    fmeasure = 2*((Recall*Precision)/(Recall+Precision))
    return nondest_acc,dest_acc,over_acc,round(Precision,2),round(Recall,2),round(fmeasure,2)

def predictmask(path_to_test,denseModel,model,patch_size=224,window_stride=64):
    
    patchsize=patch_size
    stride=window_stride
    path = path_to_test
    
    dest_dir = os.path.join(path,"destructed","*.jpg")
    nondest_dir = os.path.join(path,"non_destructed","*.jpg")
    
    imageslist = []
    imageslist.extend(glob.glob(dest_dir))
    imageslist.extend(glob.glob(nondest_dir))
    
    predictedmasklist = []
    # for modelname in models:
    #   model.load_weights(modelname)
    for k in range(len(imageslist)):
        
        imagepath = imageslist[k]
        name = imageslist[k].split("/")[-1]
        image1 = cv2.imread(imagepath)
        imgpatches = patchify(image1, (patchsize,patchsize,3), step=stride)
        mask1 = np.zeros((image1.shape[0],image1.shape[1]))
        mask2 = np.zeros((image1.shape[0],image1.shape[1]))

        patches1 = utility.convert_patches_to_list(imgpatches)
        patches1 = utility.preprocessing_patcheslist(patches1)
        features = denseModel.predict(patches1)
        features = np.reshape(features,(-1,1,1024))
        scoreslist = model.predict(features)
        scoreslist = np.reshape(scoreslist,(len(scoreslist)))

        for i in range(len(scoreslist)):
            
            col = i // int(imgpatches.shape[1])
            row = i % int(imgpatches.shape[1])

            x1 = (row*stride) 
            y1 = (col*stride)
            x2 = x1+patchsize
            y2 = y1+patchsize

            mask1[y1:y2,x1:x2] = mask1[y1:y2,x1:x2] + scoreslist[i]
            mask2[y1:y2,x1:x2] = mask2[y1:y2,x1:x2] + 1

        mask1[np.isnan(mask1)] = 0
        mask2[mask2 == 0] = 1
        b = mask1/mask2
        predictedmasklist.append(b)
        print("Mask generated of image "+str(k)+"/"+str(len(imageslist)))
    return predictedmasklist,imageslist    

def CRF_patch_accuracy():
    
    patchsize=224
    stride=64
    percent_threshold = 15.
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    
    crfmaskpath = os.path.join("predicted_masks/crf_masks","*.png")
    crfmasklist = []
    crfmasklist.extend(glob.glob(crfmaskpath))
    for i in range(len(crfmasklist)):
    
        maskname = crfmasklist[i].split("/")[-1]
        name = maskname.split('.')[0]
        total = patchsize*patchsize*3
        
  
        groundmaskpath = os.path.join("Data/masks",name+".jpg")
        
        predictedmask = cv2.imread(crfmasklist[i])
        
  
        predictedmaskpatches = patchify(predictedmask, (patchsize,patchsize,3), step=stride)
  
  
  
        if 'pre' in maskname:
            
            for j in range(predictedmaskpatches.shape[0]):
                
                for k in range(predictedmaskpatches.shape[1]):

                    pred_white_pix = np.sum(predictedmaskpatches[j][k][0] == 255)
                    pred_perc = pred_white_pix/total*100

                    if( pred_perc >= percent_threshold ):
                        FP = FP + 1
                    else:
                        TN = TN + 1
     
        else:
            groundmask = cv2.imread(groundmaskpath)
            groundmaskpatches = patchify(groundmask, (patchsize,patchsize,3), step=stride)
          
            for j in range(predictedmaskpatches.shape[0]):
                for k in range(predictedmaskpatches.shape[1]):
        
                    pred_white_pix = np.sum(predictedmaskpatches[j][k][0] == 255)
                    pred_perc = pred_white_pix/total*100
        
                    ground_white_pix = np.sum(groundmaskpatches[j][k][0] == 255)
                    ground_perc = ground_white_pix/total*100
        
                    if( pred_perc >= percent_threshold and ground_perc >= percent_threshold ):
                        TP = TP + 1
                    elif(pred_perc >= percent_threshold and ground_perc < percent_threshold):
                        FP = FP+1
                    elif(pred_perc < percent_threshold and ground_perc >= percent_threshold):
                        FN = FN+1
                    elif(pred_perc < percent_threshold and ground_perc < percent_threshold):
                        TN = TN+1
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*((precision*recall)/(precision+recall))
    return round(accuracy,2),round(precision,2),round(recall,2),round(f1,2)                      



def pixel_accuracy_prf(model1,modelretrain,imageslist):
    
    model1name = model1.split('.')[0]
    model1_maskpath = os.path.join("predicted_masks",model1name)
    iou1,over_acc1,Precision1,Recall1,fmeasure1 = pixelresult(imageslist,model1_maskpath)
    
    modelretrainname = modelretrain.split('.')[0]
    modelretrain_maskpath = os.path.join("predicted_masks",modelretrainname)
    iou2,over_acc2,Precision2,Recall2,fmeasure2 = pixelresult(imageslist,modelretrain_maskpath)
    
    crf_maskpath = os.path.join("predicted_masks","crf_masks")
    iou3,over_acc3,Precision3,Recall3,fmeasure3 = pixelresult(imageslist,crf_maskpath)
    
    
    
    print("=============================")
    print("Pixel level results")
    print("=============================")
    print("                           Model1 retrain_model CRF")
    
    print("IOU Score:                      ",iou1,iou2,iou3)
    print("Overall Accuracy:               ",over_acc1,over_acc2,over_acc3)
    print("Precision:                      ",Precision1,Precision2,Precision3)
    print("Recall:                         ",Recall1,Recall2,Recall3)
    print("F1-Score:                       ",fmeasure1,fmeasure2,fmeasure3)
    print("=============================")
    
    
    
    
    
def pixelresult(imageslist,modelmaskpath):
    TP1 = []
    FP1 = []
    FN1 = []
    TN1 = []
   

    for k in range(len(imageslist)):
        imagepath = imageslist[k]
        name = imageslist[k].split("/")[-1]
        name = name.split(".")[0]
        groundmaskpath = 'Data/masks/'+name+".jpg"
        
        image1 = cv2.imread(imagepath)
        
        predmaskpath = os.path.join(modelmaskpath,name+".png")
        pred_mask = cv2.imread(predmaskpath)
    
        
        if "post" in imagepath:
            
            ground_mask = cv2.imread(groundmaskpath)
            target = cv2.cvtColor(ground_mask, cv2.COLOR_BGR2GRAY)
            target = np.where(target > 254,1,0)
            prediction = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)
            intersection = np.logical_and(target, prediction)
            union = np.logical_or(target, prediction)
            #iou_score = np.sum(intersection) / np.sum(union) 
        
            zeros1 = np.zeros((target.shape[0],target.shape[1]))
            prediction_p = np.logical_or(prediction,zeros1)
            xor1 = np.logical_xor(prediction_p,intersection)
            FP = np.sum(np.logical_and(prediction_p,xor1))
            TP = np.sum(intersection)
            target_p = np.logical_or(target,zeros1)
            xor1 = np.logical_xor(target_p,intersection)
            FN = np.sum(np.logical_and(target_p,xor1))
            TN = np.sum(np.logical_not(union))
    
        else:
            TP = 0
            FN = 0
            prediction = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)
            zeros1 = np.zeros((image1.shape[0],image1.shape[1]))
            #prediction_p = np.logical_or(prediction,zeros1)
            #xor1 = np.logical_xor(prediction_p,intersection)
            #FP = np.sum(np.logical_and(prediction_p,xor1))
            FP = np.sum(np.logical_or(prediction,zeros1))
            union1 = np.logical_or(zeros1, prediction)
            TN = np.sum(np.logical_not(union1))
    
        TP1.append(TP)
        FP1.append(FP)
        FN1.append(FN)
        TN1.append(TN)
  
    pred_iou1 = sum(TP1)/(sum(FP1)+sum(TP1)+sum(FN1))
    accuracy = (sum(TP1)+sum(TN1)) /(sum(FP1)+sum(TP1)+sum(FN1)+sum(TN1))
    precision2 = sum(TP1) / (sum(TP1)+sum(FP1))
    recall2 = sum(TP1) / (sum(TP1)+sum(FN1))
    f1score = 2*((precision2*recall2)/(precision2+recall2))
    return round(pred_iou1,2),round(accuracy,2),round(precision2,2),round(recall2,2),round(f1score,2)



















    