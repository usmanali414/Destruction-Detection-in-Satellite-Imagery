"""
Created on Wed Sep  4 14:35:34 2019

@author: hec
"""
import glob
import utility
from patchify import patchify, unpatchify
import pickle 
import os
from keras.models import load_model
import Network
import results
from optparse import OptionParser
import sys
import numpy as np
import cv2

parser = OptionParser()
parser.add_option("--model1_name", dest="model1_name", help="Name of Model1 weights file to calculate accuracy of test data.")
parser.add_option("--model_retrain_name", dest="model_retrain_name", help="Name of Model retrain weights file to calculate accuracy of test data.")
parser.add_option("--features_filename", dest="features_filename", help="Mention if you want to use already saved or new extracted features-pass filename")
parser.add_option("--patch_stride", dest="patch_stride",type="int", help="mention stride of window to extract patches and features.", default=64)
(options, args) = parser.parse_args()

if not options.model1_name:   # if filename is not given
	parser.error('Error: Model1 name must be specified. Pass --model1_name to command line')
if not options.model_retrain_name:   # if filename is not given
	parser.error('Error: Model retrain name must be specified. Pass --model_retrain_name to command line')
if not options.features_filename:
    parser.error('Error: Features filename must be given in command line arguments')

def extract_features_testimages(features_filename,patch_size=224,window_stride=64):
    model_densenet = utility.load_denseNet()

    patchsize=patch_size
    stride=window_stride
    print("Starting to extract features of Test destruction class images")
    #===========
    test_nondestruct1 = []
    testdestruct = []
    percent_threshold = 15.
    train_destructpath = []
    dest_dir = os.path.join("Data","test","destructed","*.jpg")
    train_destructpath.extend(glob.glob(dest_dir))
    
    for i in range(len(train_destructpath)):
        img = cv2.imread(train_destructpath[i])
        maskname = train_destructpath[i].split("/")[-1]
        
        total = patchsize*patchsize*3
        maskpath = os.path.join("Data","masks",maskname)
        
        mask = cv2.imread(maskpath)
        maskpatches = patchify(mask, (patchsize,patchsize,3), step=stride)
        imgpatches = patchify(img, (patchsize,patchsize,3), step=stride)
        imgcounter = 0
        for j in range(imgpatches.shape[0]):
            for k in range(imgpatches.shape[1]):
                n_white_pix = np.sum(maskpatches[j][k][0] == 255)
                pred_perc = n_white_pix/total*100
              
                if( pred_perc > percent_threshold):
                    temp = imgpatches[j][k][0]
                    temp = utility.preprocessing(temp,patchsize)
                    feature = model_densenet.predict(temp,steps=1)
                    testdestruct.append(feature)
                else:
                    temp = imgpatches[j][k][0]
                    temp = utility.preprocessing(temp,patchsize)
                    feature = model_densenet.predict(temp,steps=1)
                    test_nondestruct1.append(feature)
            
    test_nondestruct1 = np.array(test_nondestruct1)
    testdestruct = np.array(testdestruct)
    test_nondestruct1 = np.reshape(test_nondestruct1,(-1, 1024))
    testdestruct = np.reshape(testdestruct,(-1, 1024))
    print(test_nondestruct1.shape)
    
    
    #=============================================
    print("Starting to extract features of Test Non destruction class images")
    train_nondestructpath = []
    test_nondestruct2 = []
    nondest_dir = os.path.join("Data","test","non_destructed","*.jpg")
    train_nondestructpath.extend(glob.glob(nondest_dir))
    for i in range(len(train_nondestructpath)):
        
        img = cv2.imread(train_nondestructpath[i])
        imgpatches = patchify(img, (patchsize,patchsize,3), step=stride)
        for j in range(imgpatches.shape[0]):
            for k in range(imgpatches.shape[1]):
                
                temp = imgpatches[j][k][0]
                temp = utility.preprocessing(temp,patchsize)
                feature = model_densenet.predict(temp,steps=1)
                test_nondestruct2.append(feature)
    test_nondestruct2 = np.array(test_nondestruct2)
    test_nondestruct2 = np.reshape(test_nondestruct2,(-1, 1024))
    
    #=============================================
    test_nondestruct = np.concatenate((test_nondestruct1,test_nondestruct2),axis=0)
    
    print("Shape of destructed features: ",testdestruct.shape)
    print("Shape of non-destructed features: ",test_nondestruct.shape)
    
    print("saving Test Features")
    
    picklepath= os.path.join("features",features_filename)
    with open(picklepath, 'wb') as handle:
        pickle.dump(test_nondestruct, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(testdestruct, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return testdestruct,test_nondestruct  

#==========================================================================
picklepath= os.path.join("features",options.features_filename)

if(os.path.exists(picklepath)):
    print("Successfuly found Features...")
    print("Loading Features")
    with open(picklepath, 'rb') as handle:
        test_nondestruct = pickle.load(handle)
        testdestruct = pickle.load(handle)
        
        
    print("Features Loaded successfuly..!!!")

else:
    print("Features path not found..")
    testdestruct,test_nondestruct = extract_features_testimages(options.features_filename,patch_size=224,window_stride=options.patch_stride)
#==========================================================================

#Changing features shape to input of model
test_nondestruct = np.reshape(test_nondestruct,(-1,1,1024))
testdestruct = np.reshape(testdestruct,(-1,1,1024))

#Getting images from test directory for testing of pixels.
path1 = "Data/test"
dest_dir = os.path.join(path1,"destructed","*.jpg")
nondest_dir = os.path.join(path1,"non_destructed","*.jpg")

imageslist = []
imageslist.extend(glob.glob(dest_dir))
imageslist.extend(glob.glob(nondest_dir))

"""
#model 1 loading
""" 
model1_path = os.path.join("models",options.model1_name)
if(os.path.exists(model1_path)):
    
    print("loading Trained model..")
    model1 = Network.model_attention()
    model1.load_weights(model1_path)
    
    print("Model 1 loaded Succesfully!")
else:
    sys.exit("Error: Model1 path not found")
"""
#model Retrain loading
""" 
model_retrain_path = os.path.join("models",options.model_retrain_name)
if(os.path.exists(model_retrain_path)):
    
    print("loading Retrained model..")
    model_retrain = Network.model_attention()
    model_retrain.load_weights(model_retrain_path)
    
    print("Model Retrain loaded Succesfully!")
else:
    sys.exit("Error: Model retrain path not found")
    
    print()
results.patch_accuracy_prf(model1, model_retrain, test_nondestruct, testdestruct)
results.pixel_accuracy_prf(options.model1_name,options.model_retrain_name,imageslist)
































