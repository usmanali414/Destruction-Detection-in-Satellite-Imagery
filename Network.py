#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:53:08 2019

@author: hec
"""
import keras
from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Input, Dense, Multiply,Dropout,Lambda,Reshape,LeakyReLU,Activation,BatchNormalization


#Custom loss function of Attention network
def custom_loss(layer):
    
    layer1 = K.squeeze(layer,axis=2)
    sum1 = K.sum(layer1,axis=-1)
    sum2 = K.reshape(sum1,(-1,1))
    layer2 = Lambda(lambda x: x[0]/x[1])([layer1,sum2])
   
    def loss(y_true,y_pred):
        return K.binary_crossentropy(y_true,y_pred)+(0.3* K.sum(layer2,axis=-1))
   
    # Return a function
    return loss

def model_attention():
    
    model_in =Input(shape=(None,1024))
    
    x = Dense(512, kernel_initializer='glorot_uniform', bias_initializer='zeros')(model_in)
    x =Activation('sigmoid')(x)
    x = Dropout(0.8)(x)
    x = Dense(1, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)

    att_weights = Activation('sigmoid')(x)
    att_mull = Multiply()([model_in,att_weights])

    mean = Lambda(lambda x:K.mean(x,axis=1))(att_mull)
    
    
    y = Dense(512, kernel_initializer='glorot_uniform', bias_initializer='zeros')(mean)
    y = Activation('sigmoid')(y)
    y = Dropout(0.7)(y)
    y = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', bias_initializer='zeros')(y)
    
    model = Model(model_in, y)
    
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    
    for i, layer in enumerate(model.layers):
        layer.name = 'layer_' + str(i)
        
        
    model.compile(optimizer=adam,loss = custom_loss(att_weights),metrics=['accuracy'])
    return model