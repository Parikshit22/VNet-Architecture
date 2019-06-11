# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 19:30:26 2019

@author: MUJ
"""

import keras


def downblock(x,filters,kernel_size=(3,3,3),padding="same",strides = 1):
    c = keras.layers.Conv3D(filters,kernel_size,padding = padding,strides = strides,activation = 'relu')(x)
    c = keras.layers.Conv3D(filters,kernel_size,padding = padding,strides = strides,activation = 'relu')(c)
    p = keras.layers.Conv3D(filters,kernel_size = (2,2,2),strides = 2)(c)
    return c,p

def upblock(x,skip,filters,kernel_size = (3,3,3),padding = "same",strides = 1):
    us = keras.layers.UpSampling3D((2,2,2))(x)
    concat = keras.layers.Concatenate()([us,skip])
    c = keras.layers.Conv3D(filters,kernel_size,padding = padding,strides = strides,activation = 'relu')(concat)
    c = keras.layers.Conv3D(filters,kernel_size,padding = padding,strides = strides,activation = 'relu')(c)
    return c

def bottleneck(x,filters,kernel_size = (3,3,3),padding = "same", strides = 1):
    c = keras.layers.Conv3D(filters, kernel_size,padding=padding,strides = strides, activation = "relu")(x)
    c = keras.layers.Conv3D(filters, kernel_size,padding=padding,strides = strides, activation = "relu")(c)
    return c

def VNet():
    f = [16,32,64,128,256]
    input = keras.layers.Input((128,128,128,3))
    
    c1,p1 = downblock(input,f[0])
    c2,p2 = downblock(p1,f[1])
    c3,p3 = downblock(p2,f[2])
    c4,p4 = downblock(p3,f[3])
    
    bn = bottleneck(p4,f[4])
    
    u1 = upblock(bn,c4,f[3])
    u2 = upblock(u1,c3,f[2])
    u3 = upblock(u2,c2,f[1])
    u4 = upblock(u3,c1,f[0])
    
    outputs = keras.layers.Conv3D(1,(1,1,1),padding= "same",activation = "sigmoid")(u4)
    model = keras.models.Model(input,outputs)
    return model

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

model = VNet()
model.compile(optimizer = "adam", loss=dice_coef_loss, metrics=[dice_coef])
model.summary()
    