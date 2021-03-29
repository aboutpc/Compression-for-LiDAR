#!/usr/bin/env python
# coding: utf-8

from random import randint
import os
import numpy as np
from curvative import *
from generate import *
from Loss import *
import argparse

#from pclpy import pcl
from keras.layers import *
from keras.activations import *
from keras.models import Model,load_model
from keras.optimizers import *
from keras import losses
import keras
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard

config = tf.ConfigProto()
config.gpu_options.visible_device_list="0"
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

parser = argparse.ArgumentParser(description='model')
parser.add_argument('--curvature', type=str,default=0,help='model with curvature or not')
parser.add_argument('--model_path', type=str,default='/model/',help='path of model')
parser.add_argument('--data_path', type=str,default='/data/',help='path of data')
args = parser.parse_args()

with open((os.path.join('train_txt.txt')), 'r') as f:
    data_name = f.read()
traindata_name = data_name.split(',')
traindata_name = traindata_name[0:100]

traindata = np.array([0])
data = np.array([0])
curdata = np.array([0])
cur = np.array([0])

for i in traindata_name:  
    obj=np.loadtxt('/GPUFS/sysu_khuang_2/pointcloud/txt_data/'+i)
    if traindata.ndim < 3: 
        traindata = obj
        traindata,curdata = curvature_del_to_anynum(traindata,thre=1e-1,point_num=50176+1) 
        traindata = np.expand_dims(traindata,axis=0)
    else:
        data = obj
        data,cur = curvature_del_to_anynum(data,thre=1e-2,point_num=50176+1) 
        data = np.expand_dims(data,axis=0)
        if(traindata.shape[1:] == data.shape[1:]):
            traindata = np.append(traindata,data,axis=0)

traindata_time = []
for i in range(5):
    traindata_time.append(traindata[i-5:i])
traindata_time=np.array(traindata_time)
traindata_norm = (traindata_time+101.97399)/(104.31858+101.97399)
train_xx = traindata
train_xx_tensor = np.reshape(train_xx,[1,5,1568,32,3])

if args.curvature:
    x_inp=Input(shape=(5,1568,32,3),name='input_1')
    cur_inp=Input(shape=(5,1568,32,1),name='cur_input_1')
    cur = TimeDistributed(Conv2D(32,1,strides=(2,2),padding='same',data_format='channels_last'),name="time_distributed_cur_1")(cur_inp)
    cur = ConvLSTM2D(128,3,strides=2,padding='same',return_sequences=True,name="conv_lst_m2d_cur_1")(cur)
    inp = TimeDistributed(Conv2D(32,1,strides=(2,2),padding='same',data_format='channels_last'),name="time_distributed_1")(x_inp)
    inp = ConvLSTM2D(128,3,strides=2,padding='same',return_sequences=True,name="conv_lst_m2d_1")(inp)
    inp = Multiply(name="add_1")([inp,cur])
    inp = ConvLSTM2D(256,3,strides=2,padding='same',return_sequences=True,name="conv_lst_m2d_2")(inp)
    inp = ConvLSTM2D(256,3,strides=2,padding='same',return_sequences=True,name="conv_lst_m2d_3")(inp)
    inp = TimeDistributed(Conv2D(16,1,strides=(2,2),padding='same'),name="time_distributed_2")(inp)
    inp = TimeDistributed(Conv2D(256,1,strides=(1,1),padding='same'),name="time_distributed_3")(inp)
    inp = ConvLSTM2D(256,3,padding='same',return_sequences=True,name="conv_lst_m2d_4")(inp)
    inp = TimeDistributed(Conv2DTranspose(256,1,strides=(2,2),padding='same',kernel_initializer='he_normal'),name="time_distributed_4")(inp)
    inp = ConvLSTM2D(256,3,padding='same',return_sequences=True,name="conv_lst_m2d_5")(inp)
    inp = TimeDistributed(Conv2DTranspose(256,3,strides=(2,2),padding='same'),name="time_distributed_5")(inp)
    inp = ConvLSTM2D(128,3,padding='same',return_sequences=True,name="conv_lst_m2d_6")(inp)
    inp = TimeDistributed(Conv2DTranspose(128,1,strides=(2,2),padding='same'),name="time_distributed_6")(inp)
    inp = ConvLSTM2D(64,3,padding='same',return_sequences=True,name="conv_lst_m2d_7")(inp)
    inp = TimeDistributed(Conv2DTranspose(64,3,strides=(2,2),padding='same'),name="time_distributed_7")(inp)
    inp = ConvLSTM2D(64,3,padding='same',return_sequences=False,name="conv_lst_m2d_8")(inp)
    inp = Conv2DTranspose(64,3,strides=(2,2),padding='same',name="conv2d_transpose_5")(inp)
    inp = Conv2D(3,1,strides=(1,1),padding='same',name="conv2d_4")(inp)
    output = Lambda(my_reshape,arguments={'shape':(-1,1568,32,3)},name='output')(inp)
else:
    x_inp=Input(shape=(5,1568,32,3),name='input_1')
    inp = TimeDistributed(Conv2D(32,1,strides=(2,2),padding='same',data_format='channels_last'),name="time_distributed_1")(x_inp)
    inp = ConvLSTM2D(128,3,strides=2,padding='same',return_sequences=True,name="conv_lst_m2d_1")(inp)
    inp = ConvLSTM2D(256,3,strides=2,padding='same',return_sequences=True,name="conv_lst_m2d_2")(inp)
    inp = ConvLSTM2D(256,3,strides=2,padding='same',return_sequences=True,name="conv_lst_m2d_3")(inp)
    inp = ConvLSTM2D(512,3,strides=2,padding='same',return_sequences=True,name="conv_lst_m2d_4")(inp)
    inp = TimeDistributed(Conv2D(16,1,strides=(1,1),padding='same'),name="time_distributed_2")(inp)
    inp = TimeDistributed(Conv2D(256,1,strides=(1,1),padding='same'),name="time_distributed_3")(inp)
    inp = ConvLSTM2D(256,3,padding='same',return_sequences=True,name="conv_lst_m2d_5")(inp)
    inp = TimeDistributed(Conv2DTranspose(256,1,strides=(2,2),padding='same',kernel_initializer='he_normal'),name="time_distributed_4")(inp)
    inp = ConvLSTM2D(256,3,padding='same',return_sequences=True,name="conv_lst_m2d_6")(inp)
    inp = TimeDistributed(Conv2DTranspose(256,3,strides=(2,2),padding='same'),name="time_distributed_5")(inp)
    inp = ConvLSTM2D(128,3,padding='same',return_sequences=True,name="conv_lst_m2d_7")(inp)
    inp = TimeDistributed(Conv2DTranspose(128,1,strides=(2,2),padding='same'),name="time_distributed_6")(inp)
    inp = ConvLSTM2D(64,3,padding='same',return_sequences=True,name="conv_lst_m2d_8")(inp)
    inp = TimeDistributed(Conv2DTranspose(64,3,strides=(2,2),padding='same'),name="time_distributed_7")(inp)
    inp = ConvLSTM2D(64,3,padding='same',return_sequences=False,name="conv_lst_m2d_9")(inp)
    inp = Conv2DTranspose(64,3,strides=(2,2),padding='same',name="conv2d_transpose_5")(inp)
    inp = Conv2D(3,1,strides=(1,1),padding='same',name="conv2d_4")(inp)
    output = Lambda(my_reshape,arguments={'shape':(-1,1568,32,3)},name='output')(inp)
    
model=Model(inputs=[x_inp],outputs=[output])
model.compile(Adam(lr=0.0003),loss=loss_tf)
if args.curvature:
    model.fit_generator(generator_with_cur(train_xx_tensor,cur_xx_tensor,train_xx_tensor[:,4],batch_size=4),steps_per_epoch=1,nb_epoch=100000)
else:
    model.fit_generator(generate_batch_data_random(train_xx_tensor,train_xx_tensor[:,4],batch_size=1),steps_per_epoch=1,nb_epoch=100000)
    
feature_layer_model = Model(inputs=model.input,outputs=model.get_layer('time_distributed_2').output)
feature = feature_layer_model.predict(train_xx_tensor)

np.save('feature.npy',feature)
if args.curvature:
    result=model.predict(x={'input_1':train_xx_tensor,'cur_input_1':cur_xx_tensor},steps=1)
else:
    result=model.predict(x={'input_1':train_xx_tensor},steps=1)

np.save('result.npy',result)



