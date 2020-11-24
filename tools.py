
# coding=utf-8

import numpy as np
import math
import pandas as pd
import tensorflow as tf
import os
from keras.callbacks import TensorBoard

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./Graph', **kwargs):
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

def obj_conver_to_array(path):
    read_in2=pd.read_csv(path,header=None)
    train2=read_in2[0:34834]
    train_raw2=train2[0].str.split(' ',expand=True)
    train_raw2=train_raw2.drop([0],axis=1)
    train2=train_raw2.astype('float32')  
    train_num2=np.array(train2)
    return train_num2

def ply_convert_to_array(path):
    read_in = pd.read_csv(path)
    train=read_in[11:35958]
    train_raw=train['ply'].str.split(' ',expand=True)
    train=train_raw.drop([3,4,5],axis=1)
    train=train.astype('float32')  
    train_num=np.array(train)
    return train_num
