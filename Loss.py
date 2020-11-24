
# coding=utf-8

import numpy as np
import math
import tensorflow as tf

def loss_tf(pc1,pc2):
    diff_all=pc1-pc2
    part1 = tf.sqrt(tf.reduce_mean(diff_all**2))
    part2 = tf.reduce_mean(tf.abs(diff_all))
    return -20*(tf.log(1.0/part1)/tf.log(10.0))+0+100

def loss_np(pc1,pc2):
    diff_all=pc1-pc2
    part1 = math.sqrt(np.mean(diff_all**2))
    part2 = np.mean(np.abs(pc1 - pc2))
    return -20*math.log10(1/part1)+0*part2+100

def PSNR_loss(pc1,pc2):
    diff_all=pc1-pc2
    mse = tf.sqrt(tf.reduce_mean(diff_all**2))
    return 100-20*(tf.log(1.0/mse)/tf.log(10.0))
