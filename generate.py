
# coding: utf-8
from random import randint

def generate_batch_data_random_with_cur_den(x_z,x_y,c,d,y,batch_size):
    """逐步提取batch数据到显存，降低对显存的占用"""
    ylen = y.shape[0]
    loopcount = ylen // batch_size
    while (True):
        i = randint(0,loopcount)
        #result = tf.convert_to_tensor(result)
        yield ({'input_1': x_z[i * batch_size:(i + 1) * batch_size],'input_2': x_y[i * batch_size:(i + 1) * batch_size], 'cur_input_1': c[i * batch_size:(i + 1) * batch_size], 'den_input_1': d[i * batch_size:(i + 1) * batch_size]}, {'output': y[i * batch_size:(i + 1) * batch_size]})

def generate_batch_data_random(x, y, batch_size):
    ylen = y.shape[0]
    loopcount = ylen // batch_size
    while (True):
        i = randint(0,loopcount)
        result = x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]
        yield result