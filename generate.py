
# coding: utf-8


def generator_with_cur(x,cur,y, batch_size):
    ylen = y.shape[0]
    loopcount = ylen // batch_size
    while (True):
        i = randint(0,loopcount)
        result = ({'input_1':x[i * batch_size:(i + 1) * batch_size],'cur_input_1':cur[i * batch_size:(i + 1) * batch_size]},{'output':y[i * batch_size:(i + 1) * batch_size]})
        yield result

def generate_batch_data_random(x, y, batch_size):
    ylen = y.shape[0]
    loopcount = ylen // batch_size
    while (True):
        i = randint(0,loopcount)
        result = x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]
        yield result