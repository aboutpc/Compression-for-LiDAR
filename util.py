
# coding=utf-8

import numpy as np
import math
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input

def get_outputs_of(model, start_tensors, input_layers=None):
    model = Model(inputs=model.input,
                  outputs=model.output,
                  name='outputs_of_' + model.name)
    if not isinstance(start_tensors, list):
        start_tensors = [start_tensors]
    if input_layers is None:
        input_layers = [
            Input(shape=K.int_shape(x)[1:], dtype=K.dtype(x))
            for x in start_tensors
        ]
    elif not isinstance(input_layers, list):
        input_layers = [input_layers]
    model.inputs = start_tensors
    model._input_layers = [x._keras_history[0] for x in input_layers]
    if len(input_layers) == 1:
        input_layers = input_layers[0]
    layers, tensor_map = [], set()
    for x in model.inputs:
        tensor_map.add(str(id(x)))
    depth_keys = list(model._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    for depth in depth_keys:
        nodes = model._nodes_by_depth[depth]
        for node in nodes:
            n = 0
            for x in node.input_tensors:
                if str(id(x)) in tensor_map:
                    n += 1
            if n == len(node.input_tensors):
                if node.outbound_layer not in layers:
                    layers.append(node.outbound_layer)
                for x in node.output_tensors:
                    tensor_map.add(str(id(x)))
    model._layers = layers
    outputs = model(input_layers)
    return input_layers, outputs
