#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 17:13:10 2018

@author: quinn
"""

import h5py
import json
import numpy as np

from model import model


## Section on reading weights

def read_weights(weights):
    out = {}
    if isinstance(weights, h5py.Dataset):
        return np.asarray(weights)
    for k in weights.keys():
        out[k] = read_weights(weights[k])
    return out


## Section on reading config

def read_config(config):
    out = [[],[]]
    con = json.loads(config.decode('utf-8'))
    if isinstance(con['config'],list):
        return read_layers(con['config'])
    m = con['config']['layers']
    for layer in m:
        a, b = read_model(layer)
        out[0] += a
        out[1] += b
    return out
    
def read_model(m):
    out = [[],[]]
    a, b = read_layers(m['config'])
    out[0] += a
    out[1] += b
    return out

def read_layers(layers):
    out = [[],[]]
    if isinstance(layers,dict):
        return read_layer(layers)
    elif isinstance(layers,list):
        for l in layers:
            a, b  = read_layers(l)
            out[0] += a
            out[1] += b
    return out

def read_layer(layer):
    if 'config' in layer.keys():
        return [ [layer['config']['name'] ], [layer['config']] ]
    elif 'name' in layer.keys():
        return [ [layer['name'] ], [layer] ]
    raise Exception(" unable to parse layer ")


## Section on building model


def build_model(filename, name='NONE'):
    m = model(name)
    file = h5py.File(filename)
    w_index, layers, c_index = None, None, None
    for k in file.keys():
        if 'model_weights' in k:
            w_index = read_weights(file[k])
    if w_index is None:
        raise Exception("no model weights read")
    if 'model_config' in file.attrs.keys():
            layers, c_index = read_config(file.attrs['model_config'])
    for i,n in enumerate(layers):
        if n in w_index.keys():
            print(n)
            m.add_layer(n, c_index[i], w_index[n])
        else:
            m.add_layer(n,c_index[i],None)
    file.close()
    return m

def convert_model(filename, name='NONE', path='./', verbose=True):
    m = build_model(filename, name)
    if verbose:
        print(m.p_def())
        print(m.p_func_call())
        print(m.p_header())

    file = open( path + name + '.c','w')
    file.write("""/**This file was auto generated using the NN 2 CMSIS library
 * More information can be found at github.com/quinnabrvau/Keras_2_CMSIS
 **/\n""")
    file.write('#include "' + name + '.h"\n')
    file.write(m.p_def())
    file.write(m.p_func_call())
    file.close()

    file = open( path + name + '.h','w')
    file.write(m.p_header())
    file.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="program to takes keras files and converts them to strive C, H files")
    parser.add_argument('model', help="path to source model")
    parser.add_argument('out_name', default='_no_name_', help="prefix for function calls and file name")
    parser.add_argument('out_path', default='./', help="path to put files") 
    # parser.add_argument('-V','--verbose')
    # parser.add_argument('-t','--test')
    args = parser.parse_args()    

    print("saving output in",args.out_path + args.out_name + '.c',
        ' and ', args.out_path + args.out_name + '.h')

    convert_model(args.model, 
                  path = args.out_path, 
                  name = args.out_name, 
                  verbose = True)



