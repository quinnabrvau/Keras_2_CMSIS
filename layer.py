#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 16:54:29 2018

@author: quinn
"""
from copy import deepcopy
import numpy as np

def activation_map(_act):
    act = _act.lower()
    if 'tanh' == act:
        return 'tanh'
    elif 'hard_sigmoid' == act or 'sigmoid' == act:
        return 'sigmoid'
    elif 'linear' == act or 'none' == act or '' == act:
        return 'none'
    elif 'relu' == act:
        return 'relu'
    raise Exception("activation not properly mapped to c function")

def keras_name_fix(name):
    n = name.replace(':','_')
    n = n.replace('kern','KERN')
    return n.replace('bias','BIAS')

class layer:
    name = '_missing_'
    c_function = '_missing_'
    py_function = '__TODO__'
    input_shape = () 
    output_shape = ()
    Next = None
    Prev = None
    config = None
    weights = None
    data = {}
    activation = 'none'
    kern = np.asarray([])
    bias = np.asarray([])
    def __init__(self, config=None, weights=None, prefix=''):
        if config is not None and 'name' in config.keys():
            self.name = prefix+config['name']
        else:
            self.name = prefix
        if config is not None and 'activation' in config.keys():
            self.activation=activation_map(config['activation'])
        self.config = config
        self.weights = weights

        if weights is not None:
            Keys = list(self.weights.keys())
            if len(Keys) > 0:
                index = Keys[0]
                for k in self.weights[index].keys():
                    if 'kern' in k:
                        self.kern = self.weights[index][k].T
                    elif 'bias' in k:
                        self.bias = self.weights[index][k]
        
    def find_h5(self, h5_path='', keys=None, h5_file=None):
        if h5_file is None:
            raise Exception("find_h5: No H5 file provided")
        if keys is None:
            if h5_path=='':
                raise Exception('find_h5: No keys provided to layer')
            keys = h5_path.split('\\')
        elif h5_path!='':
            raise Exception('find_h5: 2 keys provided')
        foo = h5_file
        for k in keys:
            foo = foo[k]
        for k in foo.keys():
            self.data[keras_name_fix(k)] = foo[k]
            
    def __str__(self):
        # print(str(self.input_shape),str(self.output_shape))
        foo = [self.name,str(self.input_shape),str(self.output_shape)]
        out = '\t'.join(foo) + '\n'
        for key in self.config.keys():
            out += '\n\t' + str(key) + ' : ' + str(self.config[key])
        return out +'\n'
    
    def p_def(self):
        return ''
    
    def p_func_call(self, **args):
        return self.c_function + '(' + ', '.join(args) + ');\n'
    
    def set_output_shape(self):
        self.output_shape = deepcopy(self.input_shape)
        
    def get_out_size(self,length=-1):
        if isinstance(length,str):
            return self.output_shape[1]
        length = self.size_check(length)
        return length*self.output_shape[1]
    
    def size_check(self,length,strOK=False):
        if strOK and isinstance(length,str):
            return length
        elif self.input_shape[0] == None:
            if length == -1:
                raise Exception('Input:p_func_call: No input length given for arbitrary length cnn1d')
        else:
            return self.input_shape[0]
        return length

    def _p_array(self,array):
        # print(len(array.shape), array.shape, array)
        if len(array.shape) == 1:
            return '{' + ','.join([str(i) for i in array]) + '}'
        else:
            return '{' + ','.join([self._p_array(array[i]) for i in range(array.shape[0])]) + '}'

    def _p_to_array(self, name, array):
        out = 'float32_t ' + self.name + name
        for s in array.shape:
            out += '[' + str(s) + ']'
        out += ' = ' + self._p_array(array) + ';'
        return out 

    def p_kern(self):
        return self._p_to_array('_KERN', self.kern)
        

    def p_bias(self):
        return self._p_to_array('_BIAS', self.bias)

    def _p_macro(self,name,array):
        out = ''
        for i,s in enumerate(array.shape):
            out += '#define ' + self.name.upper() + '_SIZE_' + name.upper() + '_%d (%d)\n' % (i, s) ;
        return out;

    def p_macro(self):
        out = ''
        if (self.weights is None): return ''
        Index = list(self.weights.keys())
        if len(Index) == 0: return ''
        index = Index[0]
        for k in self.weights[index].keys():
            if 'bias' in k:
                out += self._p_macro('BIAS', self.weights[index][k]) + '\n'
            if 'kern' in k:
                out += self._p_macro('KERN', self.weights[index][k]) + '\n'
        return out

    def opt(self,mode='basic'):
        pass

    def get_bufA_size(self, length=-1):
        return 0

    def get_bufB_size(self, length=-1):
        return 0

class Input(layer):
    def __init__(self, config=None, weights=None, prefix=''):
        # print(config,weights,prefix)
        layer.__init__(self, config, weights, prefix)
        self.input_shape = config['batch_input_shape'][1:]
        self.output_shape = self.input_shape
    def p_func_call(self, **args):
        return ''
    
class Activation(layer):
    c_function = 'arm_nn_activations_direct_q7'
    int_width = 0; ##TODO: calculate

    def __init__(self, config=None, weights=None, prefix='', activation=None):
        layer.__init__(self, config, weights, prefix)
        if (activation is not None):
            self.activation = activation_map(activation)
        else:
            if (self.activation == 'none'):
                raise Warning('No Activation function detected')

        if (self.activation == 'relu'):
            c_function = 'arm_relu_q7'


    def p_func_call(self, sig='_needs_source_' , length=-1, **args):
        _params = []
        if (self.activation == 'relu'):
            self.c_function = 'arm_relu_q7'
            _params = [sig, str(self.size_check(length, True))]

        elif (self.activation == 'tanh'):
            self.c_function = 'arm_nn_activations_direct_q7'
            _params = [sig, str(self.size_check(length, True)), str(self.int_width), 'ARM_TANH']

        elif (self.activation == 'sigmoid'):
            self.c_function = 'arm_nn_activations_direct_q7'
            _params = [sig, str(self.size_check(length, True)), str(self.int_width), 'ARM_SIGMOID']

        else:
            raise Exception('Activation function ('+self.activation+') is not implemented in this library')

        print(_params)
        return self.c_function + '(' + ', '.join([str(a) for a in _params]) + ');'
    
    
    
    
    
    
    
    
    