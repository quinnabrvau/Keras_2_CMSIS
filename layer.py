#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 16:54:29 2018

@author: quinn
"""
from copy import deepcopy

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
    def __init__(self, config=None, weights=None, prefix=''):
        if 'name' in config.keys():
            self.name = prefix+config['name']
        if 'activation' in config.keys():
            self.activation=activation_map(config['activation'])
        self.config = config
        self.weights = weights
    
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
            if length == -1:
                return self.input_shape[0]
            elif length != self.input_shape[0]:
                raise Warning('setting the input length for a cnn1d that expected a fixed length '+str(length))
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
        index = list(self.weights.keys())[0]
        for k in self.weights[index].keys():
            if 'kern' in k:
                return self._p_to_array('_KERN', self.weights[index][k].T)
        return ''
    def p_bias(self):
        index = list(self.weights.keys())[0]
        for k in self.weights[index].keys():
            if 'bias' in k:
                return self._p_to_array('_BIAS', self.weights[index][k])
        return ''

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
                out += self._p_macro('BIAS', self.weights[index][k])
            if 'kern' in k:
                out += self._p_macro('KERN', self.weights[index][k])
        return out

    def opt(self,mode='basic'):
        pass

class Input(layer):
    def __init__(self, config=None, weights=None, prefix=''):
        # print(config,weights,prefix)
        layer.__init__(self, config, weights, prefix)
        self.input_shape = config['batch_input_shape'][1:]
        self.output_shape = self.input_shape
    def p_func_call(self, **args):
        return ''
    