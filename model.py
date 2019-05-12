#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 17:00:13 2018

@author: quinn
"""

from copy import deepcopy

from layer import Input, Activation
from layer1d import Conv1d, Up_sample1d,  Max_pool1d, Ave_pool1d

def choose_layer(name):
    if 'input' in name:
        return Input
    elif 'conv1d' in name:
        return Conv1d
    elif 'max_pooling1d' in name:
        return Max_pool1d
    elif 'ave_pooling1d' in name:
        return Ave_pool1d
    elif 'up_sampling1d' in name:
        return Up_sample1d
    elif 'gaus' in name.lower():
        return None # don't include noise layers in final implementation
    else:
        raise Exception('No layer implemented for ' + name)

class model(list):
    name = ''
    static = False
    fixed = False
    header = {}
    def __init__(self, name = '_model_missing_name_'):
        list.__init__(self)
        self.name = name

    def add_layer(self, name, config, weights):
        layer_f = choose_layer(name)
        if layer_f is None:
            return
        layer = layer_f(config, weights, self.name)
        if len(self)==0:
            if isinstance(layer,Input):
                self.append(layer)
            else:
                self.append(Input(config, None, self.name))
                self.add(layer)
            self.fixed = (layer.input_shape[0]!=None)
        else:
            self.add(layer)
        if not self.fixed:
            raise Exception("This branch doesn't handle non fixed shape inputs")

    def add(self,layer):
        self.append(layer)
        self[-2].Next = self[-1]
        self[-1].Prev = self[-2]
        self[-1].input_shape = deepcopy(self[-2].output_shape)
        self[-1].set_output_shape()
        if layer.activation != 'none':
            self.append(Activation(prefix=self.name+layer.name+' activation', activation = layer.activation))
            layer.activation == 'none'
            self[-1].input_shape = deepcopy(self[-2].output_shape)
            self[-1].output_shape = self[-1].input_shape

        
    def __str__(self):
        out = self.name + '\n'
        for layer in self:
            out += str(layer)
        return out
    
    def p_def(self):
        out = ''
        for lay in self:
            out += lay.p_def()
        return out
    
    def p_init(self):
        self.header['init'] = 'void ' + self.name + '_init( void )'
        out = self.header['init'] + ' {\n'
        for lay in self:
            foo = lay.p_init()
            for line in foo.splitlines():
                out += '\t' + line + '\n'
        out += '}\n'
        return out
    
    def p_func_call(self):
        out = ''
        buffers, buf_size = ['buffer1','buffer2'], [0,0]
        index, last_index = 1, 1
        buf = 'signal'
        
        params = ['q7_t *signal', 'q7_t *out_buffer']
        bufA, bufB = 0, 0
        
        call_code = ''
   
        for lay in self[1:]:
            foo = ''
            foo = lay.p_func_call(sig=buf, dst=buffers[index], bufA='bufA', bufB='bufB')
            bufA = max(bufA, lay.get_bufA_size())
            bufB = max(bufB, lay.get_bufB_size())
            out_size = lay.get_out_size()
            buf_size[index] = max(out_size, buf_size[index])
            for line in foo.splitlines():
                call_code += '\t' + line + '\n'
            last_index = index
            if not lay.inPlace:
                buf = buffers[index]
                index = 1-index
                
            

        call_code = call_code.replace(buffers[1-last_index],'out_buffer')
        call_code = call_code.replace(buffers[last_index],'int_buffer')
        if self.fixed:
            call_code = '\tq7_t  int_buffer[' + str(buf_size[last_index]) + '];\n' + call_code
            call_code = '\tq15_t bufferA[' + str(bufA) + '];\n' + call_code
            call_code = '\tq7_t  bufferB[' + str(bufB) + '];\n' + call_code
        else:
            params.append('q7_t  *int_buffer')
            params.append('q15_t *bufferA')
            params.append('q7_t  *bufferB')
            params.append('uint32_t len')
         

            foo =  "/** buffer sizes\n"
            foo += " * %s\tsize\t%d * len\n" % ('bufferA', bufA)
            foo += " * %s\tsize\t%d * len\n" % ('bufferB', bufB)
            foo += " * %s\tsize\t%d * len\n" % ('out_buffer', buf_size[1-last_index])
            foo += " * %s\tsize\t%d * len\n" % ('int_buffer', buf_size[last_index])
            foo += " **/"
            self.header['size'] = foo

        if self.static:
            out = 'static\n'
        self.header['fn'] = out + 'void ' + self.name + '_fn( ' + ', '.join(params) + ' )'
        out += '\n\n' + self.header['fn'] +' {\n'
        out += call_code
        out += '}\n'
        return out
    
    def p_header(self):
        out  = '#ifndef ' + self.name + '_h_\n'
        out += '#define ' + self.name + '_h_\n'
        out += '\n'
        deps = ['arm_math.h','arm_nnfunctions.h']
        # for lay in self:
            # deps += lay.dep()
        deps = list(set( deps ))
        for dep in deps:
            out += '#include "'+dep+'"\n'
            
        out += '\n'
        for lay in self:
            foo = lay.p_macro()
            if foo != '':
                out += foo + '\n\n'

        out += '\n'
        for key in self.header.keys():
            out += self.header[key] + ';\n'
            
        out += '#endif//' + self.name + '_h_\n'
        return out

    def p_test(self):
        ## TODO
        return ''
