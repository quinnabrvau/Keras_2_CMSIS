#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 17:00:13 2018

@author: quinn
"""

from copy import deepcopy

from layer import Input
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

    def add(self,layer):
        self.append(layer)
        self[-2].Next = self[-1]
        self[-1].Prev = self[-2]
        self[-1].input_shape = deepcopy(self[-2].output_shape)
        self[-1].set_output_shape()
        
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
        buffers, buf_size, index = ['buffer1','buffer2'], [0,0], 1
        buf = 'signal'
        
        params = ['float32_t *signal', 'float32_t *out_buffer']
        conv_buf_size = 0
       
        length = -1
        if not self.fixed:
            length = 'len'
        else:
            length = self[0].input_shape[0]
        
        
            
        call_code = ''

        # print([lay.name for lay in self])
        # print(self[0])
        # print(self[1])        
        for lay in self[1:]:
            index = 1-index
            foo = lay.p_func_call(length=length, sig=buf, dst=buffers[index], buf='conv_buf')
            if isinstance(lay, Conv1d):
                conv_buf_size = max(conv_buf_size,lay.get_buf_size(length))
            out_size = lay.get_out_size(length)
            # print(lay.name, index,buffers[index],buf_size[index],out_size,max(out_size, buf_size[index]))
            buf_size[index] = max(out_size, buf_size[index])
            for line in foo.splitlines():
                call_code += '\t' + line + '\n'
            if not isinstance(lay,Input):   
                buf = buffers[index]
           
        call_code = call_code.replace(buffers[index],'out_buffer')
        call_code = call_code.replace(buffers[1-index],'int_buffer')
        if self.fixed:
            call_code = 'float32_t int_buffer[' + buf_size[1-index] + '];\n' + call_code
        else:
            params.append('float32_t *int_buffer')
            params.append('float32_t *conv_buf')
            params.append('uint32_t len')
         

        foo =  "/* buffer sizes\nconv_buffer\tsize\t%d + len\n" % conv_buf_size
        foo += "%s\tsize\t%d * len\n" % ('out_buffer', buf_size[index])
        foo += "%s\tsize\t%d * len\n */" % ('int_buffer', buf_size[1-index])
        self.header['size'] = foo

        if self.static:
            out = 'STATIC_UNLESS_TESTING\n'
        self.header['fn'] = out + 'void ' + self.name + '_fn( ' + ', '.join(params) + ' )'
        out = self.header['fn'] +' {\n'
        out += call_code
        out += '}\n'
        return out
    
    def p_header(self):
        out  = '#ifndef ' + self.name + '_h_\n'
        out += '#define ' + self.name + '_h_\n'
        out += '\n'
        deps = []
        for lay in self:
            deps += lay.dep()
        deps = list(set( deps ))
        for dep in deps:
            out += '#include "'+dep+'"\n'
            
        out += '\n'
        for lay in self:
            out += lay.p_macro()

        out += '\n'
        for key in self.header.keys():
            out += self.header[key] + ';\n'
            
        out += '#endif//' + self.name + '_h_\n'
        return out

    def p_test(self):
        ## TODO
        return ''
