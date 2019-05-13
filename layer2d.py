#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:30:39 2019

@author: quinn
"""

from copy import deepcopy

from layer import Layer

class Layer2d(Layer):
    def get_pad(self):
        if 'padding' not in self.config.keys():
            return (0,0)
        pad = self.config['padding']
        kern = self.get_kernel()
        if pad == 'valid':
            return (0,0)
        elif pad == 'same':
            return [i//2 for i in kern]
        elif pad == 'causal':
            return kern
        return (0,0)

    def get_strides(self):
        if 'strides' not in self.config.keys():
            return (0,0)
        strides = self.config['strides']
        if isinstance(strides, int):
            return [strides, strides]
        elif len(strides) == 1:
            return strides + strides
        elif len(strides) == 0:
            return (0,0)
        else:
            return strides

    def get_kernel(self):
        if 'kernel_size' not in self.config.keys():
            raise Exception('No Kernel Size detected')
        kernel_size = self.config['kernel_size']
        if isinstance(kernel_size, int):
            return [kernel_size, kernel_size]
        elif len(kernel_size) == 1:
            return kernel_size + kernel_size
        elif len(kernel_size) == 2:
            return kernel_size
        else:
            raise Exception('Invalid kernel size detected ' + str(kernel_size) + ' for layer ' + self.name)


class Conv2d(Layer2d):
    c_function = 'arm_convolve_HWC_q7_basic_nonsquare'

    def opt(self,mode='basic'):
        self.c_function = 'arm_convolve_HWC_q7_basic_nonsquare'
        if 'fast' in mode:
            if (self.input_shape[-1] %4 == 0 or self.output_shape[-1]%2 == 0):
                self.c_function = 'arm_convolve_HWC_q7_fast_nonsquare'
            else:
                print("attempted to optimize",self.name,"but encountered unsupported shape, reverting to basic")
        elif 'q15' in mode:
            self.c_function = 'arm_convolve_HWC_q15_basic_nonsquare'

    def p_def(self):
        return '\n\n'.join([self.p_kern(), self.p_bias()]) + '\n\n'

    def p_func_call(self, sig='_needs_source_', dst='_needs_dest_', bufA='_needs_buf_A', bufB='_needs_buf_B', **args):
        pad = self.get_pad()
        strides = self.get_strides()
        kern = self.get_kernel()

        params = [ sig,
                self.input_shape[0], 
                self.input_shape[1], 
                self.input_shape[2], 
                '(const q7_t*)'+self.name+'_KERN', 
                self.config['filters'], 
                kern[0],
                kern[1],
                pad[0],
                pad[1],
                strides[0],
                strides[1],
                '(const q7_t*)'+self.name+'_BIAS',
                0, # self.bias_shift,
                0, # self.out_shift,
                dst, 
                self.output_shape[0],
                self.output_shape[1],
                bufA, bufB]

        return  self.c_function + '('+', '.join([ str(a) for a in params])+');\n'
    
    def set_output_shape(self):
        self.output_shape = deepcopy(self.input_shape)
        pad = self.config['padding']
        for i in range(2):
            if pad == 'valid':
                self.output_shape[i] = (self.output_shape[i] - self.config['kernel_size'][i] + 1 )//(self.config['strides'][i])
            elif pad == 'same':
                self.output_shape[i] = (self.output_shape[i])//(self.config['strides'][i])
            elif pad == 'causal':
                self.output_shape[i] = (self.output_shape[i])//(self.config['strides'][i])
        self.output_shape[-1] = self.config['filters']

    def __p_macro(self,name,array):
        pos = ['WINDOW','INPUT','OUTPUT']
        out = ''
        for i,s in enumerate(array.shape):
            out += '#define ' + '_'.join([self.name.upper(),'SIZE',name.upper(), pos[i]]) + ' (%d)\n' % (s) ;
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
                out += self.__p_macro('KERN', self.weights[index][k])
        return out
    
    def get_bufA_size(self):
        kern = max(self.get_kernel())
        return 2*self.input_shape[-1]*kern*kern

class Max_pool2d(Layer2d):
    c_function = 'arm_maxpool_q7_HWC_nonsquare'
    
    def get_kernel(self):
        if 'pool_size' not in self.config.keys():
            raise Exception('pool2d layer has no kernel size')
        kern = self.config['pool_size']
        if isinstance(kern, int):
            return (kern,kern)
        elif len(kern) == 1:
            return kern + kern
        elif len(kern) == 2:
            return kern
        else:
            raise Exception('pool2d layer has invalid kern shape ' + str(kern) + ' for layer ' + self.name)

    def p_func_call(self, sig='_needs_source_', dst='_needs_dest_', bufA='_needs_buf_A', bufB='_needs_buf_B', **args):
        dim_im_in_x, dim_im_in_y = self.input_shape[0], self.input_shape[1]
        chan_in = self.input_shape[2]
        dim_im_out_x, dim_im_out_y = self.output_shape[0], self.output_shape[1]
        kern_x, kern_y = self.get_kernel()
        strides_x, strides_y = self.get_strides()
        pad_x, pad_y = self.get_pad()
        params = [sig, 
                    dim_im_in_x, dim_im_in_y,
                    chan_in, 
                    kern_x, kern_y, 
                    pad_x, pad_y, 
                    strides_x, strides_y, 
                    dim_im_out_x, dim_im_out_y, 
                    bufB, dst]
        return  self.c_function + '('+', '.join([ str(a) for a in params])+');\n'
    
    def set_output_shape(self):
        kern = self.get_kernel()
        strides = self.get_strides()
        pad = self.get_pad()
        kern = self.get_kernel()
        self.output_shape = deepcopy(self.input_shape)
        for i,k in enumerate(kern):
            if self.output_shape[i] is not None:
                self.output_shape[i] = (self.output_shape[i]+pad[i]) // kern[i] // strides[i]

class Ave_pool2d(Max_pool2d):
    c_function = 'arm_avepool_q7_HWC_nonsquare'

    def get_bufB_size(self):
        return 2*self.output_shape[0]*self.input_shape[1]

class Up_sample2d(Layer2d):
    c_function = 'arm_upsample_q7_HWC_nonsquare'
    
    def get_kernel(self):
        if 'size' not in self.config.keys():
            raise Exception('Up_sample2d layer has no kernel size')
        kern = self.config['size']
        if isinstance(kern, int):
            return (kern,kern)
        elif len(kern) == 1:
            return kern + kern
        elif len(kern) == 2:
            return kern
        else:
            raise Exception('Up_sample2d layer has invalid kern shape ' + str(kern) + ' for layer ' + self.name)

    def p_func_call(self, sig='_needs_source_', dst='_needs_dest_', bufA='_needs_buf_A', bufB='_needs_buf_B', **args):
        dim_im_in_x, dim_im_in_y = self.input_shape[0], self.input_shape[1]
        chan_in = self.input_shape[2]
        kern_x, kern_y = self.get_kernel()
        params = [sig, 
                  dim_im_in_x, dim_im_in_y, 
                  chan_in, 
                  kern_x, kern_y,
                  bufB, dst]
        return  self.c_function + '('+', '.join([ str(a) for a in params])+');\n'
    
    def set_output_shape(self):
        kern = self.get_kernel()
        self.output_shape = deepcopy(self.input_shape)
        for i,k in enumerate(kern):
            if self.output_shape[i] is not None:
                self.output_shape[i] *= k
        