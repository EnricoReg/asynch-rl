#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:36:57 2021

@author: Enrico Regolin
"""

#%%
import os

import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_nn import nnBase
from .custom_networks import LinearModel

#%%

# item size = 1xN (periods are considered as channels)
# item size is not needed as input for the conv modules (different from linear layers)

#kernel_size_in = 3 or 5
#strides   = [5,3] --> strides are used in maxpool layers
#C1/C2:  number of channels out of first/second convolution (arbitrary)
# F1/F2: outputs of linear layers

#%%

class ConvModel(nnBase):
    # batch size is not an input to conv layers definition
    # make sure N%stride = 0 for all elements in strides
    def __init__(self, model_version = -1, net_type='ConvModel0',lr =1e-6,  n_actions = 9, channels_in = 4, \
                 N_in = [135,4] , channels = [16, 10, 4] , fc_layers = [60,60,20], **kwargs):
        
        #ConvModel, self
        super().__init__(model_version, net_type, lr)
        #if N_in[0] % (strides[0]*strides[1]):
        #    raise ValueError('N in and strides not aligned')
        
        self.N_in = N_in        
        self.channels = channels
        self.channels_in = channels_in
        self.channels.insert(0, channels_in)
        self.n_actions = n_actions
        self.F = fc_layers
        
        # in AC mode, following weights are updated separately
        self.independent_weights = ['fc_output.weight', 'fc_output.bias']
        for i in range(1, len(self.F)+1):
            self.independent_weights.append('fc' +str(i) +'.weight')
            self.independent_weights.append('fc' +str(i) +'.bias')

        # this should always be run in the child classes after initialization
        self.complete_initialization(kwargs)
        
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'total NN trainable parameters: {pytorch_total_params}')
        
    ##########################################################################        
    def build_generic_input(self):
        a,b = self.get_net_input_shape()
        return torch.randn(a), torch.randn(b)


    ##########################################################################        
    def get_net_input_shape(self):
        return (1, self.channels_in, self.N_in[0] ), (1, self.N_in[1])


    ##########################################################################
    def update_NN_structure(self):
        
        x_test, x_test_1 = self.build_generic_input()

        # convolutional layers
        self.maxpool_3 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.maxpool_5 = nn.MaxPool1d(kernel_size=5, stride=5)
        for i in range(len(self.channels)-1):
            layer = nn.Conv1d(in_channels= self.channels[i], out_channels=self.channels[i+1], \
                              kernel_size= 3+2*int(not bool(i)),  padding= 1+ int(not bool(i)) )             
            layer_name = 'conv'+str(i+1)
            setattr(self , layer_name , layer )
            
            x_test =  layer(x_test)
            if i == 0:
                x_test =  self.maxpool_5(x_test)
            else:
                x_test =  self.maxpool_3(x_test)
                
            layer_norm = nn.LayerNorm(x_test.shape[-1])
            setattr(self , 'norm_layer_conv'+str(i+1), layer_norm)
            x_test = layer_norm(x_test)
                
        # fully connected layers
        N_linear_in = round(self.N_in[0]/(5*3**(len(self.channels)-2))*self.channels[-1] + self.N_in[1])
        x_test = torch.cat( (x_test.flatten(), x_test_1.flatten() ),dim = 0 ).unsqueeze(0)
        
        lin_model = LinearModel(0, 'linear_portion', self.lr, self.n_actions, N_linear_in, *(self.F) ) 
        layers = []
        for layer in lin_model.state_dict().keys():
            layers.append(layer.rsplit('.')[0])
        layers = sorted(list(set(layers)))
        self.net_depth = len(layers)-1

        for i,layer in enumerate(layers):
            nn_layer = getattr(lin_model, layer)
            setattr(self, layer , nn_layer  )
            x_test = nn_layer (x_test)
            if 'fc'+str(i+1) in layer:
                layer_norm = nn.LayerNorm(x_test.shape[-1])
                setattr(self , 'norm_layer_fc'+str(i+1), layer_norm)
                x_test = layer_norm(x_test)
            

    ##########################################################################
    def maxpool(self, n):
        return nn.MaxPool1d(kernel_size= n, stride= n)


    ##########################################################################
    def forward(self,x):

        x = self.conv_forward(x)

        iterate_idx = 1
        for attr in sorted(self._modules):
            if 'fc'+str(iterate_idx) in attr:
                x = F.relu(self._modules[attr](x))
                x = self._modules['norm_layer_fc'+str(iterate_idx)](x)
                iterate_idx += 1
                if iterate_idx > self.net_depth:
                    break

        x = self.fc_output(x)

        if self.softmax:
            sm = nn.Softmax(dim = 1)
            x = sm(x)

        return x
    
    
    ##########################################################################
    def conv_forward(self,x):
        """ takes a tuple of two tensors as input """
        
        x1 = x[0]
        for i in range(1,len(self.channels)):
            layer = getattr(self, 'conv'+str(i))
            x1 = F.relu(layer(x1))
            if i == 1:
                x1 = self.maxpool_5(x1)
            else:
                x1 = self.maxpool_3(x1)
            x1 = self._modules['norm_layer_conv'+str(i)](x1)
            # x1 = self.
   
        return torch.cat((x1.flatten(1), x[1].flatten(1)),dim = 1)
    
    
    ##########################################################################
    # load network parameters of only convolutional layers (net structure is assumed correct!)
    def load_conv_params(self,path_log,net_name, device):    
        
        filename_pt = os.path.join(path_log, net_name + '.pt')
        checkpoint = torch.load(filename_pt, map_location=device)
        pretrained_dict = checkpoint['model_state_dict']

        model_dict = self.state_dict(self.independent_weights)
        
        # 1. filter out unnecessary keys
        pretrained_dict = {k : v for k, v in pretrained_dict.items() if k not in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.load_state_dict(pretrained_dict)

        
        self.to(device)  # required step when loading a model on a GPU (even if it was also trained on a GPU)
        self.eval()
    
    ##########################################################################
    # update net parameters based on state_dict() (which is loaded afterwards)
    def init_layers(self, model_state):         
        """ """                

        self.channels = [ model_state['conv1.weight'].shape[1] ]
        self.n_actions = model_state['fc_output.weight'].shape[0]

        i = 1
        while i< 10:
            conv_layer = 'conv' + str(i) + '.weight'
            if conv_layer in model_state:
                self.channels.append( model_state[conv_layer].shape[0] )
            i += 1

        self.F = []
        i = 1
        while i < 10:
            fc_layer = 'fc' + str(i) + '.weight'
            if fc_layer in model_state:
                self.F.append( model_state[fc_layer].shape[0] )
            i += 1
        
        N_linear_in = round(self.N_in[0]/(5*3**(len(self.channels)-2))*self.channels[-1] + self.N_in[1])
        if N_linear_in != model_state['fc1.weight'].shape[1]:
            raise("NN consistency error")


    ##########################################################################
    # compare weights before and after the update
    def compare_weights(self, state_dict_1, state_dict_0 = None):
        
        average_diff = super().compare_weights(state_dict_1, state_dict_0)
            
        return average_diff

