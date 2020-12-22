#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:49:10 2020

@author: Enrico Regolin
"""

#%%
import torch
from tqdm import tqdm
import random
import numpy as np

from .memory import unpack_batch


#%%

class RL_Updater():
    def __init__(self,rl_env, reset_optimizer = False):
        print('initialization started')
        
        self.beta = 0.001
        
        self.update_attributes(rl_env, init = True)
        
        self.nn_updating = False
        self.load_model(reset_optimizer)
        print('initialization successful')
        
    ##################################################################################        
    # required to check if model is inherited
    def update_attributes(self,rl_env, init = False):
        self.rl_env = rl_env
        self.memory_pool = rl_env.memory_stored 
        self.n_epochs = rl_env.N_epochs 
        self.move_to_cuda = rl_env.move_to_cuda 
        self.gamma = rl_env.gamma
        self.rl_mode = rl_env.rl_mode
        self.pg_partial_update = rl_env.pg_partial_update
        
        if self.move_to_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        if not init:
            if self.memory_pool is None:
                raise("memory pool is None object!!")
            else:
                print('memory pool loaded correctly')
        
    #################################################
    def save_model(self, path, model_name):
        self.model_qv.save_net_params(path, model_name)
        if self.rl_mode == 'AC':
            self.model_pg.save_net_params(path, model_name+'_policy')
        
    ##################################################################################        
    # required to check if model is inherited
    def getAttributeValue(self, attribute):
        if attribute in self.__dict__.keys():
            return self.__dict__[attribute]  

    ##################################################################################        
    # required since ray wrapper doesn't allow accessing attributes
    def getAttributes(self):
        return [key for key in self.__dict__.keys()]  
        
    ##################################################################################        
    # required since ray wrapper doesn't allow accessing attributes
    def setAttribute(self,attribute,value):
        if attribute in self.__dict__.keys():
            self.__dict__[attribute] = value
        
        
    #################################################
    # loading occurs based on current iteration loaded in the RL environment
    def load_model(self, reset_optimizer = False):
        self.model_qv = self.rl_env.generate_model()
        if self.move_to_cuda:
            self.model_qv = self.model_qv.cuda()

        if self.rl_env.training_session_number >= 1:
            self.model_qv.load_net_params(self.rl_env.storage_path,self.rl_env.net_name, self.device, reset_optimizer = reset_optimizer)
        else:
            self.model_qv.init_weights()

        if self.rl_mode == 'AC':
            self.model_pg = self.rl_env.generate_model(pg_model=True)
            if self.move_to_cuda:
                self.model_pg = self.model_pg.cuda()
            try:
                self.model_pg.load_net_params(self.rl_env.storage_path,self.rl_env.net_name+'_policy', self.device, reset_optimizer = reset_optimizer)
            except Exception:
                raise('Existing PG model not found!')
    
    
    #################################################
    def update_DeepRL(self, net = 'state_value', policy_memory = None):
        """synchronous update of Reinforcement Learning Deep Network"""
        
        self.nn_updating = True
        print(f'Synchronous update started: {net}')
        total_loss = []
        total_mismatch = 0
            
        if net == 'state_value':

            for epoch in tqdm(range(self.n_epochs)):

                loss = self.qValue_loss_update(*self.memory_pool.extractMinibatch())
                total_loss.append(loss.cpu().detach().numpy())
                
            self.model_qv.model_version +=1

        elif net == 'policy' and policy_memory is not None :
                        
            loss, n_mismatch = self.policy_loss_update(*unpack_batch(policy_memory.memory)[:-1])
            total_loss = loss.cpu().detach().numpy()
    
            invalid_samples_pctg =  np.round(100*n_mismatch/ policy_memory.getActualSize(), 2)
            print(f'Update finished. mismatch % =  {invalid_samples_pctg}%') 
            
            if self.pg_partial_update:
                self.model_pg.load_conv_params(self.rl_env.storage_path,self.rl_env.net_name, self.device)
            
            self.model_pg.model_version +=1

        else:
            raise('Undefined Net-type')
        
        self.nn_updating = False
        return total_loss
    
    
    #################################################
    def policy_loss_update(self, state_batch, action_batch, reward_batch, state_1_batch):
        
        if self.move_to_cuda:  # put on GPU if CUDA is available
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()
        
        # we re-compute the probabilities, this time in batch (they have to match with the outputs obtained during the simulation)
        prob_distribs_batch = self.model_pg(state_batch.float())   
        action_idx_batch = torch.argmax(prob_distribs_batch, dim = 1)
        action_batch_grad = torch.zeros(action_batch.shape).cuda()
        action_batch_grad[torch.arange(action_batch_grad.size(0)),action_idx_batch] = 1
        
        self.model_pg.optimizer.zero_grad()
            
        prob_action_batch = prob_distribs_batch[torch.arange(prob_distribs_batch.size(0)), action_idx_batch].unsqueeze(1)
        entropy = -torch.sum(prob_distribs_batch*torch.log(prob_distribs_batch),dim = 1).unsqueeze(1)
        
        with torch.no_grad():
            advantage = reward_batch + self.gamma *torch.max(self.model_qv(state_1_batch.float()))\
                - torch.max(self.model_qv(state_batch.float()))

        if (action_batch == action_batch_grad).all().item():
            loss_policy = torch.mean( -torch.log(prob_action_batch)*advantage + self.beta*entropy )
            n_invalid = 0
            
        else:
            print('WARNING: selected action mismatch detected')
            valid_rows   = (action_batch == action_batch_grad).all(dim = 1)
            invalid_rows = (action_batch != action_batch_grad).any(dim = 1)
                    
            loss_policy = torch.mean( -torch.log(prob_action_batch[valid_rows])*advantage[valid_rows] + self.beta*entropy[valid_rows] )
            n_invalid = torch.sum(invalid_rows).item()
            
            n_mismatch = invalid_rows.nonzero().squeeze(1).shape[0]
            if n_mismatch > 10:
                print(f'{n_mismatch} mismatches occurred!')
                raise('mismatch problem')
        
        loss_policy.backward()
        self.model_pg.optimizer.step()  
        
        return loss_policy, n_invalid


    
    #################################################
    def qValue_loss_update(self, state_batch, action_batch, reward_batch, state_1_batch, done_batch):
        
        if self.move_to_cuda:  # put on GPU if CUDA is available
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()

        # get output for the next state
        with torch.no_grad():
            output_1_batch = self.model_qv(state_1_batch.float())
        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if done_batch[i]  #minibatch[i][4]
                                  else reward_batch[i] + self.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(reward_batch))))
        # extract Q-value
        # calculates Q value corresponding to all actions, then selects those corresponding to the actions taken
        q_value = torch.sum(self.model_qv(state_batch.float()) * action_batch, dim=1)
        
        self.model_qv.optimizer.zero_grad()
        y_batch = y_batch.detach()
        loss_qval = self.model_qv.criterion_MSE(q_value, y_batch)
        loss_qval.backward()
        self.model_qv.optimizer.step()  
        
        return loss_qval

