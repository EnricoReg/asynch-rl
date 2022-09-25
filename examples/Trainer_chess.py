#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 12:05:11 2020

@author: Enrico Regolin
"""

#%%
# compiling time debug
import cProfile
import pstats
import io


#%%
import os, sys

from asynch_rl.rl.rl_env import Multiprocess_RL_Environment
from asynch_rl.rl.utilities import clear_pycache, store_train_params, load_train_params

import sys
import psutil
import time
import os
import ray

import numpy as np

#####
from argparse import ArgumentParser

parser = ArgumentParser()


#following params always to be declared
parser.add_argument("-rl", "--rl-mode", dest="rl_mode", type=str, default='AC', help="RL mode (AC, DQL, parallelAC)")

parser.add_argument("-i", "--iter", dest="n_iterations", type = int, default= 10 , help="number of training iterations")

parser.add_argument("-p", "--parallelize", dest="ray_parallelize", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False,
                    help="ray_parallelize bool")

parser.add_argument("-a", "--agents-number", dest="agents_number", type=int, default= 6, help="Number of agents to be used")

parser.add_argument("-norm", "--normalize-layers", dest="normalize_layers", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=True,
                    help="normalize data between NN layers")

parser.add_argument("-l", "--load-iteration", dest="load_iteration", type=int, default=0, help="start simulations and training from a given iteration")

parser.add_argument("-m", "--memory-size", dest="replay_memory_size", type=int, default= 10000, help="Replay Memory Size")

parser.add_argument("-v", "--net-version", dest="net_version", type=int, default=100, help="net version used")

parser.add_argument("-ha", "--head-address", dest="head_address", type=str, default= None, help="Ray Head Address")

parser.add_argument("-rp", "--ray-password", dest="ray_password", type=str, default= None, help="Ray password")

# following params can be left as default
parser.add_argument("-msl", "--memory-save-load", dest="memory_save_load", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False,
                    help="save memory bool (for debugging purpose)")

parser.add_argument("-tot", "--tot-iterations", dest="tot_iterations", type=int, default= 300,
                    help="Max n. iterations each agent runs during simulation")

parser.add_argument("-d","--difficulty", dest = "difficulty", type=int, default=0, help = "task degree of difficulty")

parser.add_argument("-mt", "--memory-turnover-ratio", dest="memory_turnover_ratio", type=float, default=.5,
                    help="Ratio of Memory renewed at each iteration")

parser.add_argument("-lr", "--learning-rate", dest="learning_rate",  nargs="*", type=float, default=[1e-3, 5e-3],
                    help="NN learning rate [QV, PG]. If parallelized, lr[QV] is ignored. If scalar, lr[QV] = lr[PG] = lr")

parser.add_argument(  "-e", "--epochs-training",  dest = "n_epochs", type=int, default=  200,  
                        help="Number of epochs (minimum N. if parallel) per QV training iteration" )

parser.add_argument("-sim", "--sim-length-max", dest="sim_length_max", type=int, default=600,
                    help="Length of one successful run in seconds")

parser.add_argument("-mb", "--minibatch-size", dest="minibatch_size",  type=int, default= 32,
                    help="Size of the minibatches used for training [QV, PG]")

parser.add_argument("-ym", "--epsilon-min", dest="epsilon_min", type=float, default=0.2, help="minimum epsilon")

parser.add_argument("-yd", "--epsilon-decay", dest="epsilon_decay", type=float, default=0.995,
                    help="annealing factor of epsilon")

parser.add_argument("-vf", "--validation-frequency", dest="val_frequency", type=int, default=10, help="model is validated every -vf iterations")

parser.add_argument("-ro", "--reset-optimizer", dest="reset_optimizer", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False,
                    help="reset optimizer")

parser.add_argument("-g", "--gamma", dest="gamma", type=float, default=0.95, help="GAMMA parameter in QV learning")

parser.add_argument("-b", "--beta", dest="beta", type=float, default= 0.001 , help="BETA parameter for entropy in PG learning")

parser.add_argument("-cadu", "--continuous-advantage-update", dest="continuous_qv_update", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, 
                    help="latest QV model is always used for Advanatge calculation")

parser.add_argument("-ur", "--use-reinforce", dest="use_reinforce", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False,
                    help="use REINFORCE instead of AC")

parser.add_argument("-dab", "--discrete-action-bins", dest="discrete_action_bins",  type=int, default= 10, 
                    help="discrete action bins (n_actions = dab+1)")

# env specific parameters

"""
parser.add_argument("-pu", "--pg-partial-update", dest="pg_partial_update", type=bool, default=False,
                    help="Flag to update only partially the PG model (non updated layers are shared with QV model)")
"""
parser.add_argument(
  "-ll", "--layers-list",  nargs="*",  # 0 or more values expected => creates a list
  dest = "layers_list", type=int, default=[100, 100, 100, 50],  # default if nothing is provided
)

parser.add_argument(
  "-rw", "--rewards",  nargs="*",  # 0 or more values expected => creates a list
  dest = "rewards_list", type=int, default=[200,50, 1, 1e3],  # default if nothing is provided
)

args = parser.parse_args()

#####

#####

num_cpus = psutil.cpu_count(logical=False)
n_agents = 2*num_cpus -2

def main(net_version = 0, n_iterations = 5, ray_parallelize = False, \
        load_iteration = -1, agents_number = n_agents, learning_rate= (0.001,0.001) , \
        n_epochs = 400 , replay_memory_size = 5000, epsilon_min = 0.2 ,  \
        epsilon_annealing_factor = 0.95,  mini_batch_size = 64 , \
        memory_turnover_ratio = 0.1, val_frequency = 10, layers_width= (100,100), reset_optimizer = False, rl_mode = 'AC', \
        gamma = 0.99, beta = 0.001 , difficulty = 0, sim_length_max = 100, \
        continuous_qv_update = False, tot_iterations = 400, rewards = [1,1,1,1], discrete_action_bins = 8, \
        ray_password = None,  head_address = None, memory_save_load = False, dynamic_grad_weighting = False):
    

    function_inputs = locals().copy()
    
    env_type = 'Chess' 
    model_type = 'LinearModel'
    overwrite_params = ['rewards', 'rl_mode', 'layers_width', 'agents_number',\
                            'val_frequency']

    # initialize required net and model parameters if loading from saved values
    if load_iteration != 0:
        my_dict = load_train_params(env_type, model_type, overwrite_params, net_version)
        for i,par in enumerate(overwrite_params):
            exec(par + " =  my_dict['"  + par + "']")
        del( overwrite_params, my_dict)

    
    # import ray
    if ray_parallelize:
        # Start Ray.
        
        replay_memory_size *= agents_number
        
        try:
            ray.shutdown()
        except Exception:
            print('ray not active')
            
        if ray_password is not None:
            ray.init(address=head_address, redis_password = ray_password )
        else:
            ray.init()
        

    env_options = {}
    
    #single_agent_min_iterations = round(memory_turnover_ratio*replay_memory_size / (agents_number * 20) )
    
    
    rl_env = Multiprocess_RL_Environment(env_type, model_type, net_version,rl_mode = rl_mode , n_agents = agents_number, \
                                         ray_parallelize=ray_parallelize, move_to_cuda=False, n_frames = 1, \
                                         replay_memory_size = replay_memory_size, mini_batch_size = mini_batch_size, \
                                         N_epochs = n_epochs, epsilon_min = epsilon_min , rewards = rewards, \
                                         epsilon_annealing_factor=epsilon_annealing_factor, discr_env_bins = discrete_action_bins , \
                                         difficulty = difficulty, learning_rate = learning_rate, sim_length_max = sim_length_max, 
                                         tot_iterations = tot_iterations, memory_turnover_ratio = memory_turnover_ratio, \
                                         gamma = gamma, beta_PG = beta , val_frequency = val_frequency, layers_width= layers_width,\
                                         continuous_qv_update = continuous_qv_update, memory_save_load = memory_save_load )

    #rl_env.movie_frequency = 2
    rl_env.save_movie = False
    rl_env.live_plot = False
    # always update agents params after rl_env params are changed
    rl_env.updateAgentsAttributesExcept('env')

    # launch env
    time_init = time.time()
    
    # second run is to test parallel computation fo simulations and NN update
    if load_iteration != 0:
        rl_env.load( load_iteration)
    else:
        store_train_params(rl_env, function_inputs)


    rl_env.runSequence(n_iterations, reset_optimizer=reset_optimizer) 
   


if __name__ == "__main__":
    #"""
    env = main(net_version = args.net_version, n_iterations = args.n_iterations, ray_parallelize= args.ray_parallelize, \
               load_iteration =args.load_iteration, \
                replay_memory_size = args.replay_memory_size, agents_number = args.agents_number, \
                n_epochs = args.n_epochs, epsilon_min = args.epsilon_min,  \
                epsilon_annealing_factor = args.epsilon_decay, rewards = args.rewards_list, \
                mini_batch_size = args.minibatch_size, learning_rate = args.learning_rate, \
                sim_length_max = args.sim_length_max, difficulty = args.difficulty, \
                memory_turnover_ratio = args.memory_turnover_ratio, val_frequency = args.val_frequency, \
                layers_width= args.layers_list, reset_optimizer = args.reset_optimizer, rl_mode = args.rl_mode, \
                gamma = args.gamma, beta = args.beta, continuous_qv_update = args.continuous_qv_update,\
                tot_iterations = args.tot_iterations, discrete_action_bins = args.discrete_action_bins, \
                head_address = args.head_address, ray_password = args.ray_password, memory_save_load = args.memory_save_load )
    
    current_folder = os.path.abspath(os.path.dirname(__file__))
    clear_pycache(current_folder)

