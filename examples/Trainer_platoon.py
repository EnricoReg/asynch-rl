#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 13:51:07 2020

@author: Enrico Regolin
"""

#%%

# for time debug only
import cProfile
import pstats
import io
# for time debug only

import os, sys

from asynch_rl.rl.rl_env import Multiprocess_RL_Environment
from asynch_rl.rl.utilities import clear_pycache, store_train_params, load_train_params

import psutil
import time
import ray

import numpy as np

#####
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("-rl", "--rl-mode", dest="rl_mode", type=str, default='AC', help="RL mode (AC, DQL, parallelAC)")

parser.add_argument("-i", "--iter", dest="n_iterations", type = int, default= 10 , help="number of training iterations")

parser.add_argument("-p", "--parallelize", dest="ray_parallelize", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False,
                    help="ray_parallelize bool")

parser.add_argument("-a", "--agents-number", dest="agents_number", type=int, default= 5, help="Number of agents to be used")

parser.add_argument("-norm", "--normalize-layers", dest="normalize_layers", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=True,
                    help="normalize data between NN layers")

parser.add_argument("-mo", "--map-output", dest="map_output", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=True,
                    help="NN has intermediate output with estimated map")

parser.add_argument("-l", "--load-iteration", dest="load_iteration", type=int, default=0, help="start simulations and training from a given iteration")

parser.add_argument("-m", "--memory-size", dest="replay_memory_size", type=int, default= 4000, help="Replay Memory Size")

parser.add_argument("-v", "--net-version", dest="net_version", type=int, default=100, help="net version used")

parser.add_argument("-ha", "--head-address", dest="head_address", type=str, default= None, help="Ray Head Address")

parser.add_argument("-rp", "--ray-password", dest="ray_password", type=str, default= None, help="Ray password")


#####
parser.add_argument("-msl", "--memory-save-load", dest="memory_save_load", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False,
                    help="save memory bool (for debugging purpose)")

parser.add_argument("-tot", "--tot-iterations", dest="tot_iterations", type=int, default= 500,
                    help="Max n. iterations each agent runs during simulation. Influences the level of exploration which is reached by PG algorithm")

parser.add_argument("-d","--difficulty", dest = "difficulty", type=int, default= 2, help = "task degree of difficulty. 10 = random")

parser.add_argument("-sim", "--sim-length-max", dest="sim_length_max", type=int, default=150,
                    help="Length of one successful run in seconds")

parser.add_argument("-mt", "--memory-turnover-ratio", dest="memory_turnover_ratio", type=float, default=.25,
                    help="Ratio of Memory renewed at each iteration")

parser.add_argument("-lr", "--learning-rate", dest="learning_rate", nargs="*", type=float, default=[1e-4, 2e-3],
                    help="NN learning rate for DQL [0] and A/C [1]")

parser.add_argument("-e", "--epochs-training", dest="n_epochs", type=int, default= 200 , help="Number of epochs per training iteration")

parser.add_argument("-mb", "--minibatch-size", dest="minibatch_size",  type=int, default= 256,
                    help="Size of the minibatches used for QV training")

parser.add_argument("-ym", "--epsilon-min", dest="epsilon_min", type=float, default=0.2, help="minimum epsilon")

parser.add_argument("-yd", "--epsilon-decay", dest="epsilon_decay", type=float, default=0.995,
                    help="annealing factor of epsilon")

parser.add_argument("-vf", "--validation-frequency", dest="val_frequency", type=int, default=5, help="model is validated every -vf iterations")

parser.add_argument("-ro", "--reset-optimizer", dest="reset_optimizer", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False,
                    help="reset optimizer")

parser.add_argument("-fr", "--frames-number", dest="n_frames", type=int, default= 6, help="number of frames considered for convolutional network")

parser.add_argument("-scl", "--share-conv-layers", dest="share_conv_layers", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False,
                    help="Flag to share Convolutional Layers between Actor and Critic")

parser.add_argument("-g", "--gamma", dest="gamma", type=float, default=0.9, help="GAMMA parameter in QV learning")

parser.add_argument("-b", "--beta", dest="beta", type=float, default= 0.05 , help="BETA parameter for entropy in PG learning")

parser.add_argument("-cadu", "--continuous-advantage-update", dest="continuous_qv_update", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, 
                    help="latest QV model is always used for Advanatge calculation")

parser.add_argument("-ur", "--use-reinforce", dest="use_reinforce", type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False,
                    help="use REINFORCE instead of AC")



parser.add_argument( "-ll", "--layers-list",  nargs="*", dest = "layers_list", type=int, default=[40, 40, 20] )

parser.add_argument("-gc", "--change-gears", dest="change_gears", type=bool, default=False,
                    help="change gears")

parser.add_argument(
  "-rw", "--rewards",  nargs="*",  # 0 or more values expected => creates a list
  dest = "rewards_list", type=int, default = [100, 1, 20, 10],  # default if nothing is provided
)

parser.add_argument(
  "-dab", "--discrete-action-bins",  nargs="*",  # 0 or more values expected => creates a list
  dest = "discr_act_bins", type=int, default=[20,1],  # default if nothing is provided
)

args = parser.parse_args()

#####

num_cpus = psutil.cpu_count(logical=False)
n_agents = 2*num_cpus -2

def main(net_version = 0, n_iterations = 2, ray_parallelize = False,  difficulty = 0,\
         load_iteration = -1, agents_number = n_agents, learning_rate= 0.001,\
             n_epochs = 400, replay_memory_size = 5000, ctrlr_probability = 0, sim_length_max = 100, \
        epsilon_annealing_factor = 0.95,  ctrlr_prob_annealing_factor = 0.9 , mini_batch_size = 64, \
            memory_turnover_ratio = 0.1, val_frequency = 10, rewards = np.ones(4), reset_optimizer = False,
            share_conv_layers = False, n_frames = 4, rl_mode = 'DQL', beta = 0.001, epsilon_min = 0.2, \
                gamma = 0.99,  continuous_qv_update = False, tot_iterations = 400, layers_width= (100,100), \
                    ray_password = None,  head_address = None, memory_save_load = False, \
                        use_reinforce = False, normalize_layers = False, change_gears = False, discr_act_bins = 5):


    function_inputs = locals().copy()
    
    env_type = 'Platoon' 
    model_type = 'LinearModel'

    overwrite_params = ['rewards', 'rl_mode', 'share_conv_layers', 'n_frames' ,\
                        'layers_width', 'normalize_layers', 'agents_number',\
                            'val_frequency','discr_act_bins', 'change_gears']
        
    env_options = {'change_gears' : change_gears}

        
    # initialize required net and model parameters if loading from saved values
    if load_iteration != 0:

        storage_path = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))) ,"Data" , \
                                env_type, model_type+str(net_version) )

        if os.path.isfile(os.path.join(storage_path,'train_params.txt')):
            my_dict = load_train_params(env_type, model_type, overwrite_params, net_version)
            for i,par in enumerate(overwrite_params):
                exec(par + " =  my_dict['"  + par + "']")
            del( overwrite_params, my_dict)

    # import ray
    if ray_parallelize:
        # Start Ray.
        
        replay_memory_size *= agents_number
        #replay_memory_size *= 25 # (to ensure same epoch length between DQL on cluster and AC on eracle )
        
        try:
            ray.shutdown()
        except Exception:
            print('ray not active')
            
        if ray_password is not None:
            ray.init(address=head_address, redis_password = ray_password )
        else:
            ray.init()


    # launch env
    time_init = time.time()

    pr = cProfile.Profile()
    pr.enable()

    
    # second run is to test parallel computation fo simulations and NN update
        
    if load_iteration > 0:
        print('###################################################################')
        print(f'Loading Environment. iteration: {load_iteration}')
        print('###################################################################')

    rl_env = Multiprocess_RL_Environment(env_type , model_type , net_version , rl_mode = rl_mode, \
                        ray_parallelize=ray_parallelize, move_to_cuda=True, n_frames = n_frames, \
                        replay_memory_size = replay_memory_size, n_agents = agents_number,\
                        tot_iterations = tot_iterations, discr_env_bins = discr_act_bins , \
                        use_reinforce = use_reinforce,  epsilon_annealing_factor=epsilon_annealing_factor,  layers_width= layers_width,\
                        N_epochs = n_epochs, epsilon_min = epsilon_min , rewards = rewards, \
                        mini_batch_size = mini_batch_size, share_conv_layers = share_conv_layers, \
                        difficulty = difficulty, learning_rate = learning_rate, sim_length_max = sim_length_max, \
                        memory_turnover_ratio = memory_turnover_ratio, val_frequency = val_frequency ,\
                        gamma = gamma, beta_PG = beta , continuous_qv_update = continuous_qv_update , \
                        memory_save_load = memory_save_load , normalize_layers = normalize_layers, \
                            ) 

    # always update agents params after rl_env params are changed
    rl_env.updateAgentsAttributesExcept('env')


    if load_iteration != 0:
        rl_env.load( load_iteration)
        
    else:
        store_train_params(rl_env, function_inputs)
            
    rl_env.runSequence(n_iterations, reset_optimizer=reset_optimizer) 

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats()
    with open('duration_test.txt', 'w+') as f:
        f.write(s.getvalue())

    return rl_env


################################################################

if __name__ == "__main__":
    
    env = main(net_version = args.net_version, n_iterations = args.n_iterations, ray_parallelize= args.ray_parallelize, \
               load_iteration =args.load_iteration, replay_memory_size = args.replay_memory_size, \
               agents_number = args.agents_number, memory_turnover_ratio = args.memory_turnover_ratio, \
               n_epochs = args.n_epochs, epsilon_annealing_factor = args.epsilon_decay, \
               mini_batch_size = args.minibatch_size,  learning_rate = args.learning_rate, difficulty = args.difficulty, \
               sim_length_max = args.sim_length_max, val_frequency = args.val_frequency, share_conv_layers = args.share_conv_layers, \
               rewards = args.rewards_list, reset_optimizer = args.reset_optimizer, rl_mode = args.rl_mode, \
               beta = args.beta, gamma = args.gamma, continuous_qv_update = args.continuous_qv_update,\
               tot_iterations = args.tot_iterations, head_address = args.head_address, ray_password = args.ray_password ,\
               memory_save_load = args.memory_save_load, layers_width= args.layers_list, normalize_layers = args.normalize_layers, \
               use_reinforce = args.use_reinforce,   n_frames = args.n_frames, epsilon_min = args.epsilon_min , \
                   change_gears = args.change_gears, discr_act_bins = args.discr_act_bins)

    current_folder = os.path.abspath(os.path.dirname(__file__))
    clear_pycache(current_folder)

