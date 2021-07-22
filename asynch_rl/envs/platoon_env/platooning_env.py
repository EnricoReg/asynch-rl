# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:29:34 2020

@author: Enrico Regolin
"""

#%%

from .platooning_energy import Car
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

#%%

class PlatoonEnv(gym.Env):
    
    def __init__(self, sim_length_max = 200, difficulty = 1, rewards = [100, 1, 20, 10], options = {False}):
        # cartpole params
        
        self.env_type = 'Platoon'
        
        self.change_gears = False
        if 'change_gears' in options:
            self.change_gears = options['change_gears']
        
        #crash/lost track, prize, energy, control
        
        self.rewards = rewards
        
        #self.weight = [0.1, 2, .25]  # tracking error, vertical angle, changes in control action
        self.weight = [0.05, 1, .1]  # tracking error, vertical angle, changes in control action
        
        self.sim_length_max = sim_length_max # max simulation length (in seconds)
        self.dt = 0.2
        self.time_too_far_max = 5
        
        self.difficulty = difficulty
        if self.difficulty == 0: 
            self.ref_distance = 50
            self.max_tracking_error = 40
            self.max_acceleration = 7
        elif self.difficulty == 1:
            self.ref_distance = 40
            self.max_tracking_error = 30
            self.max_acceleration = 8
        elif self.difficulty == 2:
            self.ref_distance = 37
            self.max_tracking_error = 27
            self.max_acceleration = 8.5
        elif self.difficulty == 3:
            self.ref_distance = 20
            self.max_tracking_error = 15
            self.max_acceleration = 10
        
        self.leader = Car(initial_speed = 30, max_acceleration = self.max_acceleration)
        self.follower = Car(initial_speed = 30, max_acceleration = self.max_acceleration, change_gears =self.change_gears)
        
        self.max_power = self.leader.e_motor.getMaxPower()

        
        self.max_vel = self.leader._max_velocity
        self.max_state = np.array([self.max_vel,self.max_vel,  self.ref_distance + 1.2*self.max_tracking_error])

        if self.change_gears:
            self.action_space = spaces.Box(low=np.array([-1,0, 0]), high=np.array([1,1, 1]))
            self.n_actions = 3
        else:
            self.action_space = spaces.Box(low=np.array([-1,0]), high=np.array([1,1])) #, shape=(size__,))
            self.n_actions = 2
            
        self.observation_space = spaces.Box(low= np.array([0,0,0]), high=self.max_state) # , shape=(len(state_max),))

        # initialize state
        self.state = np.zeros(3) # in gym-like env state has to include target
        self.duration = None


    #####################################################################################################            
    def get_max_iterations(self):
        return int(round(self.sim_length_max / self.dt))


    #####################################################################################################            
    def get_actions_structure(self):
        pass # implemented in wrapper

        
    #####################################################################################################            
    def get_observations_structure(self):
        self.reset()
        return len(self.step()[0])


    #####################################################################################################            
    # this function returns a traditional control input to be used instead of the ML one
    def get_controller_input(self, discretized = False, bins = [10,1]):
        
        error = self.ref_distance - self.state[0]
        self.cum_error += error
        
        vel_error = self.state[2]-self.state[1]
        
        ctrl_law = + ( .15*error +.001* self.cum_error + 1.5*vel_error )
        
        a = 0.8
        norm_e_torque = (1-a)*ctrl_law + a*self.previous_def_e_tq
        
        norm_e_torque = np.clip(-norm_e_torque, -1 , 1)
        
        norm_br_torque = 0
        if error > 15 and vel_error > 0:
            norm_br_torque = 1
            norm_e_torque = -1
        elif error > 12 and vel_error > 0:
            norm_br_torque = (error - 12) / 3
            norm_e_torque = -1
            
        if discretized:
            ctrl_steps = np.linspace(-1,1,bins[0]+1)
            norm_e_torque = ctrl_steps[ np.abs(ctrl_steps - norm_e_torque).argmin()]
           
        if self.change_gears:
            #print(self.follower.current_gear== 0, self.follower.velocity> 24)
            if self.follower.current_gear == 0 and self.follower.velocity > 24:
                gear_n = 1
            elif self.follower.current_gear == 1 and self.follower.velocity < 23:
                gear_n = 0
            else:
                gear_n = self.follower.current_gear            

        self.previous_def_e_tq = norm_e_torque
        
        if self.change_gears:
            return np.array([norm_e_torque, norm_br_torque, gear_n])
        else:
            return np.array([norm_e_torque, norm_br_torque])


    #####################################################################################################            
    def get_complete_sequence(self):
        if hasattr(self, 'stored_states_sequence'):
            return self.stored_states_sequence, self.stored_ctrl_sequence
        else:
            return None


    #####################################################################################################            
    def no_action(self):
        if self.change_gears:
            return np.array([0,0,0])
        else:
            return np.array([0,0])


    #####################################################################################################            
    def step(self,action = None, *args):
        
        if action is None:
            action = self.no_action()
        
        if self.duration == 0:
            reset_run = True
            leader_position = self.state[0]
            follower_position = 0
            cum_energy = 0
        else:
            reset_run = False
            
        att_input = self.generate_attacker_move()
        
        self.leader.update(self.dt, att_input[0], att_input[1], 0)
        
        if self.change_gears:
            self.follower.update(self.dt, action[0], action[1], gear_n=action[2])
        else:
            self.follower.update(self.dt, action[0], action[1], 0)
        
        self.duration += self.dt
        
        dist_z1 = self.state[0]
        vel_att_z1 = self.state[1]
        
        new_dist = self.state[0]+ (self.state[1]-self.state[2])*self.dt
        self.state = np.array([new_dist, self.leader.velocity, self.follower.velocity])
        
        if self.states_sequence is not None:
            leader_position = self.states_sequence[-1,3]+ self.dt*self.state[1]
            follower_position = self.states_sequence[-1,4]+ self.dt*self.state[2] 
            cum_energy = self.ctrl_sequence[-1,5] + self.follower.e_power*self.dt
        else:
            follower_position = self.dt*self.state[2]
            leader_position = follower_position +  self.state[0]
            cum_energy = self.follower.e_power*self.dt
            
        extended_state = np.append(self.state.T, np.array([leader_position, follower_position, self.follower.current_gear]))[np.newaxis,:] #, axis = 1)
        extended_ctrl = np.append(action[np.newaxis,:2], \
                                  np.array([[self.follower.e_torque*self.follower.gear_ratio[self.follower.current_gear], -self.follower.br_torque,\
                                             self.follower.e_power, cum_energy, \
                                             self.follower.e_motor_speed, self.follower.e_torque]], dtype = np.float) , axis = 1)
        
        if reset_run:
            self.states_sequence = extended_state
            self.ctrl_sequence = extended_ctrl
        else:
            self.states_sequence = np.append(self.states_sequence , extended_state, axis = 0)
            self.ctrl_sequence = np.append(self.ctrl_sequence , extended_ctrl, axis = 0)
        
        
        # self.rewards
        #[0] crash/lost track, [1] prize, [2] energy, [3] control
        norm_tracking_error = (self.ref_distance - self.state[0])/self.max_tracking_error
        
        info = {'outcome':None}

        # too close  --> end simulation
        if  self.state[0] < (self.ref_distance - self.max_tracking_error)  :
            done = True
            reward = -self.rewards[0]
            info['outcome'] = 'fail'
        # too far, penalize
        elif  self.state[0] > self.ref_distance + round(1.5*self.max_tracking_error):
            done = True
            reward = -self.rewards[0]
            info['outcome'] = 'fail'
        elif self.duration >= self.sim_length_max:
            done = True
            energy_reward = self.rewards[2] * (1-cum_energy/(self.states_sequence.shape[0]*self.dt*self.max_power))
            ctrl_reward = self.rewards[3] *( 1 - np.sum(np.diff(self.ctrl_sequence[:,0])**2)/(2*self.states_sequence.shape[0])  )
            reward = (energy_reward + ctrl_reward).item()
            info['outcome'] = 'finalized'
            
            """
            print(f'done! final reward = {reward}')
            print(f'done! energy_reward = {energy_reward}')
            print(f'done! ctrl_reward = {ctrl_reward}')
            """
        else:
            done = False
            """
            #reward = 1 - (np.abs(self.state[2]-self.state[1])/ (self.state[1] + 0.01 ) )**(1/2)
            if np.abs(self.state[0] - self.ref_distance) < 15:
                relative_v_err = np.abs(self.state[2]-self.state[1])/ (self.state[1] + 1e-8 ) 
                reward = self.rewards[1] *(0.7 - (np.clip(relative_v_err, -3,3))**(1/4))
            elif self.state[0] > self.ref_distance:
                reward =  self.rewards[1] *0.05*((self.state[2] - self.state[1])**3) *np.abs(norm_tracking_error)
            elif self.state[0] < self.ref_distance:
                reward =  self.rewards[1]*((self.state[1] - self.state[2])**3)*np.abs(norm_tracking_error)
            reward = np.clip(reward, -2, 0)
            """

                        
            reward = -self.rewards[1] *(np.clip(np.abs(self.state[2]-self.state[1])/ (self.state[1] + 0.01 ), 0,1 ))**(3)

        self.duration += self.dt
        
        
        
        
        if done:
            self.stored_states_sequence = self.states_sequence
            self.stored_ctrl_sequence = self.ctrl_sequence
            
            #abs_energy_reward =  (1-cum_energy/(self.states_sequence.shape[0]*self.dt*self.max_power))
            #abs_ctrl_reward =  ( 1 - np.sum(np.diff(self.ctrl_sequence[:,0])**2)/(2*self.states_sequence.shape[0])  )
            
            #print(f'abs_energy_reward = {abs_energy_reward}')
            #print(f'abs_ctrl_reward   = {abs_ctrl_reward}')

        
        #norm_dist_z1    = (self.ref_distance - dist_z1 )/ self.max_tracking_error
        #norm_vel_att_z1 = vel_att_z1/self.max_vel
        
        #state_norm = np.array([norm_dist_z1, (self.ref_distance - self.state[0]) /self.max_tracking_error, norm_vel_att_z1, self.state[1]/self.max_vel, self.state[2]/self.max_vel ], dtype = np.float)
        state_norm = np.array([  norm_tracking_error, \
                                 self.state[1]/self.max_vel, \
                                 self.state[2]/self.max_vel ], \
                                 dtype = np.float)
        
        return state_norm, reward, done, info


    #####################################################################################################
    def generate_attacker_move(self):
        
        if self.state[0] < 20 and self.state[1]<0.8*self.state[2] and not self.state[1] < 8 :
            norm_e_torque = -1
            norm_br_torque = 1
        elif self.state[1]>1.5*self.state[2] or self.state[1] < 5 or (self.state[0] > 35 and self.state[1]>self.state[2]):
            norm_e_torque = 1
            norm_br_torque = 0
        else:
            norm_e_torque_delta = 0.25*(2*np.random.random()-1)
            norm_br_torque_delta = 0.25*(2*np.random.random()-1)
            norm_e_torque = np.clip(self.previous_att_e_tq + norm_e_torque_delta, -1, 1)
            norm_br_torque = np.clip(self.previous_att_br_tq + norm_br_torque_delta, 0, 1)
            
        self.previous_att_e_tq = norm_e_torque
        self.previous_att_br_tq = norm_br_torque
        
        return np.array([norm_e_torque, norm_br_torque])

        
    #####################################################################################################
    def reset(self, save_history = False, **kwargs):
                
        self.duration = 0
        initial_distance =  self.ref_distance + (2*np.random.random()-1)*0.8*self.max_tracking_error
        a = 0.2
        # initial speeds uniformally distributed between 20% and 80% of the max possible speed
        leader_initial_speed = a*self.max_vel + (1-2*a)*self.max_vel*np.random.random()
        follower_initial_speed = a*self.max_vel + (1-2*a)*self.max_vel*np.random.random()
        
        self.state = np.array([ initial_distance, leader_initial_speed, follower_initial_speed], dtype = np.float)

        self.leader = Car(initial_speed = self.state[1], max_acceleration = self.max_acceleration)
        self.follower = Car(initial_speed = self.state[2], max_acceleration = self.max_acceleration, change_gears = self.change_gears)

        self.states_sequence = None
        self.ctrl_sequence = None

        self.previous_att_e_tq = 0
        self.previous_att_br_tq = 0
        
        self.previous_def_e_tq = 0

        self.cum_error = 0 # for traditional control only
        self.consecutive_too_far = 0


    #####################################################################################################
    def plot_graphs(self, save_figs = False):
        time_line = np.arange(0.0, self.duration, self.dt)
        
        # fig 1
        fig1 = plt.figure()
        ax_0 = fig1.add_subplot(4,1,1)
        ax_1 = fig1.add_subplot(4,1,2)
        ax_2 = fig1.add_subplot(4,1,3)
        ax_3 = fig1.add_subplot(4,1,4)
        
        dist_err = self.ref_distance - self.stored_states_sequence[:,0]
        journey_length = self.stored_states_sequence.shape[0]
        
        ax_0.plot(dist_err)
        ax_0.plot(0*np.ones((journey_length,1)), 'k',linewidth=0.5)
        ax_0.plot(self.max_tracking_error*np.ones((journey_length,1)), 'r')
        #ax_0.plot(-20*np.ones((journey_length,1)))
        ax_0.legend(['distancing error','reference','crash safety line' ])
        
        ax_1.plot(self.states_sequence[:,3])
        ax_1.plot(self.states_sequence[:,4])
        ax_1.legend(['leader pos','car position'])
        
        ax_2.plot(self.states_sequence[:,1])
        ax_2.plot(self.states_sequence[:,2])
        ax_2.legend(['leader vel','car vel'])

        ax_3.plot(self.states_sequence[:,5])
        ax_3.legend(['inserted gear'])
        
        plt.show()
 
        # fig 2
        fig2 = plt.figure()
        ax0 = fig2.add_subplot(4,1,1)
        ax1 = fig2.add_subplot(4,1,2)
        ax2 = fig2.add_subplot(4,1,3)
        ax3 = fig2.add_subplot(4,1,4)
       
        ax0.plot(self.stored_ctrl_sequence[:,4])
        ax0.legend(['power'])
        
        ax1.plot(self.stored_ctrl_sequence[:,5])
        ax1.legend(['cum energy'])
        
        ax2.plot(self.stored_ctrl_sequence[:,0])
        ax2.plot(self.stored_ctrl_sequence[:,1])
        ax2.legend(['norm e-tq','norm br tq'])
       
        ax3.plot(self.stored_ctrl_sequence[:,2])
        ax3.plot(self.stored_ctrl_sequence[:,3])
        ax3.legend(['e-tq','br tq'])

        plt.show()
        
        if save_figs:
            fig1.savefig('state_signals.png')
            fig2.savefig('ctrl_signals.png')

        # fig 3
        self.follower.e_motor.plotEffMap( scatter_array = self.stored_ctrl_sequence[:,(6,7)])


    #####################################################################################################    
    # currently copy/paste from robot-env
    def render(self,mode = 'human',iter_params = (0,0,0), action = 0):
        pass

        


#%%

if __name__ == "__main__":
    env = PlatoonEnv(sim_length_max = 200) # , change_gears= False)
    done = False
    env.reset()
    cum_reward = 0
    t = 0 
    
    while not done:
        
        state, reward, done, info = env.step(env.get_controller_input(discretized=True, bins=[10,1]))
        cum_reward += reward
        if np.abs(reward)> 0.2:
            print(f't = {t}, reward = {np.round(reward,2)}')
        t += 1
        
    print(f'cum reward =  {cum_reward}')
    
    env.plot_graphs()
        
    
    
#%%
"""
import matplotlib.pyplot as plt
import numpy as np

v1 = v2= np.arange(5,40)
xx, yy = np.meshgrid(v1, v2)

reward = 10*(0.6 - (np.abs(xx-yy)/ (xx + 0.0001 ) )**(1/4))

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

ax1 = plt.contour(v1,v2, np.abs(xx-yy),levels = [2, 5, 10, 20] ,colors='black')
ax1 = plt.contourf(v1, v2, reward.T,levels = 30 ,cmap = 'jet')

cbar =plt.colorbar(ax1)
cbar.ax.locator_params(nbins=5)
"""
    
    
    
    
    
    
    
    
    