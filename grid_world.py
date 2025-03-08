import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import *


class GridWorld:
    def __init__(self, env_size, start_state, target_state, forbidden_states, action_space, reward_target, reward_forbidden, reward_step, animation_interval):
        self.env_size = env_size
        self.num_states = env_size[0] * env_size[1]
        self.start_state = start_state
        self.target_state = target_state
        self.forbidden_states = forbidden_states
        self.agent_state = start_state
        self.action_space = action_space          
        self.reward_target = reward_target
        self.reward_forbidden = reward_forbidden
        self.reward_step = reward_step
        self.animation_interval = animation_interval

        self.canvas = None

    def reset(self):
        self.agent_state = self.start_state
        self.traj = [self.agent_state] 
        return self.agent_state, {}


    def step(self, action):
        assert action in self.action_space, "Invalid action"

        next_state, reward  = self._get_next_state_and_reward(self.agent_state, action)
        done = self._is_done(next_state)

        x_store = next_state[0] + 0.03 * np.random.randn()
        y_store = next_state[1] + 0.03 * np.random.randn()
        state_store = tuple(np.array((x_store,  y_store)) + 0.2 * np.array(action))
        state_store_2 = (next_state[0], next_state[1])

        self.agent_state = next_state

        self.traj.append(state_store)   
        self.traj.append(state_store_2)
        return self.agent_state, reward, done, {}   
    
        
    def _get_next_state_and_reward(self, state, action):
        x, y = state
        new_state = tuple(np.array(state) + np.array(action))
        if y + 1 > self.env_size[1] - 1 and action == (0,1):    # down
            y = self.env_size[1] - 1
            reward = self.reward_forbidden  
        elif x + 1 > self.env_size[0] - 1 and action == (1,0):  # right
            x = self.env_size[0] - 1
            reward = self.reward_forbidden  
        elif y - 1 < 0 and action == (0,-1):   # up
            y = 0
            reward = self.reward_forbidden  
        elif x - 1 < 0 and action == (-1, 0):  # left
            x = 0
            reward = self.reward_forbidden 
        elif new_state == self.target_state:  # stay
            x, y = self.target_state
            reward = self.reward_target
        elif new_state in self.forbidden_states:  # stay
            x, y = state
            reward = self.reward_forbidden        
        else:
            x, y = new_state
            reward = self.reward_step
            
        return (x, y), reward
        

    def _is_done(self, state):
        return state == self.target_state
    

    def render(self):
        if self.canvas is None:
            plt.ion()                             
            self.canvas, self.ax = plt.subplots()   
            self.ax.set_xlim(-0.5, self.env_size[0] - 0.5)
            self.ax.set_ylim(-0.5, self.env_size[1] - 0.5)
            self.ax.xaxis.set_ticks(np.arange(-0.5, self.env_size[0], 1))     
            self.ax.yaxis.set_ticks(np.arange(-0.5, self.env_size[1], 1))     
            self.ax.grid(True, linestyle="-", color="gray", linewidth="1", axis='both')          
            self.ax.set_aspect('equal')
            self.ax.invert_yaxis()                           
            self.ax.xaxis.set_ticks_position('top')           
            
            idx_labels_x = [i for i in range(self.env_size[0])]
            idx_labels_y = [i for i in range(self.env_size[1])]
            for lb in idx_labels_x:
                self.ax.text(
                    lb, -0.75, 
                    str(lb+1), 
                    size=10, 
                    ha='center', va='center', 
                    color='black'
                )           
            for lb in idx_labels_y:
                self.ax.text(
                    -0.75, lb, 
                    str(lb+1), 
                    size=10, 
                    ha='center', va='center', 
                    color='black'
                )
            self.ax.tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False, labelleft=False, labeltop=False)   

            self.target_rect = state2rect(self.target_state, COLOR_TARGET)
            self.ax.add_patch(self.target_rect)     

            for forbidden_state in self.forbidden_states:
                self.ax.add_patch(state2rect(forbidden_state, COLOR_FORBID))

            self.agent_star, = self.ax.plot([], [], marker = '*', color=COLOR_AGENT, markersize=20, linewidth=0.5) 
            self.traj_obj, = self.ax.plot([], [], color=COLOR_TRAJECTORY, linewidth=0.5)

        # self.agent_circle.center = (self.agent_state[0], self.agent_state[1])
        self.agent_star.set_data([self.agent_state[0]],[self.agent_state[1]])       
        traj_x, traj_y = zip(*self.traj)         
        self.traj_obj.set_data(traj_x, traj_y)

        plt.draw()
        plt.pause(self.animation_interval)

    def add_policy(self, policy_matrix):                  
        for state, state_action_group in enumerate(policy_matrix):    
            x = state % self.env_size[0]
            y = state // self.env_size[0]
            for i, action_probability in enumerate(state_action_group):
                if action_probability !=0:
                    dx, dy = self.action_space[i]
                    if (dx, dy) != (0,0):
                        self.ax.add_patch(
                            patches.FancyArrow(
                                x, y, 
                                dx=(0.1+action_probability/2)*dx, 
                                dy=(0.1+action_probability/2)*dy, 
                                color=COLOR_POLICY, 
                                width=0.001, head_width=0.05
                            )
                        )
                    else:
                        self.ax.add_patch(
                            patches.Circle(
                                (x, y), 
                                radius=0.07, 
                                facecolor=COLOR_POLICY, 
                                edgecolor=COLOR_POLICY, 
                                linewidth=1, 
                                fill=False
                            )
                        )
    
    def add_state_values(self, values, precision=1):
        values = np.round(values, precision)
        for i, value in enumerate(values):
            x = i % self.env_size[0]
            y = i // self.env_size[0]
            self.ax.text(
                x, y, 
                str(value), 
                ha='center', va='center', 
                fontsize=10, 
                color='black'
            )