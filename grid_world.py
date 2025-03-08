import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import *


class GridWorld:
    def __init__(
            self, 
            env_size, 
            start_state, 
            target_state, 
            forbidden_states, 
            action_space, 
            reward_target, 
            reward_forbidden, 
            reward_step, 
            animation_interval
        ):
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
        # elif new_state == self.target_state:  # stay
        #     x, y = self.target_state
        #     reward = self.reward_target
        # elif new_state in self.forbidden_states:  # stay
        #     x, y = state
        #     reward = self.reward_forbidden        
        else: # do not stay
            x, y = new_state
            if new_state == self.target_state:
                reward = self.reward_target
            elif new_state in self.forbidden_states:
                reward = self.reward_forbidden
            else:
                reward = self.reward_step
            
        return (x, y), reward
        

    def _is_done(self, state):
        return state == self.target_state
    
    def draw_grid(self, ax:plt.Axes=None):        
        # setup grid
        ax.set_xlim(-0.5, self.env_size[0] - 0.5)
        ax.set_ylim(-0.5, self.env_size[1] - 0.5)
        ax.xaxis.set_ticks(np.arange(-0.5, self.env_size[0], 1))
        ax.yaxis.set_ticks(np.arange(-0.5, self.env_size[1], 1))
        ax.grid(True, linestyle="-", color="gray", linewidth="1", axis='both')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.xaxis.set_ticks_position('top')
        
        # add index labels
        idx_labels_x = [i for i in range(self.env_size[0])]
        idx_labels_y = [i for i in range(self.env_size[1])]
        for lb in idx_labels_x:
            ax.text(lb, -0.75, str(lb+1), size=10, ha='center', va='center', color='black')
        for lb in idx_labels_y:
            ax.text(-0.75, lb, str(lb+1), size=10, ha='center', va='center', color='black')
        
        ax.tick_params(bottom=False, left=False, right=False, top=False, 
                           labelbottom=False, labelleft=False, labeltop=False)

        # draw target and forbidden states
        ax.add_patch(state2rect(self.target_state, COLOR_TARGET))
        for forbidden_state in self.forbidden_states:
            ax.add_patch(state2rect(forbidden_state, COLOR_FORBID))

    def render(self, animation_interval=None):
        animation_interval = animation_interval or self.animation_interval
        
        plt.clf()
        if self.canvas is None:
            plt.ion()
            self.canvas = plt.figure("Grid World")
        ax = self.canvas.gca()
        self.draw_grid(ax)

        # Draw trajectory and agent
        if hasattr(self, 'traj'):
            traj_x, traj_y = zip(*self.traj)
            ax.plot(traj_x, traj_y, color=COLOR_TRAJECTORY, linewidth=0.5)
        
        # Draw agent (store the artist)
        ax.plot(self.agent_state[0], self.agent_state[1], 
                marker='d', color=COLOR_AGENT, markersize=10)

        plt.draw()
        plt.pause(animation_interval)


    def show_policy_and_values(self, policy_matrix=None, state_values=None, precision=1):
        if self.canvas is None:
            self.canvas = plt.figure('Policy and Values')
        ax = self.canvas.gca()
        self.draw_grid(ax)
        
        # Draw policy arrows
        if policy_matrix is not None:
            for state, state_action_group in enumerate(policy_matrix):
                x = state % self.env_size[0]
                y = state // self.env_size[0]
                for i, action_probability in enumerate(state_action_group):
                    dx, dy = self.action_space[i]
                    if (dx, dy) != (0,0):
                        ax.add_patch(
                            patches.FancyArrow(
                                x, y, 
                                dx=(0.1+action_probability/2)*dx, 
                                dy=(0.1+action_probability/2)*dy, 
                                color=COLOR_POLICY, 
                                width=0.001, head_width=0.05
                            )
                        )
                    else:
                        ax.add_patch(
                            patches.Circle(
                                (x, y), 
                                radius=0.05, 
                                facecolor=COLOR_POLICY, 
                                edgecolor=COLOR_POLICY, 
                                linewidth=1, 
                                fill=False
                            )
                        )
        
        # Draw state values if provided
        if state_values is not None:
            state_values = np.round(state_values, precision)
            for i, value in enumerate(state_values):
                x = i % self.env_size[0]
                y = i // self.env_size[0]
                ax.text(
                    x, y, 
                    str(value), 
                    ha='center', va='center', 
                    fontsize=10, 
                    color='black'
                )

        # plt.show()

    def show_values_3d(self, state_values):
        fig = plt.figure('State Values 3D')
        ax = fig.add_subplot(111, projection='3d')
        x = np.arange(self.env_size[0])
        y = np.arange(self.env_size[1])
        X, Y = np.meshgrid(x, y)
        Z = state_values.reshape(self.env_size)
        
        # Using viridis colormap which transitions from blue to yellow
        surf = ax.plot_surface(Y, X, Z, cmap='viridis', edgecolor='none')
        
        # Add a color bar to show the value range
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        ax.set_xlabel('row')
        ax.set_ylabel('column')
        ax.set_zlabel('Value')

        plt.title('3D State Values')
        plt.show()