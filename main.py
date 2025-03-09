from grid_world import GridWorld
import random
import numpy as np
import matplotlib.pyplot as plt
from utils import draw_matrix2d_smooth, draw_curve

def random_policy_demo(env:GridWorld, num_steps=1000):
    env.reset()
    for t in range(num_steps):
        env.render()
        action = random.choice(env.action_space)
        next_state, reward, done, info = env.step(action)
        print(f"Step: {t}, Action: {action}, State: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")
        if done:
            break
    env.render(animation_interval=2)

def show_policy_and_values_demo(env:GridWorld):
    policy_matrix = np.random.rand(env.num_states, len(env.action_space))
    policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]
    state_values=np.random.uniform(0,10,(env.num_states,))
    
    env.show_policy_and_values(
        policy_matrix=policy_matrix,
        state_values=state_values
    )
    env.show_values_3d(
        state_values=state_values
    )
    plt.show()

def PE_demo(env:GridWorld):
    from algorithms import policy_evaluation
    policy_matrix = np.ones((env.num_states, len(env.action_space))) / len(env.action_space)
    policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]
    state_values=np.random.uniform(0,10,(env.num_states,))
    state_values = policy_evaluation(env, policy_matrix, state_values)
    env.show_policy_and_values(state_values=state_values)
    draw_matrix2d_smooth(state_values.reshape(env.env_size), title="True State Values")
    draw_matrix2d_smooth(state_values.reshape(env.env_size), title="True State Values(k=3)", k=3)
    plt.show()

def TD_demo(env:GridWorld):
    from algorithms import TD_table, TD_linear, policy_evaluation
    from utils import draw_curve, draw_prediction
    policy_matrix = np.ones((env.num_states, len(env.action_space))) / len(env.action_space)
    policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]
    state_values0 = np.random.uniform(0,10,(env.num_states,))
    true_state_values = policy_evaluation(env, policy_matrix, state_values0)
    estimated_state_values1, rmse_list1 = TD_table(env, policy_matrix, true_state_values, alpha=0.005)
    weights, rmse_list2 = TD_linear(env, policy_matrix, true_state_values, alpha=0.0005)
    draw_curve([rmse_list1, rmse_list2],[r"TD-table: $\alpha$=0.005", r"TD-linear: $\alpha$=0.0005"])
    draw_matrix2d_smooth(estimated_state_values1.reshape(env.env_size), title="Estimated State Values(TD-Table)")
    draw_prediction(weights, m=env.env_size[0], n=env.env_size[1], title="Estimated State Values(TD-Linear)")
    plt.show()


# Example usage:
if __name__ == "__main__":
    from arguments import args        
    env = GridWorld(**vars(args))
    # random_policy_demo(env, num_steps=100)
    # show_policy_and_values_demo(env)
    PE_demo(env)
    # TD_demo(env)