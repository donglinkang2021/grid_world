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
    # policy_matrix = np.random.rand(env.num_states, len(env.action_space))
    policy_matrix = np.ones((env.num_states, len(env.action_space))) / len(env.action_space)
    policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]
    state_values=np.random.uniform(0,10,(env.num_states,))
    
    env.show_policy_and_values(
        policy_matrix=policy_matrix,
        # state_values=state_values
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
    curve_dict = {
        "data1d_list": [],
        "label_list": []
    }

    alpha_list = [0.001, 0.005, 0.01]
    for alpha in alpha_list:
        estimated_state_values, rmse_list = TD_table(env, policy_matrix, true_state_values, alpha=alpha)
        label = rf"TD-table: $\alpha$={alpha}"
        draw_matrix2d_smooth(estimated_state_values.reshape(env.env_size), title=f"Estimated State Values(TD-table_alpha={alpha})")
        curve_dict["data1d_list"].append(rmse_list)
        curve_dict["label_list"].append(label)

    basis_list = ["poly", "fourier", "fourierq"]
    p_list = [1,2,3]
    alpha_list = [0.0005, 0.001, 0.005]
    for alpha in alpha_list:
        for basis in basis_list:
            for p in p_list:
                phi_func, weights, rmse_list = TD_linear(env, policy_matrix, true_state_values, basis=basis, p=p, alpha=alpha)
                label = rf"TD-Linear({basis}-{p}): $\alpha$={alpha}"
                draw_prediction(phi_func, weights, m=env.env_size[0], n=env.env_size[1], title=f"Estimated State Values(TD-Linear_{basis}-{p}_alpha={alpha})")
                curve_dict["data1d_list"].append(rmse_list)
                curve_dict["label_list"].append(label)

    draw_curve(**curve_dict, title="Comparisom of Different Basis Functions RMSE vs Episodes")
    # plt.show()

    import json
    with open("result_curve.json", "w") as f:
        json.dump(curve_dict, f)

def plot_curve_demo():
    import json
    with open("result_curve.json", "r") as f:
        curve_dict = json.load(f)

    import pandas as pd
    # turn into dataframe

    df = pd.DataFrame(curve_dict)
    print(df.head())

    model_list = ["TD-table"]
    for basis in ["poly", "fourier", "fourierq"]:
        for p in [1,2,3]:
            model_list.append(f"TD-Linear({basis}-{p})")

    query = df.label_list.str.contains("TD-table")
    draw_curve(
        data1d_list=df[query].data1d_list.tolist(),
        label_list=df[query].label_list.tolist(),
        title=f"Comparisom of Different Step-size RMSE vs Episodes(TD-table)",
    )

    query1 = df.label_list.str.contains("poly-1")
    query2 = df.label_list.str.contains("poly-2")
    query = query1 | query2
    print(query)
    query[22] = False
    query[13] = False
    draw_curve(
        data1d_list=df[query].data1d_list.tolist(),
        label_list=df[query].label_list.tolist(),
        title=f"Comparisom of Different Step-size RMSE vs Episodes(poly)",
    )

    query = df.label_list.str.contains("fourier-")
    draw_curve(
        data1d_list=df[query].data1d_list.tolist(),
        label_list=df[query].label_list.tolist(),
        title=f"Comparisom of Different Step-size RMSE vs Episodes(fourier)",
    )

    query = df.label_list.str.contains("fourierq")
    draw_curve(
        data1d_list=df[query].data1d_list.tolist(),
        label_list=df[query].label_list.tolist(),
        title=f"Comparisom of Different Step-size RMSE vs Episodes(fourierq)",
    )
    # plt.show()

    # plt.show()


# Example usage:
if __name__ == "__main__":
    from arguments import args        
    env = GridWorld(**vars(args))
    # random_policy_demo(env, num_steps=100)
    show_policy_and_values_demo(env)
    # PE_demo(env)
    # TD_demo(env)
    # plot_curve_demo()