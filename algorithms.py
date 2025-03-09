import numpy as np
from grid_world import GridWorld
from tqdm import tqdm

# model-based method
def policy_evaluation(env:GridWorld, policy_matrix:np.ndarray, state_values:np.ndarray, gamma=0.9, theta=1e-5):
    P = env.get_model()
    while True:
        delta = 0
        for state in range(env.num_states):
            v = state_values[state]
            # P_s deterministic case
            state_values[state] = np.sum(
                policy_matrix[state][action] * (
                    P[state][action]["reward"] + gamma * state_values[P[state][action]["next_state"]]
                )
                for action in range(env.num_actions)
            )
            delta = max(delta, abs(v - state_values[state]))
        if delta < theta:
            break
    return state_values

def TD_table(
        env:GridWorld, 
        policy_matrix:np.ndarray, 
        true_state_values:np.ndarray,
        gamma=0.9, alpha=0.005, 
        n_episodes=500, n_steps=500
    ):
    P = env.get_model()
    weights = np.random.randn(env.num_states, 1)
    rmse_list = []
    pbar = tqdm(
        total=n_episodes*n_steps,
        desc=f"TD-Table: alpha={alpha}",
        dynamic_ncols=True
    )
    for episode in range(n_episodes):
        for _ in range(n_steps):
            # Sample a state index uniformly
            state_idx = np.random.randint(0, env.num_states)

            # Sample an action based on the policy
            action_idx = np.random.choice(env.num_actions, p=policy_matrix[state_idx])
            
            reward = P[state_idx][action_idx]["reward"]
            next_state_idx = P[state_idx][action_idx]["next_state"]

            # TD update
            td_error = reward + gamma * weights[next_state_idx] - weights[state_idx]
            weights[state_idx] += alpha * td_error
            pbar.update(1)


        # Calculate RMSE between estimated values and true values at the end of each episode
        rmse = np.sqrt(np.mean((weights - true_state_values)**2))
        rmse_list.append(rmse)
        pbar.set_postfix({"RMSE": rmse, "Episode": episode})
    pbar.close()

    return weights, rmse_list

def TD_linear(
        env:GridWorld, 
        policy_matrix:np.ndarray, 
        true_state_values:np.ndarray,
        basis:str = "poly", p=1,
        gamma=0.9, alpha=0.005, 
        n_episodes=500, n_steps=500
    ):
    P = env.get_model()

    if basis == "poly":
        from feature import pos2poly
        phi_func = lambda state: pos2poly(state, p)
        weights = np.random.randn((p+1)*(p+2)//2, 1)
    elif basis == "fourier":
        from feature import pos2fourier
        phi_func = lambda state: pos2fourier(state, p)
        weights = np.random.randn((p+1)*(p+2)//2, 1)
    elif basis == "fourierq":
        from feature import pos2fourierq
        phi_func = lambda state: pos2fourierq(state, p)
        weights = np.random.randn((p+1)**2, 1)
    else:
        raise NotImplementedError
    
    rmse_list = []
    pbar = tqdm(
        total=n_episodes*n_steps,
        desc=f"TD-Linear({basis}-{p}): alpha={alpha}",
        dynamic_ncols=True
    )
    for episode in range(n_episodes):
        for _ in range(n_steps):
            # Sample a state index uniformly
            state_idx = np.random.randint(0, env.num_states)
            state = env.state_space[state_idx]

            # Sample an action based on the policy
            action_idx = np.random.choice(env.num_actions, p=policy_matrix[state_idx])
            
            reward = P[state_idx][action_idx]["reward"]
            next_state_idx = P[state_idx][action_idx]["next_state"]
            next_state = env.state_space[next_state_idx]

            # TD update
            td_error = reward + gamma * phi_func(next_state) @ weights - phi_func(next_state) @ weights
            weights += alpha * td_error * phi_func(state).T

            pbar.update(1)
        

        state_np = np.array(env.state_space)
        # Calculate RMSE between estimated values and true values at the end of each episode
        rmse = np.sqrt(np.mean(((phi_func(state_np) @ weights).squeeze() - true_state_values)**2))
        rmse_list.append(rmse)
        pbar.set_postfix({"RMSE": rmse, "Episode": episode})
    pbar.close() 

    return phi_func, weights, rmse_list

    