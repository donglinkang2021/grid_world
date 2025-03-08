import numpy as np
from grid_world import GridWorld

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
        gamma=0.9, alpha=0.005, 
        n_episodes=500, n_steps=500
    ):
    P = env.get_model()
    weights = np.random.randn(env.num_states, 1)
    rmse_list = []
    for _ in range(n_episodes):
        step_rmse_list = []
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

            step_rmse_list.append(td_error)

        # Calculate RMSE between estimated values and true values at the end of each episode
        rmse = np.sqrt(np.mean(np.array(step_rmse_list)**2))
        rmse_list.append(rmse)

    return weights, rmse_list

def TD_linear(
        env:GridWorld, 
        policy_matrix:np.ndarray, 
        gamma=0.9, alpha=0.005, 
        n_episodes=500, n_steps=500
    ):
    P = env.get_model()

    def phi_func(state):
        x, y = state
        return np.array([[x, y, 1]]).T # (3, 1)

    weights = np.random.randn(3, 1)
    rmse_list = []
    for _ in range(n_episodes):
        step_rmse_list = []
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
            td_error = reward + gamma * phi_func(next_state).T @ weights - phi_func(next_state).T @ weights
            weights += alpha * td_error * phi_func(state)

            step_rmse_list.append(td_error)

        # Calculate RMSE between estimated values and true values at the end of each episode
        rmse = np.sqrt(np.mean(np.array(step_rmse_list)**2))
        rmse_list.append(rmse)

    return weights, rmse_list

    