import numpy as np
from grid_world import GridWorld

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