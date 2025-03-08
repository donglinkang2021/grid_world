from grid_world import GridWorld
import random
import numpy as np

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
    
    env.show_policy_and_values(
        policy_matrix=policy_matrix,
        # state_values=np.random.uniform(0,10,(env.num_states,))
    )

# Example usage:
if __name__ == "__main__":
    from arguments import args        
    env = GridWorld(**vars(args))
    random_policy_demo(env, num_steps=100)
    # show_policy_and_values_demo(env)
