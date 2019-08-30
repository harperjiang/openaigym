import numpy as np
import gym
from mountaincar.agent import MountainCarAgent

# Compute the reward as the energy of the car,
# which consists of the potential energy and kinetic energy
# The goal is to obtain more energy
def custom_reward(state):
    data = state[0]
    current_pos = data[0]
    current_v = data[1]
    height = abs(current_pos+0.5)

    return 100*current_v*current_v + height

def mountaincar():
    env = gym.make("MountainCar-v0")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    agent = MountainCarAgent(observation_space, action_space)
    run = 0
    for ite in range(1000):
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            env.render()
            action = agent.action(state)
            state_next, useless, terminal, info = env.step(action)
            reward = custom_reward(state)
            state_next = np.reshape(state_next, [1, observation_space])
            agent.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(agent.exploration_rate) + ", reward: " + str(reward))
                break
            agent.train()


if __name__ == "__main__":
    mountaincar()