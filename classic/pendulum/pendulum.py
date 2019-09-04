import numpy as np
import gym
from classic.pendulum.agent import PendulumAgent

def pendulum():
    env = gym.make("Pendulum-v0")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space
    agent = PendulumAgent(observation_space, action_space)
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
            state_next, reward, terminal, info = env.step(action)
            state_next = np.reshape(state_next, [1, observation_space])
            agent.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(agent.exploration_rate) + ", reward: " + str(reward))
                break
            agent.train()


if __name__ == "__main__":
    pendulum()