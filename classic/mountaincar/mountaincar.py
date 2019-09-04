import gym
import numpy as np

from classic.mountaincar.agent import DQNAgent, SarsaAgent
# Compute the reward as the energy of the car,
# which consists of the potential energy and kinetic energy
# The goal is to obtain more energy
from common import TerminateChecker


def mountaincar():
    env = gym.make("MountainCar-v0")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    # agent = SarsaAgent(observation_space, action_space)
    agent = SarsaAgent()
    checker = TerminateChecker(-110)
    run = 0
    while not checker.success():
        run += 1
        state = env.reset()
        step = 0
        while True:
            step += 1
            action = agent.action(state)
            state_next, reward, terminal, info = env.step(action)
            agent.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("Run: " + str(run) + ", reward: " + str(step))
                checker.record(-step)
                break
            agent.train()


if __name__ == "__main__":
    mountaincar()
