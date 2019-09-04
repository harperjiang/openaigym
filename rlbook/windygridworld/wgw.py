from rlbook.windygridworld.env import Environment
from rlbook.windygridworld.agent import WgwAgent

env = Environment()
agent = WgwAgent()

while True:
    state = env.reset()
    steps = 0
    while True:
        steps += 1
        action = agent.action(state)
        state_next, reward, terminated = env.step(action)
        if terminated:
            print("Terminated with step: " + str(steps))
            break
        agent.update(state, action, state_next, reward)
        state = state_next
