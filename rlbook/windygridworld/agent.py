import numpy as np

ALPHA = 0.001
GAMMA = 0.95


class WgwAgent:

    def __init__(self):
        self.qvalues = dict()
        for i in range(7):
            for j in range(10):
                actions = np.zeros(4)
                self.qvalues[(i, j)] = actions

    def action(self, state):
        return np.argmax(self.qvalues[state])

    '''
    Q(S,A) = Q(S,A) + alpha*[R+gamma*Q(S',A')-Q(S,A)]
    '''
    def update(self, state, action, state_next, reward):
        reward_next = np.max(self.qvalues[state_next])
        self.qvalues[state][action] *= (1 - ALPHA)
        self.qvalues[state][action] += ALPHA * (reward + GAMMA * reward_next)
