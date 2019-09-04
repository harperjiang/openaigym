import random

import numpy as np
from collections import deque

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

GAMMA = 0.5
LEARNING_RATE = 0.001

MEMORY_SIZE = 30
BATCH_SIZE = 30

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.001
EXPLORATION_DECAY = 0.995


class DQNAgent:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(50, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(100, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        states = []
        results = []
        for state, action, reward, state_next, terminal in self.memory:
            q_update = reward
            q_values = self.model.predict(state)
            if not terminal:
                q_update = (reward + GAMMA * (
                        np.max(self.model.predict(state_next)[0]) - q_values[0][action]))
            q_values[0][action] += q_update
            states.append(state[0])
            results.append(q_values[0])
        self.model.fit(np.array(states), np.array(results), epochs=10, verbose=0)
        self.memory.clear()
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


from tiles3 import IHT, tiles

NUM_ACTION = 3
TILES = 2048
NUM_TILINGS = 8
ALPHA = 0.1
GAMMA = 0.9


class SarsaAgent:

    def __init__(self):
        self.weights = np.random.uniform(-0.001, 0.001, TILES)
        self.iht = IHT(TILES)
        self.q = dict()
        self.features = dict()

    '''
    Update weight vectors
    '''

    def remember(self, state, action, reward, next_state, done):
        features = self.generateFeatures(state, action)
        qvalue = self.qvalue(state, action)
        if done:
            self.weights[features] += ALPHA * (reward - qvalue)
        else:
            next_qvalue = np.max([self.qvalue(next_state, act) for act in range(NUM_ACTION)])
            self.weights[features] += ALPHA * (reward + GAMMA * next_qvalue - qvalue)

    def action(self, state):
        return np.argmax(np.array([self.qvalue(state, act) for act in range(NUM_ACTION)]))

    def qvalue(self, state, action):
        features = self.generateFeatures(state, action)
        featureKey = tuple(features)
        return self.weights[features].sum()

    def generateFeatures(self, observation, action):
        positionScale = NUM_TILINGS / (0.5 + 1.2)
        velocityScale = NUM_TILINGS / (0.07 + 0.07)
        features = tiles(self.iht, NUM_TILINGS, [observation[0] * positionScale, observation[1] * velocityScale],
                         [action])
        return np.array(features)

    def train(self):
        pass
