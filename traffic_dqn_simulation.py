
# traffic_dqn_simulation.py

# Deep Q-Network (DQN) based Traffic Signal Control System
# This code simulates traffic management using Reinforcement Learning

import pygame
import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Simulation settings
WIDTH, HEIGHT = 800, 600
ROAD_WIDTH = 80
CAR_WIDTH = 40
CAR_HEIGHT = 60

# Colors
WHITE = (255, 255, 255)
GREY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]), verbose=0)[0])
            target_f = self.model.predict(np.array([state]), verbose=0)
            target_f[0][action] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Placeholder classes and functions for full simulation
class TrafficEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.car_count = [0, 0]  # North-South, East-West
        self.state = [0, 0]
        return self.state

    def step(self, action):
        self.car_count[action] += random.randint(1, 5)
        reward = -sum(self.car_count)
        next_state = self.car_count
        done = False
        return next_state, reward, done

# Training loop
if __name__ == "__main__":
    env = TrafficEnv()
    state_size = 2
    action_size = 2
    agent = DQNAgent(state_size, action_size)
    episodes = 100

    for e in range(episodes):
        state = env.reset()
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        agent.replay(32)
    print("Training completed!")
