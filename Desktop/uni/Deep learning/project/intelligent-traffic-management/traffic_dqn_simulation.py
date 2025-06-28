
import random
import numpy as np
import pygame
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
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

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > 2000:
            self.memory.pop(0)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state in minibatch:
            target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]), verbose=0)[0])
            target_f = self.model.predict(np.array([state]), verbose=0)
            target_f[0][action] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Simulation entities
class TrafficLight:
    def __init__(self, location):
        self.location = location
        self.state = "red"
        self.timer = 0

    def change_state(self, new_state):
        self.state = new_state
        self.timer = 0

    def update(self, action):
        self.timer += 1
        if self.state == "red" and action == 0:
            self.change_state("green")
        elif self.state == "green" and action == 1:
            self.change_state("yellow")
        elif self.state == "yellow" and self.timer >= 3:
            self.change_state("red")

    def draw(self):
        color = RED if self.state == "red" else GREEN if self.state == "green" else YELLOW
        pygame.draw.circle(screen, color, self.location, 10)

class Vehicle:
    def __init__(self, location, speed):
        self.location = list(location)
        self.speed = speed

    def update(self, traffic_light):
        if traffic_light.state == "red" and self.location[0] < traffic_light.location[0]:
            return
        self.location[0] += self.speed

    def draw(self):
        pygame.draw.rect(screen, WHITE, (*self.location, 10, 10))

# Main Loop
agent = DQNAgent(state_size=2, action_size=2)  # State: [vehicle_count, current_state], Actions: [0 = turn green, 1 = turn yellow]
traffic_light = TrafficLight((400, 300))
vehicles = [Vehicle((100, 290), 1), Vehicle((100, 310), 1)]

running = True
while running:
    screen.fill(BLACK)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # State: number of vehicles near the light, current light state as integer
    vehicle_count = sum(1 for v in vehicles if 380 <= v.location[0] <= 420)
    state = [vehicle_count, 0 if traffic_light.state == "red" else 1]
    action = agent.act(state)

    # Simulate traffic light and vehicle movement
    traffic_light.update(action)
    for v in vehicles:
        v.update(traffic_light)
        v.draw()
    traffic_light.draw()

    # Simulated reward: fewer vehicles near light is better
    next_vehicle_count = sum(1 for v in vehicles if 380 <= v.location[0] <= 420)
    reward = -next_vehicle_count
    next_state = [next_vehicle_count, 0 if traffic_light.state == "red" else 1]

    agent.remember(state, action, reward, next_state)
    agent.replay(16)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
