import json
import os
import random
from collections import deque

import numpy as np

from game import Game


class ANN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Initialize weights and biases
        self.W1 = np.random.randn(state_size, 64) * 0.01
        self.b1 = np.zeros((1, 64))
        self.W2 = np.random.randn(64, action_size) * 0.01
        self.b2 = np.zeros((1, action_size))

    def forward(self, state):
        self.z1 = np.dot(state, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2

    def backward(self, d_output, learning_rate, states):
        m = d_output.shape[0]
        
        # Output layer
        dW2 = np.dot(self.a1.T, d_output) / m
        db2 = np.sum(d_output, axis=0, keepdims=True) / m
        
        # Hidden layer
        d_hidden = np.dot(d_output, self.W2.T)
        d_hidden[self.z1 <= 0] = 0  # ReLU derivative
        dW1 = np.dot(states.T, d_hidden) / m
        db1 = np.sum(d_hidden, axis=0, keepdims=True) / m
        
        # Update weights
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def save_weights(self, file_path):
        np.save(file_path, {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2
        })

    def load_weights(self, file_path):
        weights = np.load(file_path, allow_pickle=True).item()
        self.W1 = weights['W1']
        self.b1 = weights['b1']
        self.W2 = weights['W2']
        self.b2 = weights['b2']


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, k):
        experiences = random.sample(self.memory, k=k)
        states = np.vstack([e[0] for e in experiences if e is not None])
        actions = np.vstack([e[1] for e in experiences if e is not None])
        rewards = np.vstack([e[2] for e in experiences if e is not None])
        next_states = np.vstack([e[3] for e in experiences if e is not None])
        dones = np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)
        
        return states, actions, rewards, next_states, dones


# Hyperparameters
number_episodes = 100000
maximum_number_steps_per_episode = 200000
epsilon_starting_value = 1.0
epsilon_ending_value = 0.001
epsilon_decay_value = 0.99
learning_rate = 0.01
minibatch_size = 100
gamma = 0.95
replay_buffer_size = int(1e5)

state_size = 16
action_size = 4
scores_on_100_episodes = deque(maxlen=100)
folder = "model"


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.local_network = ANN(state_size, action_size)
        self.target_network = ANN(state_size, action_size)
        self.memory = ReplayMemory(replay_buffer_size)
        self.t_step = 0
        self.record = -1
        self.epsilon = -1

    def get_state(self, game):
        head_x = game.snake.x[0]
        head_y = game.snake.y[0]

        point_left = [(head_x - game.BLOCK_WIDTH), head_y]
        point_right = [(head_x + game.BLOCK_WIDTH), head_y]
        point_up = [head_x, (head_y - game.BLOCK_WIDTH)]
        point_down = [head_x, (head_y + game.BLOCK_WIDTH)]
        point_left_up = [(head_x - game.BLOCK_WIDTH), (head_y - game.BLOCK_WIDTH)]
        point_left_down = [(head_x - game.BLOCK_WIDTH), (head_y + game.BLOCK_WIDTH)]
        point_right_up = [(head_x + game.BLOCK_WIDTH), (head_y - game.BLOCK_WIDTH)]
        point_right_down = [(head_x + game.BLOCK_WIDTH), (head_y + game.BLOCK_WIDTH)]

        state = [
            game.is_danger(point_left),
            game.is_danger(point_right),
            game.is_danger(point_up),
            game.is_danger(point_down),
            game.is_danger(point_left_up),
            game.is_danger(point_left_down),
            game.is_danger(point_right_up),
            game.is_danger(point_right_down),
            game.snake.direction == "left",
            game.snake.direction == "right",
            game.snake.direction == "up",
            game.snake.direction == "down",
            game.apple.x < head_x,
            game.apple.x > head_x,
            game.apple.y < head_y,
            game.apple.y < head_y,
        ]

        return np.array(state, dtype=int)

    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0 and len(self.memory.memory) > minibatch_size:
            experiences = self.memory.sample(k=minibatch_size)
            self.learn(experiences)

    def get_action(self, state, epsilon):
        state = np.array(state).reshape(1, -1)
        action_values = self.local_network.forward(state)
        
        if random.random() > epsilon:
            move = np.argmax(action_values)
        else:
            move = random.randint(0, 3)
        
        return move

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        
        next_q_targets = np.max(self.target_network.forward(next_states), axis=1, keepdims=True)
        q_targets = rewards + gamma * next_q_targets * (1 - dones)
        
        q_expected = self.local_network.forward(states)
        loss = np.mean((q_expected[np.arange(len(actions)), actions.flatten()] - q_targets.flatten()) ** 2)
        
        # Simplified gradient update
        d_output = np.zeros_like(q_expected)
        d_output[np.arange(len(actions)), actions.flatten()] = q_expected[np.arange(len(actions)), actions.flatten()] - q_targets.flatten()
        
        self.local_network.backward(d_output, learning_rate, states)
        self.soft_update()

    def soft_update(self, tau=1e-2):
        self.target_network.W1 = tau * self.local_network.W1 + (1 - tau) * self.target_network.W1
        self.target_network.b1 = tau * self.local_network.b1 + (1 - tau) * self.target_network.b1
        self.target_network.W2 = tau * self.local_network.W2 + (1 - tau) * self.target_network.W2
        self.target_network.b2 = tau * self.local_network.b2 + (1 - tau) * self.target_network.b2

    def load(self, file_name='model.npy'):
        file_path = os.path.join(folder, file_name)
        if os.path.exists(file_path):
            self.local_network.load_weights(file_path)
            print("Model Loaded")
            self.retrieve_data()

    def save_model(self, file_name='model.npy'):
        if not os.path.exists(folder):
            os.mkdir(folder)
        file_path = os.path.join(folder, file_name)
        self.local_network.save_weights(file_path)

    def retrieve_data(self):
        file_name = "data.json"
        model_data_path = os.path.join(folder, file_name)
        if os.path.exists(model_data_path):
            with open(model_data_path, 'r') as file:
                data = json.load(file)
                if data:
                    self.record = data['record']
                    self.epsilon = data['epsilon']

    def save_data(self, record, epsilon):
        if not os.path.exists(folder):
            os.mkdir(folder)
        file_path = os.path.join(folder, "data.json")
        with open(file_path, 'w') as file:
            json.dump({'record': record, 'epsilon': epsilon}, file, indent=4)


if __name__ == "__main__":
    game = Game()
    agent = Agent(state_size=state_size, action_size=action_size)
    agent.load()
    max_score = 0

    epsilon = epsilon_starting_value

    if agent.epsilon != -1:
        epsilon = agent.epsilon
        max_score = max(agent.record, max_score)
    print('epsilon starts at {}', epsilon)
    for episode in range(0, number_episodes):
        game.reset()
        score = 0

        for t in range(maximum_number_steps_per_episode):
            state_old = agent.get_state(game)
            action = agent.get_action(state_old, epsilon)
            # perform the action
            move = [0, 0, 0, 0]
            move[action] = 1
            reward, done, score = game.run(move)
            state_new = agent.get_state(game)
            agent.step(state_old, action, reward, state_new, done)
            if done:
                break
        max_score = max(max_score, score)
        scores_on_100_episodes.append(score)
        epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
        agent.save_model()
        agent.save_data(max_score, epsilon)
        if episode % 50 == 0:
            print('Episode {}\t Curr Score {}\tMax Score {}\tAvg Score {:.2f}'.format(episode, score, max_score,
                                                                                      np.mean(scores_on_100_episodes)))