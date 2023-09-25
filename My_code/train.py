import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import numpy as np
import random
from environment import BombermanEnvironment, HEIGHT, WIDTH, TILE_SIZE
import pygame
import matplotlib.pyplot as plt

rewards_history = []
import os
import sys
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class QNetwork:
    def __init__(self, input_dim, output_dim, learning_rate=0.001):
        """ Neural network model definition """
        self.input = tf.keras.layers.Input(shape=(input_dim,))
        x = tf.keras.layers.Dense(128, activation='relu')(self.input)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        self.output = tf.keras.layers.Dense(output_dim, activation='linear')(x)
        self.model = tf.keras.Model(inputs=self.input, outputs=self.output)
        self.model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate), loss='mse')

class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, learning_rate=0.001):
        """ DQN agent initialization """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_network = QNetwork(state_dim, action_dim, learning_rate)
        self.memory = []

    def act(self, state):
        """ Epsilon-greedy action selection """
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        q_values = self.q_network.model.predict(state)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        """ Store experience in memory """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """ Experience replay logic """
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.q_network.model.predict(next_state)[0])
            target_f = self.q_network.model.predict(state)
            target_f[0][action] = target
            with HiddenPrints():
                self.q_network.model.train_on_batch(state, target_f)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
def live_plot(data, figsize=(7,5), title=''):
        """ Live update the training rewards plot """
        #clear_output(wait=True)
        plt.figure(figsize=figsize)
        plt.plot(data)
        plt.title(title)
        plt.grid(True)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.show();

# Game loop and training logic
EPISODES = 500
env = BombermanEnvironment()

# Change input dimension to match the flattened grid state
state_dim = (HEIGHT // TILE_SIZE) * (WIDTH // TILE_SIZE)
agent = DQNAgent(state_dim, 4)

for episode in range(EPISODES):
    state = env.reset().reshape(1, -1)
    done = False
    total_reward = 0

    while not done:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        action = agent.act(state)
        next_state, reward, done = env.step(action)
        next_state = next_state.reshape(1, -1)

        agent.remember(state, action, reward, next_state, done)
        agent.replay(32)

        state = next_state
        total_reward += reward

        env.render()
        pygame.time.wait(100)  # Delay for visualization

    print(f"Episode: {episode}/{EPISODES}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")
    rewards_history.append(total_reward)
    # Visualize rewards every 10 episodes
    if episode % 10 == 0:
        live_plot(np.array(rewards_history), title=f"Episode: {episode}/{EPISODES}")
# Save the trained model
agent.q_network.model.save('/Users/anureddy/Desktop/SEM02/Essential_ML/Bomberman_proj_draft/bomberman_rl/saved_model')
