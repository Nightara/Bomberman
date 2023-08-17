import os
import numpy as np
import random
from environment import BombermanEnvironment, HEIGHT, WIDTH, TILE_SIZE
import pygame
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

rewards_history = []

class QNetwork(nn.Module): #basic feed-forward neural network with 3 layers 
    '''The QNetwork is the tool that evaluates how good each of the moves (up, down, left, right) is. Maybe moving up leads to a dead-end, so the QNetwork would give that action a low value.
    Meanwhile, moving right could lead to a safe spot, so the QNetwork would assign a high value to that action.
    
    The neural network is used to approximate the Q-values for a given state and possible actions. 
     This is the feed-forward process: given a state, the network produces Q-values for each action.'''
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class CoinCollectorAgent: # main logic behind the Deep Q-Learning agent.
    def __init__(self, state_dim, action_dim, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, learning_rate=0.001): 
        self.state_dim = state_dim #state_dim seems to refer to the number of tiles in the Bomberman game's grid. The game's grid is divided into tiles, and the state might represent the status of each tile (whether it's empty, contains a wall, has a bomb, etc.).
        self.action_dim = action_dim #represents the number of possible actions the agent can take in a given state
        self.gamma = gamma
        self.epsilon = epsilon #It represents the likelihood of the agent taking a random action.
        self.epsilon_decay = epsilon_decay #The factor by which epsilon is reduced after each episode.
        self.epsilon_min = epsilon_min #The minimum value to which epsilon can decay.
        self.q_network = QNetwork(state_dim, action_dim) #Initializes the Q-Network (a neural network) using the provided state and action dimensions. This Q-Network is responsible for estimating the Q-values for given states.
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.memory = [] #he agent will store its experiences (state, action, reward, next_state, done) in this memory, and sample from it to train the Q-Network.

    def act(self, state):
        ''' the act method decides on an action based on the current state: 
        it sometimes chooses a random action (to explore the environment) and sometimes 
        chooses the action that its Q-network believes will yield the highest future rewards.'''
        if np.random.rand() <= self.epsilon: #This is part of the epsilon-greedy policy, which is a way to balance exploration (trying out random actions) and exploitation (choosing actions based on what the agent has learned) in reinforcement learning.
            return np.random.choice(self.action_dim)
        state = torch.tensor(state, dtype=torch.float32) #If the agent didn't select a random action, then it needs to process the state to decide on the best action.
        q_values = self.q_network(state) #The state tensor is fed into the agent's Q-network (a neural network) to get the predicted Q-values for each action. Q-values essentially represent the agent's estimate of the future rewards for taking each action in the current state.
        return torch.argmax(q_values).item() #the agent chooses the action corresponding to the maximum Q-value.( single integer representing the chosen action.)

    def remember(self, state, action, reward, next_state, done): 
        '''the remember method is used to store transitions the agent encounters so that they can be used later for training the Q-network.'''
        self.memory.append((state, action, reward, next_state, done))
        '''In the context of Deep Q-Learning, this "memory" is often called the "replay buffer". The replay buffer is a collection of past experiences that the agent can sample from when it updates its Q-network. 
        This technique, called "experience replay", helps stabilize the learning process.'''

    def replay(self, batch_size):
        '''the replay method implements the training logic of the Q-network using experiences from the replay buffer. 
        It updates the Q-values based on the Q-learning rule, computes the loss with the current Q-network, 
        and then updates the Q-network parameters using backpropagation. 
        The agent's exploration rate is also decayed over time'''
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32) # these states will be processed by NN 
            next_state = torch.tensor(next_state, dtype=torch.float32) #these states will be processed by NN 
            target = torch.tensor(reward, dtype=torch.float32) #Initialize the target Q-value as the reward from the experience.
            if not done:
                target = reward + self.gamma * torch.max(self.q_network(next_state)[0]) #This is the Q-learning update rule.
            current_q_value = self.q_network(state)[0, action] # Q-value predicted by NN for a given acion 
            loss = self.criterion(current_q_value, target) #
            self.optimizer.zero_grad() #Zero out any gradients from previous iterations. This ensures that the gradients are fresh for the current update.
            loss.backward() #This line computes the gradients using backpropagation based on the loss between predicted and target Q-values.
            self.optimizer.step() #This line updates the weights of the network using the previously computed gradients.
        if self.epsilon > self.epsilon_min: #Update the Q-network parameters in the direction that reduces the loss.
            self.epsilon *= self.epsilon_decay
            #After the update, if the current exploration rate (epsilon) is greater than the minimum allowed value (epsilon_min), 
            # it's decayed by a factor (epsilon_decay). This reduces the probability of random actions over time,
            # making the agent more exploitative.

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
state_dim = (HEIGHT // TILE_SIZE) * (WIDTH // TILE_SIZE) # global vision 
agent = CoinCollectorAgent(state_dim, 4)
coins_collected_per_turn_history = []  # To store our new metric for each episode

for episode in range(EPISODES):
    state = env.reset().reshape(1, -1)
    done = False
    total_reward = 0
    turn_count=0

    while not done:
        # Handle pygame events
        turn_count+=1
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
    
    coins_collected = total_reward/10
    coins_collected_per_turn = coins_collected / turn_count if turn_count > 0 else 0
    coins_collected_per_turn_history.append(coins_collected_per_turn)

    print(f"Episode: {episode}/{EPISODES}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")
    rewards_history.append(total_reward)
    # Visualize rewards every 10 episodes
    if episode % 10 == 0:
        live_plot(np.array(rewards_history), title=f"Episode: {episode}/{EPISODES}")
# Save the trained model
torch.save(agent.q_network.state_dict(), '/Users/anureddy/Desktop/SEM02/Essential_ML/Bomberman_proj_draft/bomberman_rl/saved_model')

