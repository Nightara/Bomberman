import os
import numpy as np
import random
from environment import BombermanEnvironment, HEIGHT, WIDTH, TILE_SIZE
import pygame
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim



def ensure_directory_exists(path):
    """Ensure that the given directory path exists."""
    if not os.path.exists(path):
        os.makedirs(path)



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
        #print(f"Shape of x in QNetwork: {x.shape}")  
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon_min = epsilon_min #The minimum value to which epsilon can decay.
        self.q_network = QNetwork(state_dim, action_dim) #Initializes the Q-Network (a neural network) using the provided state and action dimensions. This Q-Network is responsible for estimating the Q-values for given states.
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss()
        self.memory = [] #he agent will store its experiences (state, action, reward, next_state, done) in this memory, and sample from it to train the Q-Network.
        self.q_values_history = [] # To store the Q-values(confidence scores) as 2D numpy array 
        
        
    def act(self, state):
        ''' the act method decides on an action based on the current state: 
        it sometimes chooses a random action (to explore the environment) and sometimes 
        chooses the action that its Q-network believes will yield the highest future rewards.'''
        #print(f"Shape of state at the start of act: {state.shape}")
        if np.random.rand() <= self.epsilon: #This is part of the epsilon-greedy policy, which is a way to balance exploration (trying out random actions) and exploitation (choosing actions based on what the agent has learned) in reinforcement learning.
            return np.random.choice(self.action_dim) #With a probability of self.epsilon, the agent selects a random action. This is the exploration part. As training progresses, self.epsilon is decayed (multiplied by self.epsilon_decay), so the agent gradually shifts from exploring to exploiting.
        if state.shape != (1, self.state_dim):
            print("Incorrect state shape!")
            return np.random.choice(self.action_dim)
        state = torch.tensor(state, dtype=torch.float32)#If the agent didn't select a random action, then it needs to process the state to decide on the best action.
        #print(f"Shape of state after squeeze: {state.shape}")
        q_values = self.q_network(state) #The state tensor is fed into the agent's Q-network (a neural network) to get the predicted Q-values for each action. Q-values essentially represent the agent's estimate of the future rewards for taking each action in the current state.
        self.q_values_history.append(q_values.detach().numpy()) 
        return torch.argmax(q_values).item() #the agent chooses the action corresponding to the maximum Q-value.( single integer representing the chosen action.)

    def remember(self, state, action, reward, next_state, done): 
        '''the remember method is used to store transitions the agent encounters so that they can be used later for training the Q-network.'''
        self.memory.append((state, action, reward, next_state, done))
        '''In the context of Deep Q-Learning, this "memory" is often called the "replay buffer". The replay buffer is a collection of past experiences that the agent can sample from when it updates its Q-network. 
        This technique, called "experience replay", helps stabilize the learning process.'''



    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            # Convert numpy arrays to PyTorch tensors and send to the device
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            done = torch.FloatTensor([done]).to(self.device)

            # Ensure state and next_state have the same shape
            if state.shape != next_state.shape:
                print(f"Shape mismatch: state: {state.shape}, next_state: {next_state.shape}")
                continue

            # Compute the target Q-value
            with torch.no_grad():
                target = reward + self.gamma * torch.max(self.q_network(next_state)) * (1 - done)

            # Compute the predicted Q-value
            predicted_all = self.q_network(state)
            predicted = predicted_all[0][action] 
            target = target.view_as(predicted)

            # Compute loss and perform backpropagation
            loss = self.loss_function(predicted, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def live_plot(data, figsize=(7,5), title=''):
    """ Save the training rewards plot to an image """
    plt.figure(figsize=figsize)
    plt.plot(data)
    plt.title(title)
    plt.grid(True)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Ensure the directory for saving plots exists
    plot_dir = './plots'
    ensure_directory_exists(plot_dir)
    
    plt.savefig(os.path.join(plot_dir, f"rewards_plot_episode_{len(data)}.png"))
    
def save_model(episode):
    """Function to save the model."""
    model_save_path = f'/Users/anureddy/Desktop/SEM02/Essential_ML/Bomberman_proj_draft/bomberman_rl/saved_model_ep_{episode}'
    torch.save(agent.q_network.state_dict(), model_save_path)
    print(f"\nModel saved at episode {episode} to {model_save_path}")


# Game loop and training logic
# Game loop and training logic
EPISODES = 100
env = BombermanEnvironment()
MAX_STEPS = 200
state_dim = (HEIGHT // TILE_SIZE) * (WIDTH // TILE_SIZE) + 1  # "+1" for the turn-encoded value
agent = CoinCollectorAgent(state_dim, 4)

for episode in range(EPISODES):
    print(f"Running Episode: {episode + 1}/{EPISODES}", end='\r')  # Display the current episode
    
    state = env.reset().reshape(1, -1)
    #print(f"Shape of state before reshape: {state.shape}") 
    done = False
    total_reward = 0
    turn_count = 0

    while not done and turn_count < MAX_STEPS:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Encode the current turn
        turn_encoded = 0.9 ** turn_count
        extended_state = np.append(state, turn_encoded).reshape(1, -1)
        #print(f"Shape of extended_state before feeding to network: {extended_state.shape}") 
        action = agent.act(extended_state)
        next_state, reward, done = env.step(action)
        next_state = next_state.reshape(1, -1)
        #print(f"Shape of next_state before reshape: {next_state.shape}") 
        # Append the turn_encoded to the next_state
        next_extended_state = np.append(next_state, turn_encoded).reshape(1, -1)
        #next_extended_state_tensor = torch.tensor(next_extended_state, dtype=torch.float32)
        #print(f"Shape of next_extended_state before feeding to network: {next_extended_state.shape}")
        agent.remember(extended_state, action, reward, next_extended_state, done)
        agent.replay(32)

        state = next_state
        total_reward += reward

        env.render()
        pygame.time.wait(100)  # Delay for visualization
        
        turn_count += 1
        #print("Shape of extended_state in replay:", extended_state.shape)
        #print("Shape of next_extended_state in replay:", next_extended_state.shape)

    print(f"Episode: {episode + 1}/{EPISODES}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")
    rewards_history.append(total_reward)
    
    # Save the model every 50 episodes
    if (episode + 1) % 100 == 0:
        save_model(episode + 1)

    # Visualize rewards every 10 episodes
    if (episode + 1) % 50 == 0:
        live_plot(np.array(rewards_history), title=f"Episode: {episode + 1}/{EPISODES}")
np.save('q_values_history.npy', np.array(agent.q_values_history))
print(f"\nTraining completed for {EPISODES} episodes.")

'''NOTE:
    1. the agent's "confidence" in choosing an action in a particular state is determined by the Q-values predicted by the Q-Network. The action with the highest Q-value is considered the best action in that state based on the agent's current knowledge.
    2.The agent's exploration rate (epsilon) decays over time, making the agent rely more on the Q-values and less on random exploration as it gains experience. 
    3. To load the confidence scores use 
    q_values_history = np.load('q_values_history.npy') Example: The agent encounters the first state. The Q-values predicted by the network for this state are [5.2, 6.3, 5.1, 6.0], corresponding to the four actions (up, down, left, right).'''
