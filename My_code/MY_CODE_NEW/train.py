import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from env_manual import BombermanEnvironment

GAMMA = 0.95
LEARNING_RATE = 0.001
MEMORY_SIZE = 1000000
BATCH_SIZE = 20
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
EPISODES = 100
MAX_STEPS = 100


class DQNSolver(nn.Module):

    def __init__(self, observation_space, action_space):
        super(DQNSolver, self).__init__()
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.fc1 = nn.Linear(143, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_space)

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.forward(torch.Tensor(state))
        return torch.argmax(q_values[0]).item()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self, optimizer, criterion):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * torch.max(self(torch.Tensor(state_next))))  # No need for .item() here
            q_values = self(torch.Tensor(state))
            
            # Ensure that the action index is within a valid range
            if action < self.action_space:  # Check if action is valid
                q_values[action] = q_update  # No need for .item() here
            optimizer.zero_grad()
            outputs = self(torch.Tensor(state))
            loss = criterion(outputs, q_values)
            loss.backward()
            optimizer.step()
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)





def plot_rewards(episode_rewards):
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards Over Time')
    plt.show()
    plt.close()


def train():
    env = BombermanEnvironment(render_mode=False)
    observation_space = env.reset()  # Get initial state
    action_space = 5

    model = DQNSolver(143, action_space)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    episode_rewards = []

    SAVE_INTERVAL = 10  # Save the model every 10 episodes. Adjust this value as needed.

    for episode in range(1, EPISODES + 1):
        env.reset()
        state = torch.Tensor(observation_space)
        episode_reward = 0

        for step in range(MAX_STEPS):
            action = model.act(state)
            next_state, reward, done = env.step(action)
            episode_reward += reward

            state_next = torch.Tensor(next_state)
            model.remember(state, action, reward, state_next, done)
            state = state_next

            if done or step == MAX_STEPS - 1:
                print(f"Episode: {episode}/{EPISODES}, Reward: {episode_reward}")
                episode_rewards.append(episode_reward)
                break

            if len(model.memory) > BATCH_SIZE:
                model.experience_replay(optimizer, criterion)

        # Save the model based on SAVE_INTERVAL
        if episode % SAVE_INTERVAL == 0:
            model_save_path = '/Users/anureddy/Desktop/SEM02/Essential_ML/Bomberman_proj_draft/bomberman_rl/My_code/MY_CODE_NEW/outputs/trained_model_episode_{}.pt'.format(episode)
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved successfully as '{model_save_path}'!")

    plot_rewards(episode_rewards)

if __name__ == "__main__":
    train()



