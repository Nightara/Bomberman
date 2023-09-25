import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from dummy_manual import BombermanEnvironment

GAMMA = 0.95
LEARNING_RATE = 0.001
MEMORY_SIZE = 1000000
BATCH_SIZE = 20
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
EPISODES = 100
MAX_STEPS = 400
SAVE_INTERVAL = 10
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
                q_update = (reward + GAMMA * torch.max(self(torch.Tensor(state_next))))
            q_values = self(torch.Tensor(state))
            # Ensure that the action index is within a valid range
            if action < self.action_space:
                q_values[action] = q_update
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
    observation_space = env.reset()
    action_space = 5

    model = DQNSolver(143, action_space)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    episode_rewards = []
    q_values_list = []
    all_q_values_list = []
    avg_q_values_list = []
    actions_counter = [0] * action_space
    epsilon_values = []
    aggregate_rewards = []

    for episode in range(1, EPISODES + 1):
        env.reset()
        state = torch.Tensor(observation_space)
        episode_reward = 0

        for step in range(MAX_STEPS):
            action = model.act(state)
            all_q_values_list.append(model(state).detach().numpy())
            actions_counter[action] += 1
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state_next = torch.Tensor(next_state)
            model.remember(state, action, reward, state_next, done)
            state = state_next
            if done or step == MAX_STEPS - 1:
                print(f"Episode: {episode}/{EPISODES}, Reward: {episode_reward}")
                episode_rewards.append(episode_reward)
                break
            epsilon_values.append(model.exploration_rate)
            if len(model.memory) > BATCH_SIZE:
                model.experience_replay(optimizer, criterion)

        q_values = model(state).detach().numpy()
        if episode % SAVE_INTERVAL == 0:
            average_q_value = np.mean(q_values)
            avg_q_values_list.append(average_q_value)
            aggregate_rewards.append(np.sum(episode_rewards[-SAVE_INTERVAL:]))
            model_save_path = '/Users/anureddy/Desktop/SEM02/Essential_ML/Bomberman_proj_draft/bomberman_rl/My_code/MY_CODE_NEW/outputs/trained_model_episode_dummy{}.pt'.format(episode)
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved successfully as '{model_save_path}'!")

    # Visualization of Q-Value Distributions, Actions Distribution, and Epsilon Value
    plt.figure(figsize=(12, 5))
    plt.title("Q-Value Distributions")
    plt.plot(avg_q_values_list)
    plt.xlabel(f"Episodes (x{SAVE_INTERVAL})")
    plt.ylabel("Average Q-Values")
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.title("Actions Distribution")
    plt.bar(range(action_space), actions_counter)
    plt.xlabel("Actions")
    plt.ylabel("Frequency")
    plt.xticks(range(action_space))
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.title("Epsilon Decay over Episodes")
    plt.plot(epsilon_values)
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.title("Aggregate Rewards Over Time")
    plt.plot(aggregate_rewards)
    plt.xlabel(f"Episodes (x{SAVE_INTERVAL})")
    plt.ylabel("Aggregate Rewards")
    plt.show()

if __name__ == "__main__":
    train()
