import torch
import torch.nn as nn 
import torch.optim as optim
import random
from collections import namedtuple, deque
import time
import numpy as np
import torch.nn.functional as F 
from environment import BombermanEnvironment, WIDTH, HEIGHT
from callback import Player
from torch.optim.lr_scheduler import StepLR
from IPython.display import display, clear_output
import matplotlib.pyplot as plt


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def to_tensor(data_list, dtype=torch.float32):
    if isinstance(data_list[0], np.ndarray):
        data_list = np.array(data_list)
    return torch.tensor(data_list, dtype=dtype)

class DQNNetwork(nn.Module): 
    def __init__(self, input_dim, output_dim,hidden_dim=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, input_dim, output_dim, hidden_dim=128, lr=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=1000):
        self.model = DQNNetwork(input_dim, output_dim, hidden_dim)
        self.target_model = DQNNetwork(input_dim, output_dim, hidden_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.output_dim = output_dim
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
        self.steps_done = 0
        self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.9)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.TARGET_UPDATE = 10  # adjust this value for frequency of updating the target network

    def choose_action(self, state):
        self.steps_done += 1
        self.update_epsilon()

        state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)

        if random.random() < self.epsilon:
            return random.randrange(0, self.output_dim)  # Ensure output_dim is the correct value for the action space size
        else:
            with torch.no_grad():
                return self.model(state_tensor).max(1)[1].item()

    def update_epsilon(self):
        self.epsilon -= (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
        self.epsilon = max(self.epsilon_end, self.epsilon)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state.flatten(), action, reward, next_state.flatten(), done)

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        
        transitions = self.replay_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = to_tensor([s for s in batch.next_state if s is not None])
        state_batch = to_tensor(batch.state)
        action_batch = to_tensor(batch.action, dtype=torch.int64).unsqueeze(1)
        reward_batch = to_tensor(batch.reward).unsqueeze(1)

        state_action_values = self.model(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(batch_size)
        next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()
        next_state_values = next_state_values.unsqueeze(1) 
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Update the target network
        if self.steps_done % self.TARGET_UPDATE == 0:
            self.target_model.load_state_dict(self.model.state_dict())


def dynamic_plot(rewards, episode_durations, ylabel, title):
    clear_output(wait=True)
    
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))
    
    # Plotting rewards
    axs[0].plot(rewards, label='Reward')
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel(ylabel)
    axs[0].set_title(title)
    
    # Plotting episode durations
    axs[1].plot(episode_durations, label='Episode Duration (s)')
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Duration (s)")
    axs[1].set_title("Episode Duration over Time")

    plt.legend()
    plt.show()

       
        

def train_dqn(agent, env, episodes, batch_size, save_model_every=20):
    start_time = time.time()
    rewards = []
    episode_durations = []
    MAX_STEPS_PER_EPISODE = 500

    for episode in range(episodes):
        episode_start_time = time.time()
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < MAX_STEPS_PER_EPISODE: 
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            agent.train(batch_size)

            state = next_state
            total_reward += reward
            steps += 1

        # If you reach the maximum steps without finishing the game, adjust reward accordingly
        if steps >= MAX_STEPS_PER_EPISODE: 
            if env.coins_collected == 0:
                reward = -50
            elif env.coins_collected < 3:
                reward = -20
            else:
                reward = -10
            total_reward += reward

        coins_per_turn = env.coins_collected / steps if steps else 0
        agent.update_epsilon()
        episode_end_time = time.time()
        rewards.append(total_reward)
        episode_durations.append(episode_end_time - episode_start_time)
        current_lr = agent.optimizer.param_groups[0]['lr']
        print(f"Episode: {episode + 1}/{episodes}, Steps: {steps}, Total Reward: {total_reward}, Coins Collected: {env.coins_collected}, Coins per Turn: {coins_per_turn:.2f}, Epsilon: {agent.epsilon:.4f}, Episode Duration: {episode_end_time - episode_start_time:.2f} seconds")
        if episode % 10 == 0: 
            dynamic_plot(rewards, episode_durations, ylabel="Total Reward", title="Training Performance")
        if episode % save_model_every == 0:
            torch.save(agent.model.state_dict(), f"model_episode_{episode}.pth")

    end_time = time.time()
    print(f"Training complete. Total Duration: {end_time - start_time:.2f} seconds")

    return rewards, episode_durations





if __name__ == "__main__":
    # Hyperparameters
    EPSILON_DECAY_STEPS = 5000
    LEARNING_RATE = 0.0003
    GAMMA = 0.99

    env = BombermanEnvironment()
    agent = DQNAgent(input_dim=WIDTH*HEIGHT + 1, output_dim=4, lr=LEARNING_RATE, gamma=GAMMA, epsilon_decay_steps=EPSILON_DECAY_STEPS)
    episodes = 100
    batch_size = 32

    rewards, episode_durations = train_dqn(agent, env, episodes, batch_size)
