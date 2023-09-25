import pygame
import torch
from env_manual import BombermanEnvironment
from train import DQNSolver
import time

# Initialize Pygame
pygame.init()
action_space=5
# Load the trained agent
MODEL_PATH = "MY_CODE_NEW/outputs/trained_model_episode_40.pt"  # Replace with the path to your saved model
agent = DQNSolver(143, action_space)  # Adjust dimensions if changed during training
agent.load_state_dict(torch.load(MODEL_PATH))
agent.eval()  # Set the agent to evaluation mode

# Create the game environment
env = BombermanEnvironment(render_mode=True)

# Initial state
state = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)
done = False
total_reward = 0

while not done:
    action = agent.act(state)
    
    # Apply the action in the environment
    next_state, reward, done = env.step(action)
    total_reward += reward
    
    # Render the environment and add a delay for visualization
    env.render()
    time.sleep(0.5)  # Add a delay of 0.5 seconds for better visualization
    
    state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

print(f"Total Reward Achieved: {total_reward}")

pygame.quit()  # Close the Pygame window
