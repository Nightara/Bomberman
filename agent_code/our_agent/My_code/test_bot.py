import os
import numpy as np
import pygame
import torch
from bomber_environment import BombermanEnvironment, HEIGHT, WIDTH, TILE_SIZE
from train_pytorch import CoinCollectorAgent, MAX_STEPS

def test_trained_agent(model_path, episodes=10):
    # Initialize environment and agent
    env = BombermanEnvironment()
    state_dim = (HEIGHT // TILE_SIZE) * (WIDTH // TILE_SIZE) + 1
    agent = CoinCollectorAgent(state_dim, 4)
    
    # Load trained model
    agent.q_network.load_state_dict(torch.load(model_path))
    agent.q_network.eval()

    for episode in range(episodes):
        state = env.reset().reshape(1, -1)
        done = False
        total_reward = 0
        turn_count = 0
        coins_collected = 0  # Initialize coins_collected to 0

        while not done and turn_count < MAX_STEPS:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            
            turn_encoded = 0.9 ** turn_count
            extended_state = np.append(state, turn_encoded).reshape(1, -1)
            
            action = agent.act(extended_state) #comment q_values and remove the return of qvalue in coincollectingagent class 
            next_state, reward, done = env.step(action)
            next_state = next_state.reshape(1, -1)
            
            #if q_values is not None:  # Skip printing Q-values if a random action was taken
                #print(f"Confidence (Q-values): {q_values}")
            
            if reward == 15:  # Check if a coin has been collected
                coins_collected += 1  # Increment coins_collected

            state = next_state
            total_reward += reward
            
            env.render()
            pygame.time.wait(100)  # Delay for visualization
            
            turn_count += 1

        print(f"Episode: {episode + 1}/{episodes}, Total Reward: {total_reward}, Coins Collected: {coins_collected}")

if __name__ == "__main__":
    model_folder = "bomberman_rl/My_code/Outputs"  # Replace with your directory where models are saved
    for episode in range(0, 501, 100):
        model_file = f"model_episode_{episode}.pth"
        model_path = os.path.join(model_folder, model_file)
        
        if os.path.exists(model_path):
            print(f"Testing model from episode {episode}...")
            test_trained_agent(model_path, episodes=10)
        else:
            print(f"Model file {model_path} does not exist. Skipping...")
