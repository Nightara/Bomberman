import numpy as np
import random
from callback import Player

WIDTH, HEIGHT = 13, 11
TILE_SIZE = 1  # Just a representation since we're not using pygame

class BombermanEnvironment:
    def __init__(self, max_coins=10, max_turns=200): 
        self.max_coins = max_coins
        self.max_turns = max_turns
        self.wall_positions, self.crate_positions, self.coin_positions = self.generate_environment_positions()
        self.turn = 0
        self.coins_collected = 0  
        self.player = Player(0, 0, self.wall_positions, self.crate_positions, WIDTH, HEIGHT)
        self.player.recent_locations = []
        self.player.visited_locations = set()
        self.last_action = None
    def reward(self, action):
        current_position = tuple(self.player.get_position())

        # If the player collects a coin
        if current_position in self.coin_positions:
            #self.coins_collected += 1
            return 10

        # If the player tries to move into a wall
        if current_position in self.wall_positions:
            return -2

        prev_distance = self.calculate_distance_to_nearest_coin(self.player.prev_position)
        current_distance = self.calculate_distance_to_nearest_coin(current_position)
    
        if current_distance < prev_distance:
          return 0.5  # Reward for moving closer to a coin
        else:
          return -0.5  # Reduced penalty for regular movement

  

    def generate_environment_positions(self):
        walls = [(x, y) for x in range(0, WIDTH, 2*TILE_SIZE) for y in range(0, HEIGHT, 2*TILE_SIZE)]
        possible_coin_positions = [(x, y) for x in range(0, WIDTH, TILE_SIZE) for y in range(0, HEIGHT, TILE_SIZE) if (x, y) not in walls]
        
        # Ensure that the number of possible coin positions is not less than max_coins
        num_possible_coin_positions = len(possible_coin_positions)
        if num_possible_coin_positions < self.max_coins:
            self.max_coins = num_possible_coin_positions
        
        coins = random.sample(possible_coin_positions, self.max_coins)
        print(f"Number of coins generated: {len(coins)}")  # Add this line
        return walls, [], set(coins)

    def reset(self):
        self.coins_collected = 0
        print(f"At reset: Coins collected = {self.coins_collected}")
        self.wall_positions, self.crate_positions, self.coin_positions = self.generate_environment_positions()
        self.player = Player(1, 1, self.wall_positions, self.crate_positions, WIDTH, HEIGHT)
        self.turn = 0
        self.last_action = None
        return self.get_state()


    
    def calculate_distance_to_nearest_coin(self, position):
        distances = [abs(position[0] - coin_x) + abs(position[1] - coin_y) for coin_x, coin_y in self.coin_positions]
        if distances:
            return min(distances)
        return 0 



    def step(self, action):
      dx, dy = 0, 0
      if action == 0: dy = -1
      elif action == 1: dy = 1
      elif action == 2: dx = -1
      elif action == 3: dx = 1

      self.player.move(dx, dy)
      current_position = tuple(self.player.get_position())

      # Check for coin collection first
      if current_position in self.coin_positions:
          self.coins_collected += 1  # Increment coin count
          self.coin_positions.remove(current_position)  # Remove the coin from the environment once collected
          print(f"Coin collected at position {current_position} by action {action}. Total coins now: {self.coins_collected}")
          reward = 10
      else:
          reward = self.reward(action)

      done = self.coins_collected >= self.max_coins or len(self.coin_positions) == 0
      self.turn += 1

      info = {'coins_collected': self.coins_collected}
      return self.get_state(), reward, done, info




  

    def get_state(self):
        grid = np.zeros((WIDTH, HEIGHT))
        grid[self.player.x][self.player.y] = 1
        for (x, y) in self.coin_positions:
            grid[x][y] = 2
        for (x, y) in self.crate_positions:
            grid[x][y] = 3
        for (x, y) in self.wall_positions:
            grid[x][y] = 4
        flattened_grid = grid.flatten()/4.0
        turn_encoded = 0.9 ** self.turn
        return np.concatenate([flattened_grid, [turn_encoded]])

def print_coin_collection(self):
    current_position = tuple(self.player.get_position())
    if current_position in self.coin_positions:
        print(f"Coin collected at position {current_position} by action {self.last_action}. Total coins now: {self.coins_collected}")
        self.coin_positions.remove(current_position)
        self.last_action = None
    else:
        print("No coins collected")


 
