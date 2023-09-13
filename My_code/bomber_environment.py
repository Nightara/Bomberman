import pygame
import random
import numpy as np
from My_code.callback import Player
## pygame.init() # Thsi initializes all the modules required for pygame

# Load images/assets
## WALL_IMAGE = pygame.image.load("brick.png")
WALL_IMAGE = None
## CRATE_IMAGE = pygame.image.load("crate.png")
CRATE_IMAGE = None
## PLAYER_IMAGE = pygame.image.load("player2.png")
PLAYER_IMAGE = None
## COIN_IMAGE = pygame.image.load("coin.png")
COIN_IMAGE = None

#The width of the WALL_IMAGE is used as the TILE_SIZE, representing the size of each grid cell.
TILE_SIZE = 30
## TILE_SIZE = WALL_IMAGE.get_width()
WIDTH = 13 * TILE_SIZE
HEIGHT = 11 * TILE_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Bomberman Environment')

#This class defines the Bomberman game environment.
class BombermanEnvironment:
    def __init__(self): 
        self.wall_positions, self.coin_positions, self.crate_positions = self.generate_environment_positions()
        self.player = Player(TILE_SIZE, TILE_SIZE, self.wall_positions, self.crate_positions, WIDTH, HEIGHT)
        self.grid_state = self.get_initial_grid_state()

    def pos_to_grid(self, x, y):
        return y // TILE_SIZE, x // TILE_SIZE

    def grid_to_pos(self, i, j):
        return j * TILE_SIZE, i * TILE_SIZE

    def generate_environment_positions(self):
    # Initialize empty sets for walls and coins
        walls = set()
        coin_positions = set()
        crate_positions = set()

        # Create a helper function to check if a position is surrounded by walls
        def is_surrounded(x, y, wall_set):
            neighbors = [
                (x + TILE_SIZE, y),
                (x - TILE_SIZE, y),
                (x, y + TILE_SIZE),
                (x, y - TILE_SIZE)
            ]
            return all(neighbor in wall_set for neighbor in neighbors)

        # Randomly place walls, ensuring the player's starting position is not surrounded
        num_walls = 50  # You can adjust this number
        while len(walls) < num_walls:
            x = random.randint(0, (WIDTH // TILE_SIZE) - 1) * TILE_SIZE
            y = random.randint(0, (HEIGHT // TILE_SIZE) - 1) * TILE_SIZE
            if (x, y) != (TILE_SIZE, TILE_SIZE) and not is_surrounded(x, y, walls):
                walls.add((x, y))

        # Place coins, ensuring they are not on walls or surrounded by walls
        while len(coin_positions) < 10:
            x = random.randint(0, (WIDTH // TILE_SIZE) - 1) * TILE_SIZE
            y = random.randint(0, (HEIGHT // TILE_SIZE) - 1) * TILE_SIZE
            if (x, y) not in walls and not is_surrounded(x, y, walls):
                coin_positions.add((x, y))
        
        #Place crates, ensuring they are not on walls or surrounded by walls
        num_crates = 20
        while len(crate_positions) < num_crates:
            x = random.randint(0, (WIDTH//TILE_SIZE) - 1) * TILE_SIZE
            y = random.randint(0, (HEIGHT//TILE_SIZE) - 1) * TILE_SIZE
            if(x,y) not in walls and not is_surrounded(x, y, walls):
                crate_positions.add((x,y))

        return walls, coin_positions, crate_positions



    def get_initial_grid_state(self):
        '''This method initializes the state of the grid. It's a matrix where each cell can represent an entity (wall, coin, or player).'''
        state = np.zeros((HEIGHT // TILE_SIZE, WIDTH // TILE_SIZE), dtype=int)
        for x, y in self.wall_positions:
            state[self.pos_to_grid(x, y)] = 1
        for x, y in self.coin_positions:
            state[self.pos_to_grid(x, y)] = 3
        for x,y in self.crate_positions:
            state[self.pos_to_grid(x, y)] = 2
        state[self.pos_to_grid(self.player.x, self.player.y)] = 4
        return state
    def reset(self):
        '''Resets the game to its initial state. Useful for starting new episodes in learning algorithms.'''
        self.wall_positions, self.coin_positions, self.crate_positions = self.generate_environment_positions()
        self.player = Player(TILE_SIZE, TILE_SIZE, self.wall_positions, self.crate_positions, WIDTH, HEIGHT)
        self.grid_state = self.get_initial_grid_state()
        return self.grid_state.copy()


    def step(self, action): # reward policy 
        prev_pos = (self.player.x, self.player.y)
        dx, dy = 0, 0
        if action == 0:
            dy = -TILE_SIZE
        elif action == 1:
            dy = TILE_SIZE
        elif action == 2:
            dx = -TILE_SIZE
        elif action == 3:
            dx = TILE_SIZE
        
        can_move = self.player.is_valid_move(dx, dy)
        self.player.move(dx, dy)
        
        reward = 0  # Initialize with zero reward
        if not can_move:
            reward = -1  # Large negative reward for hitting a wall
        
        # Update grid_state for player's new position
        self.grid_state[self.pos_to_grid(*prev_pos)] = 0
        self.grid_state[self.pos_to_grid(self.player.x, self.player.y)] = 4

        if (self.player.x, self.player.y) in self.coin_positions:
            self.coin_positions.remove((self.player.x, self.player.y))
            self.grid_state[self.pos_to_grid(self.player.x, self.player.y)] = 4  # Mark as player position
            reward = 15  # Positive reward for collecting a coin

        done = len(self.coin_positions) == 0
        return self.grid_state.copy(), reward, done


    def get_state(self):
        '''This method returns the current state of the game. 
        It's used for machine learning algorithms to understand the current situation of the game.'''
        max_coins = 10
        max_crates = 10
        state = [self.player.x, self.player.y]
        for coin in list(self.coin_positions)[:max_coins]:
            state.extend(coin)
        # Padding with zeros if there are fewer coins
        for crate in list(self.crate_positions)[:max_crates]:
            state.extend(max_crates)
        while len(state) < 2 + max_coins * 2:
            state.extend([0, 0])
        while len(state) < 2 + max_crates * 2:
            state.extend([0,0])
        return np.array(state)


    def render(self):
        screen.fill((0, 0, 0))
        for x, y in self.wall_positions:
            screen.blit(WALL_IMAGE, (x, y))
        for x, y in self.coin_positions:
            screen.blit(COIN_IMAGE, (x, y))
        for x, y in self.crate_positions:
            screen.blit(CRATE_IMAGE, (x, y))
        self.player.draw(screen, PLAYER_IMAGE)
        pygame.display.flip()
