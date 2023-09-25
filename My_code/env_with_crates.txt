import pygame
import random
import numpy as np
from callback import Player 
pygame.init() # Thsi initializes all the modules required for pygame 

# Load images/assets
WALL_IMAGE = pygame.image.load("brick.png")
CRATE_IMAGE = pygame.image.load("crate.png")
PLAYER_IMAGE = pygame.image.load("player2.png")
COIN_IMAGE = pygame.image.load("coin.png")

#The width of the WALL_IMAGE is used as the TILE_SIZE, representing the size of each grid cell.
TILE_SIZE = WALL_IMAGE.get_width()
WIDTH = 13 * TILE_SIZE
HEIGHT = 11 * TILE_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Bomberman Environment')

#This class defines the Bomberman game environment.
class BombermanEnvironment:
    def __init__(self): 
        self.wall_positions, self.crate_positions, self.coin_positions = self.generate_environment_positions()
        self.player = Player(TILE_SIZE, TILE_SIZE, self.wall_positions, self.crate_positions, WIDTH, HEIGHT)
        self.grid_state = self.get_initial_grid_state()

    def pos_to_grid(self, x, y):
        return y // TILE_SIZE, x // TILE_SIZE

    def grid_to_pos(self, i, j):
        return j * TILE_SIZE, i * TILE_SIZE

    def generate_environment_positions(self):
        ''' This method generates the positions for walls, crates, and coins. It also ensures no crates are adjacent to the player's starting position.'''
        walls = [(x, y) for x in range(0, WIDTH, 2*TILE_SIZE) for y in range(0, HEIGHT, 2*TILE_SIZE)]
        crates = [(x, y) for x in range(0, WIDTH, TILE_SIZE) for y in range(0, HEIGHT, TILE_SIZE) if (x, y) not in walls and random.random() < 0.3]
        
        # Ensure no crates around the player's starting position
        player_start_positions = [
            (TILE_SIZE, TILE_SIZE), 
            (TILE_SIZE + TILE_SIZE, TILE_SIZE),
            (TILE_SIZE - TILE_SIZE, TILE_SIZE),
            (TILE_SIZE, TILE_SIZE + TILE_SIZE),
            (TILE_SIZE, TILE_SIZE - TILE_SIZE)
        ]

        for pos in player_start_positions:
            if pos in crates:
                crates.remove(pos)
        
        coin_positions = set()
        while len(coin_positions) < 10:
            x = random.randint(0, WIDTH // TILE_SIZE - 1) * TILE_SIZE
            y = random.randint(0, HEIGHT // TILE_SIZE - 1) * TILE_SIZE
            if (x, y) not in walls and (x, y) not in crates:
                coin_positions.add((x, y))
        return walls, crates, coin_positions

    def get_initial_grid_state(self):
        '''This method initializes the state of the grid. It's a matrix where each cell can represent an entity (wall, crate, coin, or player).'''
        state = np.zeros((HEIGHT // TILE_SIZE, WIDTH // TILE_SIZE), dtype=int)
        for x, y in self.wall_positions:
            state[self.pos_to_grid(x, y)] = 1
        for x, y in self.crate_positions:
            state[self.pos_to_grid(x, y)] = 2
        for x, y in self.coin_positions:
            state[self.pos_to_grid(x, y)] = 3
        state[self.pos_to_grid(self.player.x, self.player.y)] = 4
        return state

    def reset(self):
        '''Resets the game to its initial state. Useful for starting new episodes in learning algorithms.'''
        self.wall_positions, self.crate_positions, self.coin_positions = self.generate_environment_positions()
        self.player = Player(TILE_SIZE, TILE_SIZE, self.wall_positions, self.crate_positions, WIDTH, HEIGHT)
        self.grid_state = self.get_initial_grid_state()
        return self.grid_state.copy()

    def step(self, action):
        '''This method returns the current state of the game. It's used for machine learning algorithms to understand the current situation of the game.'''
        prev_pos = (self.player.x, self.player.y)
        dx, dy = 0, 0
        if action == 0: #player should move up by one tile,so the y-coordinate is decremented by the TILE_SIZE
            dy = -TILE_SIZE
        elif action == 1:#the player should move down by one tile. The y-coordinate is increased by the TILE_SIZE.
            dy = TILE_SIZE
        elif action == 2: #the player should move left by one tile. Thus, the x-coordinate is decremented by the TILE_SIZE.
            dx = -TILE_SIZE
        elif action == 3: #the player should move right by one tile. Therefore, the x-coordinate is incremented by the TILE_SIZE.
            dx = TILE_SIZE
        
        self.player.move(dx, dy)
        reward = -1  # Default reward
        
        # Update grid_state for player's new position
        self.grid_state[self.pos_to_grid(*prev_pos)] = 0
        self.grid_state[self.pos_to_grid(self.player.x, self.player.y)] = 4

        if (self.player.x, self.player.y) in self.coin_positions:
            self.coin_positions.remove((self.player.x, self.player.y))
            self.grid_state[self.pos_to_grid(self.player.x, self.player.y)] = 4  # Mark as player position
            reward = 10

        done = len(self.coin_positions) == 0
        return self.grid_state.copy(), reward, done


    def get_state(self):
        '''This method returns the current state of the game. 
        It's used for machine learning algorithms to understand the current situation of the game.'''
        max_coins = 10
        state = [self.player.x, self.player.y]
        for coin in list(self.coin_positions)[:max_coins]:
            state.extend(coin)
        # Padding with zeros if there are fewer coins
        while len(state) < 2 + max_coins * 2:
            state.extend([0, 0])
        return np.array(state)


    def render(self):
        screen.fill((0, 0, 0))
        for x, y in self.wall_positions:
            screen.blit(WALL_IMAGE, (x, y))
        for x, y in self.crate_positions:
            screen.blit(CRATE_IMAGE, (x, y))
        for x, y in self.coin_positions:
            screen.blit(COIN_IMAGE, (x, y))
        self.player.draw(screen, PLAYER_IMAGE)
        pygame.display.flip()
