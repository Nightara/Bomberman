import pygame
import random

# Initialize pygame
pygame.init()

# Load images/assets
WALL_IMAGE = pygame.image.load("brick.png")
PLAYER_IMAGE = pygame.image.load("player2.png")
COIN_IMAGE = pygame.image.load("coin.png")
CRATE_IMAGE = pygame.image.load("crate.png")
BOMB_IMAGE = pygame.image.load("bomb_blue.png")
EXPLOSION_IMAGE = pygame.image.load("explosion_0.png")

# Define constants using the loaded images
TILE_SIZE = WALL_IMAGE.get_width()
WIDTH = 13 * TILE_SIZE
HEIGHT = 11 * TILE_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Bomberman Environment')

# Colors
BLACK = (0, 0, 0)
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

# Actions
MOVE_UP = 0
MOVE_DOWN = 1
MOVE_LEFT = 2
MOVE_RIGHT = 3
PLACE_BOMB = 4

num_coins = 5
# Bomb and explosion

COIN_REWARD = 10
PLAYER_DIED_REWARD = -100

# Bomb and explosion
BOMB_TIMER = 3
EXPLOSION_TIMER = 3
from random import randint, choice

# Import Player class from callbacks.py
from callbacks import Player
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Bomberman Environment')
class BombermanEnvironment:
    def __init__(self):

        # Walls are set as alternate grids
        self.wall_positions = [(x, y) for x in range(0, WIDTH, 2*TILE_SIZE) for y in range(0, HEIGHT, 2*TILE_SIZE)]
        
        # Randomly place crates, ensuring they don't overlap with walls or the player's initial position
        self.crate_positions = [(randint(0, (WIDTH//TILE_SIZE)-1)*TILE_SIZE, randint(0, (HEIGHT//TILE_SIZE)-1)*TILE_SIZE) for _ in range(30)]
        self.crate_positions = [pos for pos in self.crate_positions if pos not in self.wall_positions and pos != (TILE_SIZE, TILE_SIZE)]
        
        # Randomly place coins within crates
        self.coin_positions = random.sample(self.crate_positions, num_coins)
        self.coin_positions_hidden = [choice(self.crate_positions) for _ in range(5)]  # Example for 5 coins

        # Initialize player, bomb, and explosion attributes
        self.player = Player(TILE_SIZE, TILE_SIZE, TILE_SIZE, self.wall_positions, self.crate_positions, WIDTH, HEIGHT)
        self.bomb_positions = []
        self.explosion_positions = []
        self.bomb_timer = -1
        self.explosion_timer = -1

    def generate_environment_positions(self):
        # Generate wall positions
        wall_positions = [(i, j) for i in range(0, self.width, self.tile_size*2) for j in range(0, self.height, self.tile_size*2)]

        # Generate crate positions
        crate_positions = [(i, j) for i in range(0, self.width, self.tile_size) for j in range(0, self.height, self.tile_size) if (i, j) not in wall_positions and random.random() < 0.7]

        # Generate coin positions (inside crates)
        coin_positions = [(i, j) for i, j in crate_positions if random.random() < 0.2]

        return wall_positions, coin_positions, crate_positions

    def step(self, action):
        reward = 0
        done = False

        # Handle player movement
        if action == MOVE_LEFT:
            self.player.move(-1, 0)
        elif action == MOVE_RIGHT:
            self.player.move(1, 0)
        elif action == MOVE_UP:
            self.player.move(0, -1)
        elif action == MOVE_DOWN:
            self.player.move(0, 1)
        elif action == PLACE_BOMB and not self.bomb_positions:
            self.bomb_positions.append((self.player.x, self.player.y))
            self.bomb_timer = BOMB_TIMER

        # Handle bomb explosion
        if self.bomb_timer == 0:
            explosion_positions = self.handle_explosion()
            self.explosion_positions.extend(explosion_positions)
            self.explosion_timer = EXPLOSION_TIMER
        elif self.bomb_timer > 0:
            self.bomb_timer -= 1

        if self.explosion_timer == 0:
            self.explosion_positions = []
        elif self.explosion_timer > 0:
            self.explosion_timer -= 1

        # Handle coin collection
# Handle coin collection
        if (self.player.x, self.player.y) in self.coin_positions:
            self.coin_positions.remove((self.player.x, self.player.y))
            reward += COIN_REWARD


        # Handle player death if in explosion range
        if (self.player.x, self.player.y) in self.explosion_positions:
            reward += PLAYER_DIED_REWARD
            done = True

        return reward, done

    def handle_explosion(self):
        explosion_positions = []
        if self.bomb_timer == 0 and self.bomb_positions:
            bomb_x, bomb_y = self.bomb_positions[0]
            explosion_positions.append((bomb_x, bomb_y))
            
            # Check explosion in all directions
            for dx, dy in DIRECTIONS:
                x, y = bomb_x + dx*TILE_SIZE, bomb_y + dy*TILE_SIZE
                
                # If explosion hits a wall, stop in that direction
                if (x, y) in self.wall_positions:
                    continue
                
                # Add explosion position
                explosion_positions.append((x, y))
                
                # If explosion hits a crate, remove the crate and check for hidden coins
                if (x, y) in self.crate_positions:
                    self.crate_positions.remove((x, y))
                    if (x, y) in self.coin_positions_hidden:
                        self.coin_positions.append((x, y))
                        self.coin_positions_hidden.remove((x, y))
            
            # Remove the bomb that just exploded
            self.bomb_positions.pop(0)
        return explosion_positions





    def render(self):
        if not self.render_mode:
            return
    # Fill the screen with the background color
        screen.fill(BLACK)

        # Draw walls
        for x, y in self.wall_positions:
            screen.blit(WALL_IMAGE, (x, y))

        # Draw crates (ensuring they aren't drawn over walls)
        for x, y in self.crate_positions:
            if (x, y) not in self.wall_positions:
                screen.blit(CRATE_IMAGE, (x, y))

        # Draw coins
        for x, y in self.coin_positions:
            if (x, y) not in self.crate_positions:  # Only display coins that aren't inside crates
                screen.blit(COIN_IMAGE, (x, y))

        # Draw player
        screen.blit(PLAYER_IMAGE, (self.player.x, self.player.y))

        # Draw bombs
        for x, y in self.bomb_positions:
            screen.blit(BOMB_IMAGE, (x, y))

        # Draw explosions
        for x, y in self.explosion_positions:
            screen.blit(EXPLOSION_IMAGE, (x, y))

        # Update the display
        pygame.display.flip()


# Create environment
env = BombermanEnvironment()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                env.step(MOVE_LEFT)
            elif event.key == pygame.K_RIGHT:
                env.step(MOVE_RIGHT)
            elif event.key == pygame.K_UP:
                env.step(MOVE_UP)
            elif event.key == pygame.K_DOWN:
                env.step(MOVE_DOWN)
            elif event.key == pygame.K_SPACE:
                env.step(PLACE_BOMB)
    env.render()

pygame.quit()
