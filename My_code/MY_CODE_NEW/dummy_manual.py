import pygame
import random
from callbacks import Player
# Constants
MOVE_UP = 0
MOVE_DOWN = 1
MOVE_LEFT = 2
MOVE_RIGHT = 3
PLACE_BOMB = 4
WALL_IMAGE = pygame.image.load("brick.png")
PLAYER_IMAGE = pygame.image.load("robot_blue.png")
COIN_IMAGE = pygame.image.load("coin.png")
CRATE_IMAGE = pygame.image.load("crate.png")
BOMB_IMAGE = pygame.image.load("bomb_blue.png")
EXPLOSION_IMAGE = pygame.image.load("explosion_0.png")
ENEMY_IMAGE = pygame.image.load("robot_green.png")
# Load images
#WALL_IMAGE = pygame.image.load("brick.png")
TILE_SIZE = WALL_IMAGE.get_width()
WIDTH = 13 * TILE_SIZE
HEIGHT = 11 * TILE_SIZE

class BombermanEnvironment:
    def __init__(self, render_mode=True):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Bomberman Test Environment')
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode

        self.initialize_game()

    def initialize_game(self):
        self.wall_positions = [(i * TILE_SIZE, j * TILE_SIZE) for i in range(WIDTH // TILE_SIZE) for j in range(HEIGHT // TILE_SIZE) if i % 2 == 1 and j % 2 == 1]
        self.crate_positions = [(random.randint(0, (WIDTH//TILE_SIZE)-1)*TILE_SIZE, random.randint(0, (HEIGHT//TILE_SIZE)-1)*TILE_SIZE) for _ in range(25)]
        self.coin_positions = [(random.randint(0, (WIDTH//TILE_SIZE)-1)*TILE_SIZE, random.randint(0, (HEIGHT//TILE_SIZE)-1)*TILE_SIZE) for _ in range(5)]
        self.enemy_positions = [(random.randint(0, (WIDTH//TILE_SIZE)-1)*TILE_SIZE, random.randint(0, (HEIGHT//TILE_SIZE)-1)*TILE_SIZE) for _ in range(2)]
        
        self.player = Player(30, 30, TILE_SIZE, self.wall_positions, self.crate_positions, self.coin_positions, self.enemy_positions, WIDTH, HEIGHT)
        self.game_over = False

    def reset(self):
        self.initialize_game()
        state_representation = self.get_binary_grid_representation() 
        return state_representation

    def get_binary_grid_representation(self):
    # 1. Initialization
        grid = [[0 for _ in range(WIDTH // TILE_SIZE)] for _ in range(HEIGHT // TILE_SIZE)]

        # 2. Populate the grid
        for wall in self.wall_positions:
            grid[wall[1] // TILE_SIZE][wall[0] // TILE_SIZE] = 1

        for crate in self.crate_positions:
            grid[crate[1] // TILE_SIZE][crate[0] // TILE_SIZE] = 2

        for coin in self.coin_positions:
            grid[coin[1] // TILE_SIZE][coin[0] // TILE_SIZE] = 3

        grid[self.player.y // TILE_SIZE][self.player.x // TILE_SIZE] = 4

        for enemy in self.enemy_positions:
            grid[enemy[1] // TILE_SIZE][enemy[0] // TILE_SIZE] = 5

        for bomb in self.player.bombs:
            grid[bomb[1] // TILE_SIZE][bomb[0] // TILE_SIZE] = 6

        # 3. Flatten the Grid
        state_representation = [item for sublist in grid for item in sublist]

        return state_representation

    def step(self, action):
        """
        Executes an action in the environment to change the state.
        Returns the new state, reward, and whether the game is done.
        """
        # Initial reward is set to 0
        reward = 0

        # Execute the player action
        if action == MOVE_UP:
            self.player.move(0, -1)
        elif action == MOVE_DOWN:
            self.player.move(0, 1)
        elif action == MOVE_LEFT:
            self.player.move(-1, 0)
        elif action == MOVE_RIGHT:
            self.player.move(1, 0)
        elif action == PLACE_BOMB:
            self.player.place_bomb()
        reward = -0.1

        # Check for coin collection
        player_pos = (self.player.x, self.player.y)
        if player_pos in self.coin_positions:
            self.coin_positions.remove(player_pos)
            reward += 50

        # Handle bomb logic and explosions
        player_killed_by_bomb, crates_destroyed = self.player.update_bombs(self)

        # Update rewards based on bomb effects
        reward += len(crates_destroyed) * 5
        if player_killed_by_bomb:
            reward -= 15

        # Check game-ending conditions
        done = False

        # All coins are collected
        if not self.coin_positions:
            done = True
            print("Game Over Reason: All coins are collected!")
        
        # Player killed by bomb
        if player_killed_by_bomb:
            done = True
            print("Game Over Reason: Player killed by bomb explosion!")

        # Max steps reached (assuming you have a max_steps attribute in the environment)
        if self.steps >= self.max_steps:
            done = True
            print("Game Over Reason: Max steps reached!")

        # TODO: Define the state representation logic
        # For example, using player position, coin positions, and enemy positions
        state_representation = self.get_binary_grid_representation()

        return state_representation, reward, done


    def render(self):
         if not self.render_mode:
             return

         self.screen.fill((0, 0, 0))

         # Draw walls
         for wall in self.wall_positions:
             self.screen.blit(WALL_IMAGE, wall)

         # Draw crates
         for crate in self.crate_positions:
             self.screen.blit(CRATE_IMAGE, crate)

         # Draw coins
         for coin in self.coin_positions:
            self.screen.blit(COIN_IMAGE, coin)

         # Draw enemies
         for enemy in self.enemy_positions:
             self.screen.blit(ENEMY_IMAGE, enemy)

         # Draw player
         player_pos = (self.player.x, self.player.y)
         self.screen.blit(PLAYER_IMAGE, player_pos)

         pygame.display.flip()
         self.clock.tick(60)

