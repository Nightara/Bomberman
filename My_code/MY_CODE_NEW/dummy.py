import pygame
import numpy as np
import random
import time

# Constants
BOARD_SIZE = 10
WALL = 1
CRATE = 2
COIN = 3
PLAYER = 4
EMPTY = 0
BOMB = 5
EXPLOSION = 6
MAX_STEPS = 400
BOMB_TIME = 3  # seconds before bomb explodes

# Initialize pygame
pygame.init()

# Load images
wall_image = pygame.image.load('brick.png')
crate_image = pygame.image.load('crate.png')
coin_image = pygame.image.load('coin.png')
player_image = pygame.image.load('player2.png')
bomb_image = pygame.image.load('bomb_blue.png')
explosion_image = pygame.image.load('explosion_0.png')


class BombermanEnvironment:
    def __init__(self, render_mode=True):
        pygame.init()
        self.screen = pygame.display.set_mode((BOARD_SIZE * 32, BOARD_SIZE * 32))
        pygame.display.set_caption('Bomberman Environment')
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        self.player_position = None
        self.game_over = False
        self.steps = 0
        self.bomb_position = None
        self.bomb_timer = None
        self.render_mode = render_mode

    def reset(self):
        self.board.fill(EMPTY)
        self.initialize_board()
        self.place_player_at_corner()
        self.steps = 0
        self.bomb_position = None
        self.bomb_timer = None
        return self.get_state()

    def initialize_board(self):
        # Place walls
        self.board[1::2, 1::2] = WALL
        # Place random crates and coins
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == EMPTY and (i, j) not in [(0, 0), (0, 1), (1, 0)]:
                    if random.random() < 0.5:
                        self.board[i][j] = CRATE if random.random() < 0.75 else COIN

    def place_player_at_corner(self):
        self.player_position = (0, 0)
        self.board[0][0] = PLAYER

    def get_state(self):
        return self.board.copy()

    def is_valid_move(self, x, y):
        return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and self.board[x][y] in [EMPTY, COIN]

    def check_game_over(self):
        if self.steps >= MAX_STEPS:
            return True
        if not np.any(self.board == COIN):
            return True
        if self.game_over:
            return True
        return False

    def place_bomb(self):
        if not self.bomb_position:
            x, y = self.player_position
            self.board[x][y] = BOMB
            self.bomb_position = (x, y)
            self.bomb_timer = time.time() + BOMB_TIME

    def check_bomb_explosion(self):
        if self.bomb_position and time.time() > self.bomb_timer:
            x, y = self.bomb_position
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if 0 <= x + dx < BOARD_SIZE and 0 <= y + dy < BOARD_SIZE:
                        if self.board[x + dx][y + dy] in [CRATE, COIN]:
                            self.board[x + dx][y + dy] = EMPTY
                        elif self.board[x + dx][y + dy] == PLAYER:
                            self.game_over = True
            self.bomb_position = None
            self.bomb_timer = None

        def step(self, action):
            x, y = self.player_position
            new_x, new_y = x, y  # Keep track of potential new position

            # Determine potential new position based on the action
            if action == "UP" and self.is_valid_move(x-1, y):
                new_x = x-1
            elif action == "DOWN" and self.is_valid_move(x+1, y):
                new_x = x+1
            elif action == "LEFT" and self.is_valid_move(x, y-1):
                new_y = y-1
            elif action == "RIGHT" and self.is_valid_move(x, y+1):
                new_y = y+1
            elif action == "PLACE_BOMB":
                self.place_bomb()

            # Clear the player's previous position
            self.board[x][y] = EMPTY if self.board[x][y] != BOMB else BOMB

            # Update the player's position
            x, y = new_x, new_y
            self.player_position = (x, y)
            self.board[x][y] = PLAYER

            # Check bomb explosion
            self.check_bomb_explosion()

            # Update steps and check game over conditions
            self.steps += 1
            done = self.check_game_over()
            
            # Return the new state, reward, and done flag
            return self.get_state(), self.get_reward(x, y), done


    def get_reward(self, x, y):
        if self.board[x][y] == COIN:
            return 50
        elif self.board[x][y] == EXPLOSION:
            self.game_over = True
            return -50
        elif self.board[x][y] == CRATE:
            return 20
        return 0
    def render(self):
        self.screen.fill((0, 0, 0))  # Clear screen

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == WALL:
                    self.screen.blit(wall_image, (j * 32, i * 32))
                elif self.board[i][j] == CRATE:
                    self.screen.blit(crate_image, (j * 32, i * 32))
                elif self.board[i][j] == COIN:
                    self.screen.blit(coin_image, (j * 32, i * 32))
                elif self.board[i][j] == PLAYER:
                    self.screen.blit(player_image, (j * 32, i * 32))
                elif self.board[i][j] == BOMB:
                    self.screen.blit(bomb_image, (j * 32, i * 32))
                elif self.board[i][j] == EXPLOSION:
                    self.screen.blit(explosion_image, (j * 32, i * 32))

        pygame.display.flip()

# To run the game
env = BombermanEnvironment()

# Initial render
env.reset()
env.render()

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                env.steps("UP")
            elif event.key == pygame.K_DOWN:
                env.steps("DOWN")
            elif event.key == pygame.K_LEFT:
                env.steps("LEFT")
            elif event.key == pygame.K_RIGHT:
                env.steps("RIGHT")
            elif event.key == pygame.K_SPACE:
                env.steps("PLACE_BOMB")
            env.render()

pygame.quit()
