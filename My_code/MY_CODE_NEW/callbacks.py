import pygame

# Actions
MOVE_UP = 0
MOVE_DOWN = 1
MOVE_LEFT = 2
MOVE_RIGHT = 3
PLACE_BOMB = 4

BOMB_TIMER = 3  # Number of steps before a bomb explodes

class Player:
    def __init__(self, x, y, speed, wall_positions, crate_positions, coin_positions, enemy_positions, width, height):
        self.x = x
        self.y = y
        self.speed = speed
        self.width = width
        self.height = height
        self.wall_positions = wall_positions
        self.crate_positions = crate_positions
        self.coin_positions = coin_positions
        self.enemy_positions = enemy_positions
        self.bombs = []
        self.is_alive = True

    def move(self, dx, dy):
        new_x = self.x + dx * self.speed
        new_y = self.y + dy * self.speed

        if (new_x, new_y) not in self.wall_positions and (new_x, new_y) not in self.crate_positions and 0 <= new_x < self.width and 0 <= new_y < self.height:
            self.x = new_x
            self.y = new_y

    def place_bomb(self):
        self.bombs.append((self.x, self.y, BOMB_TIMER))

    def update_bombs(self, env):
        crates_destroyed = []
        player_killed_by_bomb = False

        # Iterate through bombs and decrease their timers
        for bomb in self.bombs:
            x, y, timer = bomb

            if timer <= 1:  # Bomb is about to explode
                # Define explosion range (for simplicity, let's consider a cross-shaped explosion)
                explosion_tiles = [(x, y), (x + 40, y), (x - 40, y), (x, y + 40), (x, y - 40)]
                
                # Check for crates destroyed
                for tile in explosion_tiles:
                    if tile in env.crate_positions:
                        crates_destroyed.append(tile)
                        env.crate_positions.remove(tile)

                # Check if player is killed by bomb
                if (self.x, self.y) in explosion_tiles:
                    player_killed_by_bomb = True
                    self.is_alive = False

                # Remove the bomb from the list as it has exploded
                self.bombs.remove(bomb)
            else:
                # Decrease the bomb timer
                self.bombs[self.bombs.index(bomb)] = (x, y, timer - 1)

        return player_killed_by_bomb, crates_destroyed
