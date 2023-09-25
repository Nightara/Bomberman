# agent_code/callback.py
import pygame

# ... You can also import any other necessary modules or libraries here ...

class Player:
    def __init__(self, x, y, walls, crates, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.walls = walls
        self.crates = crates

    def move(self, dx, dy):
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Check boundaries
        if new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height:
            return
        
        if (new_x, new_y) not in self.walls and (new_x, new_y) not in self.crates:
            self.x = new_x
            self.y = new_y

    def draw(self, screen, image):
        screen.blit(image, (self.x, self.y))


