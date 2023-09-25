import pygame

# Define the size of the image
WIDTH = 32
HEIGHT = 32

# Create a red player
player = pygame.Surface((WIDTH, HEIGHT))
player.fill((255, 0, 0))

# Draw a circle in the center of the player
pygame.draw.circle(player, (255, 255, 255), (16, 16), 16)

# Save the image
pygame.image.save(player, "player.png")