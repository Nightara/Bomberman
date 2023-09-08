class Player:
    def __init__(self, x, y, wall_positions, WIDTH, HEIGHT):
        self.x = x
        self.y = y
        self.wall_positions = wall_positions
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.recent_locations = []
        self.visited_locations = set()

    def get_position(self):
        """Returns the current position (x, y) of the player."""
        return self.x, self.y

    def is_valid_move(self, dx, dy):
        new_x = self.x + dx
        new_y = self.y + dy

        # Check boundaries
        if new_x < 0 or new_x >= self.WIDTH or new_y < 0 or new_y >= self.HEIGHT:
            return False

        # Check if the new position collides with a wall
        if (new_x, new_y) in self.wall_positions:
            return False

        return True

    def move(self, dx, dy):
        if self.is_valid_move(dx, dy):
            self.x += dx
            self.y += dy
            
    def draw(self, screen, image):
        """Draws the player on the given screen using the provided image."""
        screen.blit(image, (self.x, self.y))

    # Additional methods related to rendering or any other operations...
