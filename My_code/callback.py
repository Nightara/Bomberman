class Player:
    def __init__(self, x, y, wall_positions, crate_positions, WIDTH, HEIGHT):
        self.x = x
        self.y = y
        self.prev_position = (x, y)  # Initialize the prev_position to the starting position
        self.wall_positions = wall_positions
        self.crate_positions = crate_positions
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

        # Check if the new position collides with a wall or crate
        if (new_x, new_y) in self.wall_positions or (new_x, new_y) in self.crate_positions:
            return False

        return True

    def move(self, dx, dy):
        if self.is_valid_move(dx, dy):
            # Update prev_position before making the move
            self.prev_position = (self.x, self.y)
            self.x += dx
            self.y += dy

    # Additional methods related to rendering or any other operations...
