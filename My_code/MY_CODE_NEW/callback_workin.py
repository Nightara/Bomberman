class Player:
    
    def __init__(self, x, y, tile_size, wall_positions, crate_positions, width, height):
        self.x = x
        self.y = y
        self.speed = tile_size
        self.wall_positions = wall_positions
        self.crate_positions = crate_positions
        self.width = width
        self.height = height

 # Assuming a TILE_SIZE of 40, update as needed

    def move(self, dx, dy):
        new_x = self.x + dx * self.speed
        new_y = self.y + dy * self.speed

        # Check boundaries
        if new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height:
            return

        # Check walls and crates
        if (new_x, new_y) in self.wall_positions or (new_x, new_y) in self.crate_positions:
            return

        self.x = new_x
        self.y = new_y