from random import shuffle
import numpy as np


def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on, and they will be persistent even across multiple games.
    You can also use the `self.logger` object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug("Successfully entered setup code")


def next_move(self, game_state):
    """Returns the best move, but also this agent's confidence in its correctness.
    """
    self.logger.info("Picking action according to rule set")
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        self.current_round = game_state["round"]
        # Gather information about the game state
    arena = game_state["field"]
    _, _, _, (x, y) = game_state["self"]
    bombs = game_state["bombs"]
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state["others"]]
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state["explosion_map"][d] < 1) and
                (bomb_map[d] > 0) and
                (d not in others) and
                (d not in bomb_xys)):
            valid_tiles.append(d)

    if (x - 1, y) in valid_tiles:
        valid_actions.append("LEFT")
    if (x + 1, y) in valid_tiles:
        valid_actions.append("RIGHT")
    if (x, y - 1) in valid_tiles:
        valid_actions.append("UP")
    if (x, y + 1) in valid_tiles:
        valid_actions.append("DOWN")
    if (x, y) in valid_tiles:
        valid_actions.append("WAIT")

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = [("UP", 0.1), ("DOWN", 0.1), ("LEFT", 0.1), ("RIGHT", 0.1)]
    shuffle(action_ideas)

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        # Try random direction if directly on top of a bomb
        if xb == x and yb == y:
            action_ideas.append(("UP", 1.0))
            action_ideas.append(("DOWN", 1.0))
            action_ideas.append(("LEFT", 1.0))
            action_ideas.append(("RIGHT", 1.0))
        elif xb == x and abs(yb - y) < 4:
            # Run away
            if yb > y:
                action_ideas.append(("UP", 1.0))
            if yb < y:
                action_ideas.append(("DOWN", 1.0))
            # If possible, turn a corner
            action_ideas.append(("LEFT", 2.0))
            action_ideas.append(("RIGHT", 2.0))
        elif yb == y and abs(xb - x) < 4:
            # Run away
            if xb > x:
                action_ideas.append(("LEFT", 1.0))
            if xb < x:
                action_ideas.append(("RIGHT", 1.0))
            # If possible, turn a corner
            action_ideas.append(("UP", 1.0))
            action_ideas.append(("DOWN", 1.0))

    # Normalize confidence, then sort actions
    total_conf = sum(conf for (move, conf) in action_ideas)
    action_ideas = [(move, conf / total_conf) for (move, conf) in action_ideas]
    action_ideas.sort(key=lambda m: m[1])

    # Pick the highest rated action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            return a

    # Return the default move WAIT if nothing else works
    return "WAIT", 0.1


def act(self, game_state):
    """Wraps the next_move method to only return the move and drop the confidence score.
    """
    move, _ = next_move(self, game_state)
    return move
