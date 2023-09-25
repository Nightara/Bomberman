import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent_code.neural_survivor.callbacks import gather_input


translate_move = {
    "UP": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
    "DOWN": np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
    "LEFT": np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
    "RIGHT": np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
    "WAIT": np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
}


def setup_training(self):
    """Prepares the model for training.
    """
    self.cache = {
        "inputs": [],
        "labels": [],
    }
    self.loss_fn = nn.BCELoss()
    self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)


def game_events_occurred(self, old_game_state, self_action, new_game_state, event):
    self.cache["inputs"].append(gather_input(old_game_state))
    move = trainer_move(old_game_state)
    self.cache["labels"].append(translate_move[move[0]] * move[1])


def end_of_round(self, last_game_state, last_action, event):
    game_events_occurred(self, last_game_state, last_action, None, event)
    loss = self.loss_fn(self.model(torch.stack(self.cache["inputs"])), torch.tensor(np.stack(self.cache["labels"]), dtype=torch.float32))
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    torch.save(self.model, "model.dat")


def trainer_move(game_state):
    """Returns the best move, but also this agent's confidence in its correctness.
    """
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
    action_ideas = []

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
            action_ideas.append(("UP", 2.0))
            action_ideas.append(("DOWN", 2.0))

    # Normalize confidence, then sort actions
    if len(action_ideas) > 0:
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
