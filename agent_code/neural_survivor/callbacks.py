import os

import numpy as np
import torch
import torch.nn as nn


def setup(self):
    """Loads the model from file.
    """
    self.moves = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT"]
    if os.path.isfile("model.dat"):
        self.logger.info("Loading network from file.")
        self.model = torch.load("model.dat")
    else:
        self.logger.info("Creating new network.")
        self.model = nn.Sequential(
            nn.Linear(25, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
        )


def gather_input(game_state):
    """Generates the NN input from the current game state.
    """
    arena = game_state["field"]
    _, _, _, (pos_x, pos_y) = game_state["self"]
    others = [xy for (n, s, b, xy) in game_state["others"]]
    for (x, y) in others:
        arena[x][y] = -1

    bomb_map = np.ones(arena.shape) * 5
    for (x, y), t in game_state["bombs"]:
        for (i, j) in [(x + h, y) for h in range(-3, 4)] + [(x, y + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    model_input = []
    for x in range(-2, 3):
        for y in range(-2, 3):
            try:
                d = (pos_x + x, pos_y + y)
                model_input.append(1 if bomb_map[d] == 0 or game_state["explosion_map"][d] < 1 else arena[d])
            except IndexError:
                model_input.append(-1)

    return torch.tensor(model_input, dtype=torch.float32)


def next_move(self, game_state):
    """Returns the best move, but also this agent's confidence in its correctness.
    """
    results = self.model(gather_input(game_state))
    best_move = torch.tensor(0.1), 4
    for index, confidence in enumerate(results):
        if confidence.item() > best_move[0].item():
            best_move = confidence, index

    return self.moves[best_move[1]], best_move[0].item()


def act(self, game_state):
    """Wraps the next_move method to only return the move and drop the confidence score.
    """
    move, _ = next_move(self, game_state)
    return move
