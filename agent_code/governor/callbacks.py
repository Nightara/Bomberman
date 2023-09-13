import numpy as np

from agent_code.rule_based_survivor.callbacks import setup as setup_survivor, next_move as next_move_survivor
from My_code.train_pytorch import CoinCollectorAgent

def setup(self):
    bomber = CoinCollectorAgent(13, 4)

    # Set up agents here
    self.setup_callbacks = [
        setup_survivor,
    ]

    # Agent weights
    self.weights = [
        1.0,
        1.0,
    ]

    # Insert all callbacks here
    self.sub_agents_callbacks = [
        lambda game_state: next_move_survivor(self, game_state),
        bomber.next_move,
    ]

    # Run all setup functions
    for setup_callback in self.setup_callbacks:
        setup_callback(self)


def act(self, game_state: dict):
    move_suggestions = [next_move(game_state) for next_move in self.sub_agents_callbacks]
    weighted_certainties = np.multiply([suggestion[1] for suggestion in move_suggestions], self.weights)
    max_index = np.argmax(weighted_certainties)

    return move_suggestions[max_index][0]
