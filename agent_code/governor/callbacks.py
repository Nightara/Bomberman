import numpy as np

def setup(self):
    # Set up agents here
    self.setup_callbacks = [

    ]

    # Agent weights
    self.weights = [

    ]

    # Insert all callbacks here
    self.sub_agents_callbacks = [

    ]

    # Run all setup functions
    for setup_callback in self.setup_callbacks:
        setup_callback(self)


def act(self, game_state: dict):
    move_suggestions = [next_move(self, game_state) for next_move in self.sub_agents_callbacks]
    weighted_certainties = np.multiply([suggestion[1] for suggestion in move_suggestions], self.weights)
    max_index = np.argmax(weighted_certainties)

    return move_suggestions[max_index][0]
