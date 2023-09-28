import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from random import shuffle

from .Model import Bombing
from .Bombing import state_to_features
from .BombingRuleBased import act_rulebased, initialize_rule_based

import events as e

PARAMETERS = 'last_save' #select parameter_set stored in network_parameters/

ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']


def setup(self):
    self.network = Bombing()

    if self.train:
        self.logger.info("Training a new model.")
    else:
        self.logger.info(f"Loading model '{PARAMETERS}'.")
        filename = os.path.join("network_parameters", f'{PARAMETERS}.pt')
        self.network.load_state_dict(torch.load(filename))
        self.network.eval()
    
    initialize_rule_based(self)

    self.bomb_buffer = 0

def act(self, game_state: dict) -> str:
    if game_state is None:
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    features = state_to_features(self, game_state)
    Q = self.network(features)

    if self.train:
        eps = self.epsilon_arr[self.episode_counter]
        if random.random() <= eps:
            if eps > 0.1 and np.random.randint(4) == 0:
                action = act_rulebased(self)
                self.logger.info(f"Choosing action {action} based on rule-based agent.")
                return action
            else:
                action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
                self.logger.info(f"Choosing action {action} completely randomly.")
                return action

    action_prob = np.array(torch.softmax(Q, dim=1).detach().squeeze())
    best_action = ACTIONS[np.argmax(action_prob)]
    self.logger.info(f"test:{np.argmax(action_prob)}")
    self.logger.info(f"Choosing action {best_action} based on the Q-function.")

    return best_action#,np.argmax(action_prob)/10

def action(self, game_state: dict) -> str:
    if game_state is None:
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    features = state_to_features(self, game_state)
    Q = self.network(features)

    if self.train:
        eps = self.epsilon_arr[self.episode_counter]
        if random.random() <= eps:
            if eps > 0.1 and np.random.randint(4) == 0:
                action = act_rulebased(self)
                self.logger.info(f"Choosing action {action} based on rule-based agent.")
                return action
            else:
                action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
                self.logger.info(f"Choosing action {action} completely randomly.")
                return action

    action_prob = np.array(torch.softmax(Q, dim=1).detach().squeeze())
    best_action = ACTIONS[np.argmax(action_prob)]
    self.logger.info(f"test:{np.argmax(action_prob)}")
    self.logger.info(f"Choosing action {best_action} based on the Q-function.")

    return best_action,np.argmax(action_prob)/10