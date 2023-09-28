import random
import numpy as np
import torch 
import copy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import events as e
from scipy.ndimage.filters import uniform_filter1d

from .Bombing import state_to_features
    
ACTIONS_IDX = {'LEFT':0, 'RIGHT':1, 'UP':2, 'DOWN':3, 'WAIT':4, 'BOMB':5}

def reward_from_events(self, events) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.KILLED_OPPONENT: 500,
        e.MOVED_RIGHT: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -1,
        e.INVALID_ACTION: -10,
        e.BOMB_DROPPED: -1,
        e.KILLED_SELF: 0,
        e.GOT_KILLED: -700,
    }

    reward_sum = sum(game_rewards.get(event, 0) for event in events)
    return reward_sum

def rewards_from_own_events(self, events):
    '''
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param events: pre defined events (events.py) that occured in game step

    return: reward based on own events
    '''
    reward_sum = crate_rewards(self, events)
    self.logger.info(f"Awarded {reward_sum} for own transition events")
    return reward_sum

def crate_rewards(self, events):
    '''
    Give the crate rewards imediately after placing the crates. This makes our agent place bombs more often (better)

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param events: pre defined events (events.py) that occured in game step

    return: reward based on how many crates will be destroyed by dropped bomb
    '''
    if e.BOMB_DROPPED in events:
        reward = self.destroyed_crates * 33
        self.logger.info(f"reward for the {self.destroyed_crates} that are going to be destroyed -> +{reward}")
        return reward
    return 0

def generate_eps_greedy_policy(network, q):
    '''
    :param network: the network that is used for training 
            (contains eps-threshold for start and end of training 
            and the number of total episodes)
    :param q: the fraction of the training where the eps-threshold is linearly diminished

    returns: array containing the eps-thresholds for training
    '''
    N = network.training_episodes
    N_1 = int(N*q)
    N_2 = N - N_1
    eps1 = np.linspace(network.epsilon_begin, network.epsilon_end, N_1)
    if N_1 == N:
        return eps1
    eps2 = np.ones(N_2) * network.epsilon_end
    return np.concatenate((eps1, eps2))

def add_experience(self, old_game_state, action, new_game_state, events, n=5):
    '''
    fills the experience buffer

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: dict contoining the old game state
    :param self_action: action executed by the agent
    :param new_game_state: dict contoining the new game state
    :param events: events that occured 
    :param n: n-step Q-learning, default 5
    '''
    old_features = state_to_features(self, old_game_state)

    if old_features is None:
        return

    if new_game_state is None:
        new_features = old_features
    else:
        new_features = state_to_features(self, new_game_state)

    reward = reward_from_events(self, events) + rewards_from_own_events(self, events)

    action_idx = ACTIONS_IDX[action]
    action_hot_one = torch.zeros(6)
    action_hot_one[action_idx] = 1

    add_remaining_experience(self, events, reward, new_features, n)

    self.experience_buffer.append((old_features, action_hot_one, reward, new_features, 0))

    while len(self.experience_buffer) > self.network.buffer_size:
        self.experience_buffer.pop(0)

def add_remaining_experience(self, events, new_reward, new_features, n):
    '''
    subsequently update the reward of the last n steps (n step TD Q-learning)

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param events: events that occured 
    :param new_reward: reward of current step
    :param new_features: result of the state to features function
    :param n: n-step Q-learning, default 5
    '''
    steps_back = min(len(self.experience_buffer), n)
    for i in range(1, steps_back+1):
        old_old_features, action, reward, old_new_features, number_of_additional_rewards = self.experience_buffer[-i]
        reward += (self.network.gamma**i)*new_reward
        number_of_additional_rewards += 1
        assert number_of_additional_rewards == i
        self.experience_buffer[-i] = (old_old_features, action, reward, new_features, number_of_additional_rewards)


def train_network(self):
    '''
    updates the parameters of the double network that is used in the training

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    '''
    new_network = self.new_network  # apply updates to
    old_network = self.network      # used in training, used to calculate Y
    experience_buffer = self.experience_buffer
    new_network.device = "cpu"

    # randomly choose batch out of the experience buffer
    number_of_elements_in_buffer = len(experience_buffer)
    batch_size = min(number_of_elements_in_buffer, old_network.batch_size)
    random_i = [random.randrange(number_of_elements_in_buffer) for _ in range(batch_size)]

    # Compute for each experience in the batch 
    # - the Ys using n-step TD Q-learning
    # - the current guess for the Q function
    sub_batch = [experience_buffer[i] for i in random_i]
    Y = []

    for b in sub_batch:
        old_features, action, reward, new_features, number_of_additional_rewards = b

        y = reward
        if new_features is not None:
            with torch.no_grad():
                y += old_network.gamma**(number_of_additional_rewards+1) * torch.max(old_network(new_features)).item()

        Y.append(y)

    Y = torch.tensor(Y, device=new_network.device)

    # Qs
    states = torch.cat([b[0] for b in sub_batch], dim=0)  # Put all states of the sub_batch in one batch
    q_values = new_network(states)
    actions = torch.cat([b[1].unsqueeze(0) for b in sub_batch])
    Q = torch.sum(q_values*actions, dim=1)

    # Prioritized experience replay => choose 50 batch entries with highest residuals
    Residuals = torch.abs(Y-Q)
    batch_size = min(len(Residuals), 50)
    _, indices = torch.topk(Residuals, batch_size)

    Y_reduced = Y[indices]
    Q_reduced = Q[indices]

    # Calc the loss and update parameters
    loss = new_network.loss_function(Q_reduced, Y_reduced)
    new_network.optimizer.zero_grad()
    loss.backward()
    new_network.optimizer.step()


def update_network(self):
    '''
    overwrite the old network with the double network
    (updates only the parameters)

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    '''
    self.network = copy.deepcopy(self.new_network)




def save_parameters(self, string):
    '''
    saves the network parameters of the current network

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param string: file name
    '''
    torch.save(self.network.state_dict(), f"network_parameters/{string}.pt")


def get_score(events):
    '''
    tracks the true score

    :param events: events that occured in game step
    '''
    true_game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
    }
    score = 0
    for event in events:
        if event in true_game_rewards:
            score += true_game_rewards[event]
    return score


def track_game_score(self, smooth=False):
    '''
    Plot our gamescore -> helpful to see if our training is working without much time effort

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param smooth: calculate running mean if smooth==True, default: False
    '''

    self.game_score_arr.append(self.game_score)
    self.game_score = 0

    y = self.game_score_arr
    if smooth:
        window_size = self.total_episodes // 25
        if window_size < 1:
            window_size = 1
        y = uniform_filter1d(y, window_size, mode="nearest", output="float")
    x = range(len(y))

    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_title('score during training', fontsize=35, fontweight='bold')
    ax.set_xlabel('episode', fontsize=25, fontweight='bold')
    ax.set_ylabel('points', fontsize=25, fontweight='bold')
    ax.grid(axis='y', alpha=0.2, color='gray', zorder=-1)
    # ax.set_yticks(range(255)[::10])
    ax.set_yticks(range(255))
    ax.tick_params(labelsize=16)

    ax.plot(x,y,color='gray',linewidth=0.5, alpha=0.7, zorder=0)

    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["red","darkorange","green"])
    ax.scatter(x,y,c=y,cmap=cmap,s=40, alpha=0.5, zorder=1)
    try:
        plt.savefig('training_progress.png')
    except:
        ...
    plt.close()


