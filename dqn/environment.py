
import random
import numpy as np
import pandas as pd

class OurEnvironment():
  def __init__(self, config):
    self.config = config
    self.n_actions, self.n_features = config.n_actions, config.n_features

    self.display = config.display
    self.dims = (self.n_features)

    self._screen = None
    self.reward = 0
    self.terminal = True

  def load_data(self):
    self.edge_list = []
    with open(self.config.file_edgelist, 'r') as f:
      for l in f:
        self.edge_list.append([int(l.strip().split()[0]), int(l.strip().split()[1])])

  def act(self, state, action, path, label, edgelist, is_training=True):
    self.state_, self.reward, self.terminal = self.env_step(state, action, path, label, edgelist)
    return self.state_, self.reward, self.terminal

  def new_random_game(self):
    self.state = random.randint(0, self.n_actions-1)
    return self.state

  def env_step(self, state, action, path, label, edge_list):
    observation_ = action
    if label[state] == -1 or label[action] == -1:
      reward_label = self.config.immediate_reward_test
    else:
      reward_label = 1 if (label[state] == label[action]) else -1

    if [state, action] in edge_list or [action, state] in edge_list:
      reward_network = 1
    else:
      reward_network = -1

    reward = self.config.beta * reward_label + (1-self.config.beta) * 10 * reward_network

    terminal = True if (len(path) == self.config.path_length) else False
    return observation_, reward, terminal
