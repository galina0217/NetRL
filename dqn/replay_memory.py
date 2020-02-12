"""Code from https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""

import os
import random
import logging
import numpy as np

from .utils import save_npy, load_npy

class ReplayMemory:
  def __init__(self, config, p_dim, model_dir):
    self.model_dir = model_dir

    self.p_dim = p_dim

    self.memory_size = config.memory_size
    self.states = np.empty((self.memory_size, config.n_features + config.n_embedding), dtype=np.float16)
    self.paths_s = np.empty((self.memory_size, p_dim), dtype=np.float16)
    self.actions = np.empty((self.memory_size, 1), dtype=np.integer)
    self.rewards = np.empty((self.memory_size, 1), dtype=np.float)
    self.states_ = np.empty((self.memory_size, config.n_features + config.n_embedding), dtype=np.float16)
    self.paths_s_ = np.empty((self.memory_size, p_dim), dtype=np.float16)
    self.terminals = np.empty((self.memory_size, 1), dtype=np.bool)
    self.history_length = config.history_length
    self.dims = (config.n_features + config.n_embedding)
    self.batch_size = int(config.batch_size * config.batch_ratio)
    self.count = 0
    self.current = 0

    memory_size_delay = int(self.memory_size / config.path_length)
    self.memory_size_delay = memory_size_delay
    self.states_delay = np.empty((memory_size_delay, config.n_features + config.n_embedding), dtype=np.float16)
    self.paths_delay_s = np.empty((memory_size_delay, p_dim), dtype=np.float16)
    self.actions_delay = np.empty((memory_size_delay, 1), dtype=np.integer)
    self.rewards_delay = np.empty((memory_size_delay, 1), dtype=np.float)
    self.states_delay_ = np.empty((memory_size_delay, config.n_features + config.n_embedding), dtype = np.float16)
    self.paths_delay_s_ = np.empty((memory_size_delay, p_dim), dtype=np.float16)
    self.terminals_delay = np.empty((memory_size_delay, 1), dtype=np.bool)
    self.batch_size_delay = config.batch_size - self.batch_size
    self.count_delay = 0
    self.current_delay = 0

    # pre-allocate prestates and poststates for minibatch
    self.prestates = np.empty((self.batch_size, self.dims), dtype=np.float16)
    self.poststates = np.empty((self.batch_size, self.dims), dtype=np.float16)

    # pre-allocate prestates and poststates for minibatch
    self.prestates_delay = np.empty((self.batch_size_delay, self.dims), dtype=np.float16)
    self.poststates_delay = np.empty((self.batch_size_delay, self.dims), dtype=np.float16)


  def add(self, state, reward, action, state_, terminal, path_s, path_s_):
    # assert state.shape == self.dims
    # NB! screen is post-state, after action and reward
    if terminal == False:
      self.states[self.current, ...] = state
      self.paths_s[self.current, ...] = path_s
      self.actions[self.current] = action
      self.rewards[self.current] = reward
      self.states_[self.current, ...] = state_
      self.paths_s_[self.current, ...] = path_s_
      # self.paths_a_[self.current, ...] = path_a_
      self.terminals[self.current] = terminal
      self.count = max(self.count, self.current + 1)
      self.current = (self.current + 1) % self.memory_size
      # print('\n' + str(self.count) +'\t'+ str(self.current))
    else:
      self.states_delay[self.current_delay, ...] = state
      self.paths_delay_s[self.current_delay, ...] = path_s
      self.actions_delay[self.current_delay] = action
      self.rewards_delay[self.current_delay] = reward
      self.states_delay_[self.current_delay, ...] = state_
      self.paths_delay_s_[self.current_delay, ...] = path_s_
      # self.paths_delay_a_[self.current_delay, ...] = path_a_
      self.terminals_delay[self.current_delay] = terminal
      self.count_delay = max(self.count_delay, self.current_delay + 1)
      self.current_delay = (self.current_delay + 1) % self.memory_size_delay

  def getState(self, index, type='pre', mode='immediate'):
    if mode == 'immediate':
      assert self.count > 0, "replay memory is empy, use at least --random_steps 1"
      # normalize index to expected range, allows negative indexes
      index = index % self.count
      # if is not in the beginning of matrix
      if index >= self.history_length - 1:
        # use faster slicing
        if type == 'pre':
          return self.states[(index - (self.history_length - 1)):(index + 1), ...]
        elif type == 'pos':
          return self.states_[(index - (self.history_length - 1)):(index + 1), ...]
      else:
        # otherwise normalize indexes and use slower list based access
        indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
        if type == 'pre':
          return self.states[indexes, ...]
        if type == 'pos':
          return self.states_[indexes, ...]
    elif mode == 'delay':
      assert self.count_delay > 0, "replay memory is empy, use at least --random_steps 1"
      # normalize index to expected range, allows negative indexes
      index = index % self.count_delay
      # if is not in the beginning of matrix
      if index >= self.history_length - 1:
        # use faster slicing
        if type == 'pre':
          return self.states_delay[(index - (self.history_length - 1)):(index + 1), ...]
        elif type == 'pos':
          return self.states_delay_[(index - (self.history_length - 1)):(index + 1), ...]
      else:
        # otherwise normalize indexes and use slower list based access
        indexes = [(index - i) % self.count_delay for i in reversed(range(self.history_length))]
        if type == 'pre':
          return self.states_delay[indexes, ...]
        if type == 'pos':
          return self.states_delay_[indexes, ...]

  def sample(self):
    # samples from immediate reward
    tran = np.hstack((self.states, self.actions, self.rewards, self.states_, self.terminals,
                      self.paths_s, self.paths_s_))[:self.count, :]
    tran_unique = list(set(map(lambda x: tuple(x), tran)))
    # if self.count < self.batch_size or len(tran_unique) < self.batch_size:
    if len(tran_unique) < self.batch_size:
      tran_unique = np.array(tran_unique)
      n_unrepeat = int(self.batch_size / len(tran_unique))
      tran_sample = np.vstack([tran_unique for i in range(n_unrepeat)])
      if self.batch_size - len(tran_sample) > 0:
        tran_sample = np.vstack((tran_sample, np.array(random.sample(
          tran.tolist(), self.batch_size-len(tran_sample)))))
    else:
      tran_sample = np.array(random.sample(tran_unique, self.batch_size))
    self.prestates = tran_sample[:, :self.dims]
    actions = tran_sample[:, self.dims:self.dims+1]
    rewards = tran_sample[:, self.dims+1:self.dims+2]
    self.poststates = tran_sample[:, self.dims+2:2*self.dims+2]
    terminals = tran_sample[:, 2*self.dims+2:2*self.dims+3]
    path_s = tran_sample[:, 2*self.dims+3:2*self.dims+3+self.p_dim]
    path_s_ = tran_sample[:, 2*self.dims+3+self.p_dim:]

    # samples from delay reward
    if self.batch_size_delay > 0:
      tran_delay = np.hstack((self.states_delay, self.actions_delay, self.rewards_delay,
                              self.states_delay_, self.terminals_delay,
                              self.paths_delay_s, self.paths_delay_s_))[:self.count_delay, :]
      tran_unique_delay = list(set(map(lambda x: tuple(x), tran_delay)))
      if len(tran_unique_delay) < self.batch_size_delay:
        tran_unique_delay = np.array(tran_unique_delay)
        n_unrepeat = int(self.batch_size_delay / len(tran_unique_delay))
        tran_delay_sample = np.vstack([tran_unique_delay for i in range(n_unrepeat)])
        if self.batch_size_delay - len(tran_delay_sample) > 0:
          tran_delay_sample = np.vstack((tran_delay_sample, np.array(random.sample(
            tran_delay.tolist(), self.batch_size_delay-len(tran_delay_sample)))))
      else:
        tran_delay_sample = np.array(random.sample(tran_unique_delay, self.batch_size_delay))
      self.prestates = np.vstack((self.prestates, tran_delay_sample[:, :self.dims]))
      actions = np.vstack((actions, tran_delay_sample[:, self.dims:self.dims+1]))
      rewards = np.vstack((rewards, tran_delay_sample[:, self.dims+1:self.dims+2]))
      self.poststates = np.vstack((self.poststates, tran_delay_sample[:, self.dims+2:2*self.dims+2]))
      terminals = np.vstack((terminals, tran_delay_sample[:, 2*self.dims+2:2*self.dims+3]))
      path_s = np.vstack((path_s, tran_delay_sample[:, 2 * self.dims + 3:2 * self.dims + 3 + self.p_dim]))
      path_s_ = np.vstack((path_s_, tran_delay_sample[:, 2 * self.dims + 3 + self.p_dim:]))

    shuffle_index = [i for i in range(len(actions))]
    random.shuffle(shuffle_index)
    actions = np.array([actions[i] for i in shuffle_index])
    rewards = np.array([rewards[i] for i in shuffle_index])
    terminals = np.array([terminals[i] for i in shuffle_index])
    self.prestates = np.array([self.prestates[i] for i in shuffle_index])
    self.poststates = np.array([self.poststates[i] for i in shuffle_index])
    path_s = np.array([path_s[i] for i in shuffle_index])
    path_s_ = np.array([path_s_[i] for i in shuffle_index])

    return self.prestates, np.squeeze(actions).astype(int), np.squeeze(rewards), self.poststates,\
           np.squeeze(terminals).astype(bool), path_s, path_s_

  def save(self):
    for idx, (name, array) in enumerate(
        zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
            [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
      save_npy(array, os.path.join(self.model_dir, name))

  def load(self):
    for idx, (name, array) in enumerate(
        zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
            [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
      array = load_npy(os.path.join(self.model_dir, name))
