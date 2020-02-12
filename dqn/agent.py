# -*- coding: UTF-8 -*- 
from __future__ import print_function
import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
import sys
import math
from .base import BaseModel
from .history import History
from .replay_memory import ReplayMemory
from .ops import linear, conv2d, clipped_error
from .utils import get_time, save_pkl, load_pkl
import copy
from scipy.sparse.linalg import svds, eigs
import gc

from predictor.utils import *
from predictor.train import GraphConvolutionalNetwork
# from predictor_binary.utils import *
# from predictor_binary.train import GraphConvolutionalNetwork

class Agent(BaseModel):
  def __init__(self, config, environment, sess):
    super(Agent, self).__init__(config)

    self.sess = sess
    self.weight_dir = 'weights'

    self.pad_vector = [0 for i in range(self.config.n_embedding)]

    self.row_diag = [i for i in range(self.config.n_actions)]
    self.col_diag = [i for i in range(self.config.n_actions)]

    self.env = environment

    with tf.variable_scope('step'):
      self.step_op = tf.Variable(0, trainable=False, name='step')
      self.step_input = tf.placeholder('int32', None, name='step_input')
      self.step_assign_op = self.step_op.assign(self.step_input)

    self.load_data()
    self.path_generator()
    self.build_dqn()
    self.model_gcn = GraphConvolutionalNetwork(self.config.mode, self.config.env_name, self.features, self.label,
                                               self.n_actions, self.n_features, model='gcn',
                                               trainsize=0.1)

    self.memory = ReplayMemory(self.config, self.p_dim, self.model_dir)

  def train(self):
    start_step = self.step_op.eval()
    num_game, self.update_count, ep_reward = 0, 0, 0.   # num_game没用
    total_reward, self.total_loss, self.total_q = 0., 0., 0.  # total和test_step相关
    max_avg_ep_reward = 0
    ep_rewards, actions = [], []
    path = []

    state = self.env.new_random_game()
    path.append(state)

    for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
      if self.step == self.learn_start:
        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        ep_rewards, actions = [], []

      # 1. predict
      action = self.predict(self.feature_embed_dict[state], state, path)
      path.append(action)
      # 2. act
      state_, reward, terminal = self.env.act(state, action, path, self.y_train, self.edge_list, is_training=True)
      # 3. observe
      if terminal:
        if self.config.delay == True:
          reward_d, macro_f1_train, micro_p, macro_p, micro_r, macro_r, micro_f, macro_f = self.delayed_reward(path)
        else:
          reward_d = 0
        reward += reward_d
        self.observe(self.feature_embed_dict[state], reward, action, terminal,
                     self.feature_embed_dict[state_], path)
        ep_reward += reward
        state_ = self.env.new_random_game()
        path = [state_]

        num_game += 1
        ep_rewards.append(ep_reward)
        ep_reward = 0.
      else:
        self.observe(self.feature_embed_dict[state], reward, action, terminal,
                     self.feature_embed_dict[state_], path)
        ep_reward += reward

      actions.append(action)
      total_reward += reward
      state = state_
      # print(reward)

      if self.step >= self.learn_start:
        if self.step % self.test_step == self.test_step - 1:
          macro_f1_train, micro_p, macro_p, micro_r, macro_r, micro_f, macro_f = self.inference_path()

          avg_reward = total_reward / self.test_step
          avg_loss = self.total_loss / self.update_count
          avg_q = self.total_q / self.update_count

          try:
            max_ep_reward = np.max(ep_rewards)
            min_ep_reward = np.min(ep_rewards)
            avg_ep_reward = np.mean(ep_rewards)
          except:
            max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

          if self.step > 30000:
              self.step_assign_op.eval({self.step_input: self.step + 1})
              self.save_model(self.step + 1)

          #   max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

          if self.step > 180:
            self.inject_summary({
              'average.reward': avg_reward,
              'average.loss': avg_loss,
              'average.q': avg_q,
              'episode.max reward': max_ep_reward,
              'episode.min reward': min_ep_reward,
              'episode.avg reward': avg_ep_reward,
              # 'ep': self.ep,
              'episode.rewards': ep_rewards,
              'episode.actions': actions,
              'episode.train macro f1': macro_f1_train,
              'episode.test macro precision': macro_p,
              'episode.test macro recall': macro_r,
              'episode.test macro f1': macro_f,
              'training.learning_rate': self.learning_rate_op.eval({self.learning_rate_step: self.step}),
            }, self.step)


          num_game = 0
          total_reward = 0.
          self.total_loss = 0.
          self.total_q = 0.
          self.update_count = 0
          ep_reward = 0.
          ep_rewards = []
          actions = []


  def inference_path(self):

    if self.config.infer_mode == 'path':
      paths = []
      for n in range(self.config.n_actions):
        state = n
        p = [state]
        for i in range(3):
          action = self.inference_predict(self.feature_embed_dict[state], state, p, self.config.infer_mode)
          p.append(action)

          state = action
        paths.append(p)
    elif self.config.infer_mode == 'top':
      paths = []
      for state in range(self.config.n_actions):
        action = self.inference_predict(self.feature_embed_dict[state], state, [], self.config.infer_mode)
        for i in range(len(action)):
          paths.append([state, action[i]])
    elif self.config.infer_mode == 'local':
      paths = []
      for state in range(self.config.n_actions):
        action = self.inference_predict(self.feature_embed_dict[state], state, [], self.config.infer_mode)
        for i in range(len(action)):
          paths.append([state, action[i]])
    elif self.config.infer_mode == 'global':
      q_table = []
      for s_i in range(self.config.n_actions):
          a_range = list(self.dist_indices[s_i][1:])
          a_t_batch = [self.feature_embed_dict[a] for a in a_range]
          s_t_batch = [self.feature_embed_dict[s_i] for i in range(len(a_range))]
          q = list(np.array(self.q.eval({self.s_t: s_t_batch, self.a_t: a_t_batch})).flatten())
          q_table.append(q)
      q_table = np.array(q_table).flatten()
      q_max = max(q_table)
      q_min = min(q_table)
      self.q_choose = q_max-self.config.n_percent*(q_max-q_min)

      paths = []
      for state in range(self.config.n_actions):
        action = self.inference_predict(self.feature_embed_dict[state], state, [], self.config.infer_mode)
        for i in range(len(action)):
          paths.append([state, action[i]])
    elif self.config.infer_mode == 'path_local':
      paths = []
      for n in range(self.config.n_actions):
        state = n
        p = [state]
        for i in range(3):
          action_max, action = self.inference_predict(self.feature_embed_dict[state], state,
                                                      p, self.config.infer_mode)
          for i in range(len(action)):
            paths.append([state, action[i]])
          p.append(action_max)
          state = action_max
      # self.save_path(paths, self.step, self.config.infer_mode)
      # reward_d, macro_f1_train, micro_p, macro_p, micro_r, macro_r, micro_f, macro_f = \
      #   self.delayed_reward(paths,mode='inference')

    if self.step > 20000:
      self.save_path(paths, self.step, self.config.infer_mode)

    reward_d, macro_f1_train, micro_p, macro_p, micro_r, macro_r, micro_f, macro_f = \
      self.delayed_reward(paths, mode='inference')
    return macro_f1_train, micro_p, macro_p, micro_r, macro_r, micro_f, macro_f

  def gen_path(self, path):
    path = path[-self.config.path_history:]
    path = [self.embed_dict[i] for i in path]
    path = [self.pad_vector] * (self.config.path_history - len(path)) + path

    p_o = self.p_output.eval({self.p: [path]})
    return np.squeeze(p_o)

  def predict(self, s_t, s_i, path):
    if self.config.policy == 'stochastic':
      action_range = list(self.dist_indices[s_i][1:])
      a_t_batch = [self.feature_embed_dict[a] for a in action_range]
      s_t_batch = [s_t for i in range(len(action_range))]
      p_vector = self.gen_path(path[:-1])
      path_t_batch = [p_vector for i in range(len(action_range))]
      q_action = np.array(self.q.eval({self.s_t: s_t_batch,
                                       self.a_t: a_t_batch,
                                       self.path_s_t: path_t_batch})).flatten()
      # stochastic policy
      q_action = np.exp(q_action) / np.sum(np.exp(q_action), axis=0)  # softmax
      index = np.random.choice(range(len(q_action)), p=q_action.ravel())  # 根据概率来选 action
      action = action_range[index]
    elif self.config.policy == 'deterministic':
      if self.config.ep_greedy == True:
      # self.ep = test_ep or (self.ep_end +
      #     max(0., (self.ep_start - self.ep_end) * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))
        if self.step == 0:
          self.ep = 1
        else:
          self.ep = self.ep / (math.pow(self.step, 1.0 / self.config.ep_coe))
      else:
        self.ep = 1

      action_range = list(self.dist_indices[s_i][1:])
      if random.random() < self.ep:
        action = random.sample(action_range, 1)[0]
      else:
        a_t_batch = [self.feature_embed_dict[a] for a in action_range]
        s_t_batch = [s_t for i in range(len(action_range))]
        p_vector = self.gen_path(path[:-1])
        path_t_batch = [p_vector for i in range(len(action_range))]
        q_action = np.array(self.q.eval({self.s_t: s_t_batch,
                                         self.a_t: a_t_batch,
                                         self.path_s_t: path_t_batch})).flatten()

        # if there exists multiple maximal values, random pick one
        index = random.choice(np.where(q_action == max(q_action))[0])
        action = action_range[index]

        # repeated case
        if action in path:
            action = random.sample(action_range, 1)[0]

    return action

  def inference_predict(self, s_t, s_i, p, mode):
    action_range = list(self.dist_indices[s_i][1:])
    a_t_batch = [self.feature_embed_dict[a] for a in action_range]
    s_t_batch = [s_t for i in range(len(action_range))]
    p_vector = self.gen_path(p[:-1])
    path_t_batch = [p_vector for i in range(len(action_range))]
    q_action = np.array(self.q.eval({self.s_t: s_t_batch,
                                     self.a_t: a_t_batch,
                                     self.path_s_t: path_t_batch})).flatten()

    if mode == 'path':
      q_action = np.array(q_action)
      index = random.choice(np.where(q_action == max(q_action))[0])
      action = action_range[index]
    elif mode == 'top':
      action_index = sorted(range(len(q_action)), key=lambda i: q_action[i])[-self.config.n_max:]
      action = [action_range[i] for i in action_index]
    elif mode == 'local':
      self.q_choose = max(q_action) - self.config.n_percent * (max(q_action)-min(q_action))
      action_index = np.where(q_action > self.q_choose)[0].tolist()
      action = [action_range[i] for i in action_index]
    elif mode == 'global':
      action_index = np.where(q_action > self.q_choose)[0].tolist()
      action = [action_range[i] for i in action_index]
    elif mode == 'path_local':
      q_action = np.array(q_action)
      index = random.choice(np.where(q_action == max(q_action))[0])
      action_max = action_range[index]

      self.q_choose = max(q_action) - self.config.n_percent * (max(q_action)-min(q_action))
      action_index = np.where(q_action > self.q_choose)[0].tolist()
      action = [action_range[i] for i in action_index]
      return action_max, action

    return action

  def observe(self, state, reward, action, terminal, state_, path):
    # reward = max(self.min_reward, min(self.max_reward, reward))

    path_s = self.gen_path(path[:-2])
    path_s_ = self.gen_path(path[:-1])
    self.memory.add(state, reward, action, state_, terminal, path_s, path_s_)

    if self.step > self.learn_start:
      if self.step % self.train_frequency == 0:
        self.q_learning_mini_batch()

      if self.step % self.target_q_update_step == self.target_q_update_step - 1:
        self.update_target_q_network()

  def predict_action(self, s_t, s_i, p_t):
    action_range = list(self.dist_indices[s_i][1:])
    a_t_batch = [self.feature_embed_dict[a] for a in action_range]
    s_t_batch = [s_t for i in range(len(action_range))]
    p_t_batch = [p_t for i in range(len(action_range))]
    q_action = np.array(self.q.eval({self.s_t: s_t_batch,
                                     self.a_t: a_t_batch,
                                     self.path_s_t: p_t_batch})).flatten()

    # if there exists multiple maximal values, random pick one
    index = random.choice(np.where(q_action == max(q_action))[0])
    action = action_range[index]
    return action

  def q_learning_mini_batch(self):
    if self.memory.count < self.history_length:
      return
    else:
      s_t, action, reward, s_t_plus_1, terminal, path_s_t, path_s_t_plus_1 = self.memory.sample()

    if self.double_q:
      pred_action = [self.predict_action(s_t_plus_1[i], action[i], path_s_t_plus_1[i])
                     for i in range(self.config.batch_size)]
      q_t_plus_1_with_pred_action = self.target_q.eval({self.target_s_t: s_t_plus_1,
                                                        self.target_a_t: [self.feature_embed_dict[a_i]
                                                                          for a_i in pred_action],
                                                        self.target_path_s_t: path_s_t_plus_1})
      target_q_t = (1. - terminal) * self.discount * np.squeeze(q_t_plus_1_with_pred_action) + reward
    else:
      terminal = np.array(terminal) + 0.
      action_range = np.array([self.dist_indices[i][1:] for i in action])
      action_batch = [self.feature_embed_dict[a] for a in action_range.flatten()]
      s_t_plus_1_batch = [si for si in s_t_plus_1 for i in range(action_range.shape[1])]
      path_s_t_plus_1_batch = [pi for pi in path_s_t_plus_1 for i in range(action_range.shape[1])]

      q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1_batch,
                                       self.target_a_t: action_batch,
                                       self.target_path_s_t: path_s_t_plus_1_batch})

      q_t_plus_1 = q_t_plus_1.reshape((self.config.batch_size, -1))
      max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
      target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward

    _, q_t, loss, delta = \
      self.sess.run([self.optim, self.q, self.loss, self.delta], {
        self.target_q_t: target_q_t,
        self.s_t: s_t,
        self.path_s_t: path_s_t,
        self.a_t: s_t_plus_1,   # at = st+1
        self.learning_rate_step: self.step,
      })

    self.total_loss += loss
    self.total_q += q_t.mean()
    self.update_count += 1

  def path_generator(self):
    with tf.variable_scope('path'):
      self.p = tf.placeholder('float32', [None, self.config.path_history, self.n_embedding], name='path_s_t')
      self.p_t = tf.transpose(self.p, [0, 2, 1])
      self.maxp_p_t = tf.layers.max_pooling1d(inputs=self.p_t,
                                              pool_size=int(self.n_embedding / 4),
                                              strides=int(self.n_embedding / 4))
      self.maxp_p_t_t = tf.transpose(self.maxp_p_t, [0, 2, 1])
      self.maxp_p_t_t_flatten = tf.contrib.layers.flatten(self.maxp_p_t_t)

      self.meanp_p_t = tf.layers.average_pooling1d(inputs=self.p_t,
                                                   pool_size=int(self.n_embedding / 4),
                                                   strides=int(self.n_embedding / 4))
      self.meanp_p_t_t = tf.transpose(self.meanp_p_t, [0, 2, 1])
      self.meanp_p_t_t_flatten = tf.contrib.layers.flatten(self.meanp_p_t_t)

      self.p_output = tf.concat([self.maxp_p_t_t_flatten, self.meanp_p_t_t_flatten], 1)

    self.p_dim = int(self.p_output.get_shape()[-1])
    # print('pdim:',self.p_dim)

  def build_dqn(self):
    self.w = {}
    self.t_w = {}

    activation_fn = None

    # training network
    with tf.variable_scope('prediction'):
      self.s_t = tf.placeholder('float32', [None, self.n_features + self.n_embedding], name='s_t')
      self.a_t = tf.placeholder('float32', [None, self.n_features + self.n_embedding], name='a_t')
      self.path_s_t = tf.placeholder('float32', [None, self.p_dim], name='path_s_t')

      self.input_s_t = tf.concat([self.s_t, self.path_s_t], 1)
      self.input = tf.concat([self.input_s_t, self.a_t], 1)

      if self.config.n_layers == 2:

        self.hidden_layer, self.w['q_w1'], self.w['q_b1'] = \
          linear(self.input, 50, activation_fn=tf.nn.relu, name='hidden')

        self.q, self.w['q_w2'], self.w['q_b2'] = \
          linear(self.hidden_layer, 1, activation_fn=activation_fn, name='q')

      elif self.config.n_layers == 3:

        self.hidden_layer1, self.w['q_w1'], self.w['q_b1'] = \
          linear(self.input, 50, activation_fn=tf.nn.relu, name='hidden1')

        self.hidden_layer2, self.w['q_w2'], self.w['q_b2'] = \
          linear(self.hidden_layer1, 10, activation_fn=tf.nn.relu, name='hidden2')

        self.q, self.w['q_w3'], self.w['q_b3'] = \
          linear(self.hidden_layer2, 1, activation_fn=activation_fn, name='q')

      elif self.config.n_layers == 1:

        self.q, self.w['q_w'], self.w['q_b'] = \
          linear(self.input, 1, activation_fn=activation_fn, name='q')

    # target network
    with tf.variable_scope('target'):
      self.target_s_t = tf.placeholder('float32', [None, self.n_features + self.n_embedding], name='target_s_t')
      self.target_a_t = tf.placeholder('float32', [None, self.n_features + self.n_embedding], name='target_a_t')
      self.target_path_s_t = tf.placeholder('float32', [None, self.p_dim], name='target_path_s_t')

      self.target_input_s_t = tf.concat([self.target_s_t, self.target_path_s_t], 1)
      self.target_input = tf.concat([self.target_input_s_t, self.target_a_t], 1)

      if self.config.n_layers == 2:

        self.target_hidden_layer, self.t_w['q_w1'], self.t_w['q_b1'] = \
              linear(self.target_input, 50, activation_fn=tf.nn.relu, name='target_hidden')

        self.target_q, self.t_w['q_w2'], self.t_w['q_b2'] = \
              linear(self.target_hidden_layer, 1, activation_fn=activation_fn, name='target_q')

      elif self.config.n_layers == 3:

        self.target_hidden_layer1, self.t_w['q_w1'], self.t_w['q_b1'] = \
          linear(self.target_input, 50, activation_fn=tf.nn.relu, name='target_hidden1')

        self.target_hidden_layer2, self.t_w['q_w2'], self.t_w['q_b2'] = \
          linear(self.target_hidden_layer1, 10, activation_fn=tf.nn.relu, name='target_hidden2')

        self.target_q, self.t_w['q_w3'], self.t_w['q_b3'] = \
          linear(self.target_hidden_layer2, 1, activation_fn=activation_fn, name='target_q')

      elif self.config.n_layers == 1:

        self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
          linear(self.target_input, 1, activation_fn=activation_fn, name='target_q')

    with tf.variable_scope('pred_to_target'):
      self.t_w_input = {}
      self.t_w_assign_op = {}

      for name in self.w.keys():
        self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
        self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

    # optimizer
    with tf.variable_scope('optimizer'):
      self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')

      self.delta = self.target_q_t - tf.squeeze(self.q, axis=-1)

      self.global_step = tf.Variable(0, trainable=False)

      self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')
      self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
      self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
          tf.train.exponential_decay(
              self.learning_rate,
              self.learning_rate_step,
              self.learning_rate_decay_step,
              self.learning_rate_decay,
              staircase=True))

      self.optim = tf.train.AdamOptimizer(
          self.learning_rate_op).minimize(self.loss)

    with tf.variable_scope('summary'):
      scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', 'episode.max reward', 'episode.min reward',
                             'episode.avg reward', 'training.learning_rate', 'episode.train macro f1',
                             'episode.test macro precision', 'episode.test macro recall', 'episode.test macro f1']

      self.summary_placeholders = {}
      self.summary_ops = {}

      for tag in scalar_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag] = tf.summary.scalar("%s-%s/%s" % (self.env_name, self.env_type, tag), self.summary_placeholders[tag])

      histogram_summary_tags = ['episode.rewards', 'episode.actions']

      for tag in histogram_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag] = tf.summary.histogram(tag, self.summary_placeholders[tag])

      self.writer = tf.summary.FileWriter('./logs/%s' % self.model_dir, self.sess.graph)

    tf.initialize_all_variables().run()

    self._saver = tf.train.Saver(list(self.w.values()) + [self.step_op], max_to_keep=30)

    self.update_target_q_network()

  def update_target_q_network(self):
    for name in self.w.keys():
      self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

  def inject_summary(self, tag_dict, step):
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, self.step)

  def load_data(self):
    print('Start loading data----')
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
      with open("data/ind.{}.{}".format(self.config.env_name, names[i]), 'rb') as f:
        if sys.version_info > (3, 0):
          objects.append(pkl.load(f, encoding='latin1'))
        else:
          objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(self.config.env_name))
    test_idx_range = np.sort(test_idx_reorder)

    if self.config.env_name == 'citeseer':
      # Fix citeseer dataset (there are some isolated nodes in the graph)
      # Find isolated nodes, add them as zero-vecs into the right position
      test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
      tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
      tx_extended[test_idx_range - min(test_idx_range), :] = tx
      tx = tx_extended
      ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
      ty_extended[test_idx_range - min(test_idx_range), :] = ty
      ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    if self.config.env_name == 'citeseer':
      features = features.toarray()
      for i in range(self.config.n_ori_features):
        d_nz = np.nonzero(features[:, i])
        nnz = d_nz[0].shape[0]
        if nnz > 0:
          features[:, i] = features[:, i] * np.log(self.config.n_actions / nnz)
      features = csr_matrix(features, dtype=float)
      u, s, vt = svds(features, k=self.config.n_features)
      self.features = u * s
    else:
      self.features = features.toarray()

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    self.label = []
    for el in labels:
      if 1 in el:
        self.label.append(list(el).index(1))
      else:
        self.label.append(0)


    train_num = int(self.config.n_actions * self.config.train_split)
    self.y_train = [-1] * len(self.label)
    self.y_train[:train_num] = self.label[:train_num]

    self.embed_dict = {}
    with open(self.config.file_em) as f:
      for i, l in enumerate(f):
        line = l.strip().split()
        if i != 0:
          nodeid = int(line[0])
          self.embed_dict[nodeid] = [float(ei) for ei in line[1:]]

    self.feature_dict = {}
    self.feature_embed_dict = {}
    embed_matrix = []
    for nodeid in range(self.config.n_actions):
      self.feature_dict[nodeid] = list(self.features[nodeid])
      try:
        embed_vector = self.embed_dict[nodeid]
        embed_matrix.append(embed_vector)
        self.feature_embed_dict[nodeid] = list(self.features[nodeid]) + embed_vector
      except:
        embed_vector = self.pad_vector
        self.embed_dict[nodeid] = embed_vector
        embed_matrix.append(embed_vector)
        self.feature_embed_dict[nodeid] = list(self.features[nodeid]) + embed_vector
    self.feature_embed_dict_inverse = {tuple(np.float64(v)): k for k, v in self.feature_embed_dict.items()}

    nbrs = NearestNeighbors(n_neighbors=self.config.n_neig, algorithm='auto').fit(embed_matrix)
    distances, self.dist_indices = nbrs.kneighbors(embed_matrix)
    # a = [self.dist_indices[nodeid] for nodeid in tmp]

    self.edge_list = []
    with open(self.config.file_edgelist, 'r') as f:
      for l in f:
        self.edge_list.append([int(l.strip().split()[0]), int(l.strip().split()[1])])

    self.adj_ori = 0

    print('Finish loading data----')


  def delayed_reward(self, paths, mode='train'):
    row = []
    col = []
    if mode == 'train':
      for i in range(len(paths) - 1):
        row += [paths[i], paths[i + 1]]
        col += [paths[i + 1], paths[i]]
    else:
      for path in paths:
        for i in range(len(path) - 1):
          row += [path[i], path[i + 1]]
          col += [path[i + 1], path[i]]
    row += self.row_diag
    col += self.col_diag
    data = [1 for i in range(len(row))]
    adj = csr_matrix((data, (row, col)), shape=(self.n_actions, self.n_actions)).astype('bool').astype('int')

    macro_f1_train, micro_p, macro_p, micro_r, macro_r, micro_f, macro_f = self.model_gcn.fit_model(adj)
    reward_gcn = (macro_f1_train - self.config.delayed_reward_baseline) * self.config.path_length * \
                 self.config.delayed_reward_scale
    return reward_gcn, macro_f1_train, micro_p, macro_p, micro_r, macro_r, micro_f, macro_f
