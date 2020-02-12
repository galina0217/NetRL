from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np

from predictor.utils import *
from predictor.models import GCN, MLP

# Set random seed
seed = 123
# seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)

class GraphConvolutionalNetwork:
    def __init__(self, mode, env_name, features, label, sample_num, feature_dim,
                 model='gcn', learning_rate=0.01, epochs=200, hidden1=16, dropout=0.5,
                 weight_decay=5e-4, early_stopping=10, max_degree=3, trainsize=0.8, dataset_str='hikivision'):
        # self.graph = graph
        self.mode = mode
        self.env_name = env_name
        # self.adj_ori = adj_ori
        # self.adj = adj
        self.features = features
        self.label = label
        self.sample_num = sample_num
        self.feature_dim = feature_dim
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden1 = hidden1
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.max_degree = max_degree
        self.trainsize = trainsize
        self.dataset_str = dataset_str

        # Load data
        self.y_train, self.y_test, self.train_mask, self.test_mask \
            = construct_dataset(self.label, self.sample_num, self.feature_dim,
                                self.trainsize, self.dataset_str, self.env_name)
        self.load_model()

    def load_model(self):
        # Some preprocessing
        if self.model == 'gcn':
            model_func = GCN
            num_supports = 1
        elif self.model == 'gcn_cheby':
            model_func = GCN
            num_supports = 1 + self.max_degree
        elif self.model == 'dense':
            model_func = MLP
            num_supports = 1
        else:
            raise ValueError('Invalid argument for model: ' + str(self.model))

        # Define placeholders
        if self.dataset_str != 'hikivision':
            self.features = preprocess_features(self.features)
        if self.dataset_str == 'hikivision':
            self.placeholders = {
                'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
                'features': tf.placeholder(tf.float32, shape=(None, self.features.shape[1])),
                'labels': tf.placeholder(tf.float32, shape=(None, self.y_train.shape[1])),
                'labels_mask': tf.placeholder(tf.int32),
                'dropout': tf.placeholder_with_default(0., shape=()),
                'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
            }
            # Create model
            self.model_graph = model_func(self.placeholders, input_dim=self.features.shape[1],
                                          hidden1=self.hidden1,
                                          learning_rate=self.learning_rate,
                                          weight_decay=self.weight_decay,
                                          dataset_str=self.dataset_str,
                                          logging=True)
        else:
            self.placeholders = {
                'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
                'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(self.features[2], dtype=tf.int64)),
                'labels': tf.placeholder(tf.float32, shape=(None, self.y_train.shape[1])),
                'labels_mask': tf.placeholder(tf.int32),
                'dropout': tf.placeholder_with_default(0., shape=()),
                'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
            }
            # Create model
            self.model_graph = model_func(self.placeholders, input_dim=self.features[2][1],
                                          hidden1=self.hidden1,
                                          learning_rate=self.learning_rate,
                                          weight_decay=self.weight_decay,
                                          dataset_str=self.dataset_str,
                                          logging=True)


    def fit_model(self, adj):
        # Some preprocessing
        if self.model == 'gcn':
            support = [preprocess_adj(adj)]
        elif self.model == 'gcn_cheby':
            support = chebyshev_polynomials(adj, self.max_degree)
        elif self.model == 'dense':
            support = [preprocess_adj(adj)]  # Not used
        else:
            raise ValueError('Invalid argument for model: ' + str(self.model))

        # Initialize session
        self.sess = tf.Session()

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        cost_val = []

        # Train model
        for epoch in range(self.epochs):

            # t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(self.features, support, self.y_train, self.train_mask, self.placeholders)
            feed_dict.update({self.placeholders['dropout']: self.dropout})

            # Training step
            outs = self.sess.run([self.model_graph.opt_op, self.model_graph.loss, self.model_graph.accuracy,
                                  self.model_graph.micro_p, self.model_graph.macro_p,
                                  self.model_graph.micro_r, self.model_graph.macro_r,
                                  self.model_graph.micro_f, self.model_graph.macro_f]
                                 , feed_dict=feed_dict)

            # Validation
            cost, acc, micro_p, macro_p, micro_r, macro_r, \
            micro_f, macro_f, duration = self.evaluate(self.features, support,
                                                       self.y_test, self.test_mask, self.placeholders)
            cost_val.append(cost)

            if epoch > self.early_stopping and cost_val[-1] > np.mean(cost_val[-(self.early_stopping+1):-1]):
                # print("Early stopping...")
                break

        # print("Optimization Finished!")
        self.sess.close()

        return outs[8], micro_p, macro_p, micro_r, macro_r, micro_f, macro_f

    # Define model evaluation function
    def evaluate(self, features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = self.sess.run([self.model_graph.loss, self.model_graph.accuracy, self.model_graph.micro_p,
                                  self.model_graph.macro_p, self.model_graph.micro_r, self.model_graph.macro_r,
                                  self.model_graph.micro_f, self.model_graph.macro_f], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], outs_val[2], outs_val[3], outs_val[4], outs_val[5], \
               outs_val[6], outs_val[7], (time.time() - t_test)


