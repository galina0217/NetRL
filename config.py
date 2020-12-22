class AgentConfig(object):
  mode = 'generate'
  display_list = ['beta', 'double_q', 'train_split', 'discount',
                  'batch_ratio', 'delayed_reward_scale', 'noisy_perc']

  infer_mode = 'path_local'

  policy = 'deterministic'
  ep_greedy = True

  beta = 0.7  # percentage of immediate reward according to label information
  delay = True

  path_length = 100
  inference_path_length = 0
  path_history = 10
  n_max = 5
  n_percent = 0.05
  immediate_reward_test = 0
  n_layers = 3
  scale = 10
  display = False

  n_neig = 100

  max_step = 3000000
  memory_size = 500
  memory_size_min = int(memory_size/path_length)  # 100
  memory_size_init = 150
  memory_size_decayr = 0.3
  memory_size_decays = 2
  batch_ratio = 0.9
  batch_size = 64
  random_start = 30
  cnn_format = 'NCHW'
  discount = 0.1
  target_q_update_step = 1 * scale
  learning_rate = 1e-4
  learning_rate_minimum = 1e-3
  learning_rate_decay = 0.96
  learning_rate_decay_step = 10 * scale

  ep_end = 0.1
  ep_start = 1.
  ep_end_t = memory_size
  # ep_temp = 0
  ep_coe = 3e5

  history_length = 1
  train_frequency = 1
  learn_start = 200

  min_delta = -1
  max_delta = 1

  double_q = True
  dueling = False

  _test_step = 1000
  _save_step = _test_step * 10


class EnvironmentConfig(object):

  env_name = 'cora'
  train_split = 0.8
  # #-------------------0.1--------------------------------
  noisy_perc = 0
  delayed_reward_baseline = 0.99118 #test_f1= 0.82048
  # noisy_perc = 0.2
  # delayed_reward_baseline = 0.97631 #test_f1= 0.74656
  # noisy_perc = 0.4
  # delayed_reward_baseline = 0.97759 #test_f1= 0.63880
  # noisy_perc = 0.6
  # delayed_reward_baseline = 0.95988 #test_f1= 0.47951
  # noisy_perc = 0.8
  # delayed_reward_baseline = 0.92789 #test_f1= 0.33719
  delayed_reward_scale = 1
  n_actions = 2708
  n_features = 1433
  n_embedding = 64
  max_reward = 20.
  min_reward = 0.
  file_edgelist = 'data/' + env_name + '.edgelist.n' + str(noisy_perc)
  file_em = 'data/' + env_name + '.embedding.n' + str(noisy_perc)

  # env_name = 'citeseer'
  # train_split = 0.8
  # #noisy_perc = 0
  # #delayed_reward_baseline = 0.87428 #test_f1= 0.61052
  # #noisy_perc = 0.2
  # #delayed_reward_baseline = 0.80181 #test_f1= 0.52998
  # #noisy_perc = 0.4
  # #delayed_reward_baseline = 0.70495 #test_f1= 0.47099
  # #noisy_perc = 0.6
  # #delayed_reward_baseline = 0.72486 #test_f1= 0.40732
  # noisy_perc = 0.8
  # delayed_reward_baseline = 0.61710 #test_f1= 0.35313
  # delayed_reward_scale = 1
  # n_actions = 3327
  # n_features = 200
  # n_ori_features = 3703
  # n_embedding = 64
  # max_reward = 20.
  # min_reward = 0.
  # file_edgelist = 'data/' + env_name + '.edgelist.n' + str(noisy_perc)
  # file_em = 'data/' + env_name + '.embedding.n' + str(noisy_perc)


class DQNConfig(AgentConfig, EnvironmentConfig):
  model = ''
  pass

class M1(DQNConfig):
  backend = 'tf'
  env_type = 'detail'
  action_repeat = 1

def get_config(FLAGS):
  config = M1
  return config
