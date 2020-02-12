# -*- coding: UTF-8 -*- 
from __future__ import print_function
import random
import tensorflow as tf
import os

from dqn.agent import Agent
from dqn.environment import OurEnvironment
from config import get_config

flags = tf.app.flags

# Environment
flags.DEFINE_string('env_name', 'Hikvision', 'The name of environment to use')

# Etc
flags.DEFINE_boolean('use_gpu', False, 'Whether to use gpu or not')
flags.DEFINE_string('gpu_id', '1', 'Which gpu to use')
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

FLAGS = flags.FLAGS

# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

if FLAGS.use_gpu == True:
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id
else:
  os.environ['CUDA_VISIBLE_DEVICES'] = ""

if FLAGS.gpu_fraction == '':
  raise ValueError("--gpu_fraction should be defined")

def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print(" [*] GPU : %.4f" % fraction)
  return fraction

def main(_):
  gpu_options = tf.GPUOptions(
      per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))

  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    config = get_config(FLAGS) or FLAGS

    env = OurEnvironment(config)

    if not tf.test.is_gpu_available() and FLAGS.use_gpu:
      raise Exception("use_gpu flag is true when no GPUs are available")

    agent = Agent(config, env, sess)

    agent.train()

if __name__ == '__main__':
  tf.app.run()
