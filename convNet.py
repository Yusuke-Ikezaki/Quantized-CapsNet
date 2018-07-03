import numpy as np
import tensorflow as tf

from config import cfg
from utils import get_batch_data, quantize


class ConvNet(object):
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.X, self.labels = get_batch_data(cfg.dataset, cfg.batch_size, cfg.num_threads)
                self.Y = tf.one_hot(self.labels, depth=10, axis=1, dtype=tf.float32)

                self.build_arch()
                self.loss()
                self._summary()

                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer()
                self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)
            else:
                self.X = tf.placeholder(tf.float32, shape=(cfg.batch_size, 28, 28, 1))
                self.labels = tf.placeholder(tf.int32, shape=(cfg.batch_size, ))
                self.Y = tf.reshape(self.labels, shape=(cfg.batch_size, 10, 1))
                self.build_arch()

        tf.logging.info('Setting up the main structure')

    def build_arch(self):
        with tf.variable_scope('Conv1_layer'):
            # Conv1, [batch_size, 20, 20, 256]
            conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=32, kernel_size=5, stride=1, padding='SAME')
            assert conv1.get_shape() == [cfg.batch_size, 20, 20, 256]

        with tf.variable_scope('Pooling1_layer'):
            # Pooling1, [batch_size, 14, 14, 256]
            pooling1 = tf.contrib.layers.pooling2d(conv1)

'''
w = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Conv/weights")[0]
qw = tf.quantize(w, tf.reduce_min(w), tf.reduce_max(w), tf.qint8)
dqw = tf.dequantize(qw, tf.reduce_min(w), tf.reduce_max(w))
'''
