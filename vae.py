"""
Variational Auto-encoder
"""

import tensorflow as tf
import numpy as np


# Import MNIST data
from tf.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


class VAE(object):

    def __init__(self, model_dir, sess, batch_size):
        self.model_dir=None,
        self.sess=sess,
        self.batch_size=batch_size


    def encoder(self, x, n_z):
        """
        Encoder function
        :param x: input data
        :param n_z: dimensionality of z
        :return: mu and log_sigma
        """
        with tf.variable_scope("encoder", reuse=None):
            enc = tf.layers.dense(x, units=256, activation=tf.nn.relu, name="enc1")
            enc = tf.layers.dense(enc, units=128, activation=tf.nn.relu, name="enc2")
            mu = tf.layers.dense(enc, units=n_z, activation=None, name="enc_mu")
            log_sigma = tf.layers.dense(enc, units=n_z, activation=None, name="enc_sigma")
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_z]))
            z = mu + tf.multiply(epsilon, tf.exp(0.5*log_sigma))

            return z, mu, log_sigma


    def decoder(self, z):
        """
        Decoder function
        :param z: latent layer z
        :return: 
        """

        with tf.variable_scope("decoder", reuse=None):
            dec = tf.layers.dense(z, units=256, activation=tf.nn.relu, name="dec1")
            dec = tf.layers.dense(dec, units=128, activation=tf.nn.relu, name="dec2")
            res = tf.layers.dense(dec, units=64, activation=None, name="out_layer")
            return res


