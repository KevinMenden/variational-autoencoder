"""
Variational Auto-encoder
"""

import tensorflow as tf
import numpy as np


# Import MNIST data
from tf.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


class VAE(object):

    def __init__(self, model_dir, sess, batch_size, learning_rate):
        self.model_dir=None
        self.sess=sess
        self.batch_size=batch_size
        self.learning_rate=learning_rate




    def encoder(self, x, n_z):
        """
        Encoder function
        :param x: input data
        :param n_z: dimensionality of z
        :return: mu and sigma
        """
        with tf.variable_scope("encoder", reuse=None):
            enc = tf.layers.dense(x, units=256, activation=tf.nn.relu, name="enc1")
            enc = tf.layers.dense(enc, units=128, activation=tf.nn.relu, name="enc2")
            mu = tf.layers.dense(enc, units=n_z, activation=None, name="enc_mu")
            sigma = tf.layers.dense(enc, units=n_z, activation=None, name="enc_sigma")
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_z]))
            z = mu + tf.multiply(epsilon, tf.exp(0.5*sigma))

            return z, mu, sigma


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

    def compute_loss(self, logits, targets, mu, sigma):
        """
        Compute VAE loss as combinatoin of reconstruction loss and KL divergence
        :param logits: predictions
        :param targets: labels
        :return: loss
        """
        recon_loss = tf.reduce_sum(tf.squared_difference(logits, targets), 1)
        kl_div = -0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) -1, [1])
        loss = tf.reduce_mean(recon_loss + kl_div)
        return loss


