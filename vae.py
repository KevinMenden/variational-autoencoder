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




    def encoder(self, x, n_z, width=28, height=28):
        """
        Encoder function
        :param x: input data
        :param n_z: dimensionality of z
        :return: mu and sigma
        """
        activation = tf.nn.relu
        with tf.variable_scope("encoder", reuse=None):
            enc = tf.reshape(x, shape=[-1, width, height, 1])

            # Conv layers
            enc = tf.layers.conv2d(enc, filters=64, kernel_size=4, strides=2, padding="same", activation=activation, name="enc_conv1")
            enc = tf.layers.conv2d(enc, filters=64, kernel_size=4, strides=2, padding="same", activation=activation, name="enc_conv2")
            enc = tf.layers.conv2d(enc, filters=64, kernel_size=4, strides=2, padding="same", activation=activation, name="enc_conv3")
            enc = tf.reshape(enc, [self.batch_size, -1])

            # Out layers
            mu = tf.layers.dense(enc, units=n_z, activation=None, name="enc_mu")
            sigma = tf.layers.dense(enc, units=n_z, activation=None, name="enc_sigma")
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_z]))
            z = mu + tf.multiply(epsilon, tf.exp(0.5*sigma))

            return z, mu, sigma


    def decoder(self, z, width=28, height=28):
        """
        Decoder function
        :param z: latent layer z
        :return:
        """
        activation=tf.nn.relu
        with tf.variable_scope("decoder", reuse=None):
            dec = tf.layers.dense(z, units=256, activation=tf.nn.relu, name="dec_dense1")
            dec = tf.layers.dense(dec, units=128, activation=tf.nn.relu, name="dec_dense2")
            dec = tf.reshape(dec, [-1, 7, 7, 1])
            dec = tf.layers.conv2d_transpose(dec, filters=64, kernel_size=4, strides=2, padding="same", activation=activation, name="dec_conv1")
            dec = tf.layers.conv2d_transpose(dec, filters=64, kernel_size=4, strides=2, padding="same", activation=activation, name="dec_conv2")
            dec = tf.reshape(dec, [self.batch_size, -1])
            dec = tf.layers.dense(dec, units=width*height, activation=tf.nn.sigmoid)
            img = tf.reshape(dec, shape=[-1, width, height])
            return img

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


