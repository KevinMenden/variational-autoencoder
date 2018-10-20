"""
Variational Auto-encoder
"""

import tensorflow as tf
import numpy as np



class VAE(object):
    """
    Variational Autoencoder object
    """

    def __init__(self, sess, model_dir, batch_size, learning_rate, width, height, cdim, n_z, model_name='vae_model'):
        self.sess=sess
        self.model_dir=model_dir
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.width=width
        self.height=height
        self.cdim=cdim
        self.n_z=n_z
        self.model_name=model_name




    def encoder(self, x, reuse=False):
        """
        Encoder function
        :param x: input data
        :param n_z: dimensionality of z
        :return: mu and sigma
        """
        activation = tf.nn.relu
        with tf.variable_scope("encoder", reuse=reuse):
            enc = tf.reshape(x, shape=[-1, self.width, self.height, self.cdim])

            # Conv layers
            enc = tf.layers.conv2d(enc, filters=64, kernel_size=4, strides=2, padding="same", activation=activation, name="enc_conv1")
            enc = tf.layers.conv2d(enc, filters=64, kernel_size=4, strides=2, padding="same", activation=activation, name="enc_conv2")
            enc = tf.layers.conv2d(enc, filters=64, kernel_size=4, strides=2, padding="same", activation=activation, name="enc_conv3")
            enc = tf.reshape(enc, [self.batch_size, -1])

            # Out layers
            mu = tf.layers.dense(enc, units=self.n_z, activation=None, name="enc_mu")
            sigma = tf.layers.dense(enc, units=self.n_z, activation=None, name="enc_sigma")
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.n_z]))
            z = mu + tf.multiply(epsilon, tf.exp(0.5*sigma))

            return z, mu, sigma


    def decoder(self, z, reuse=False):
        """
        Decoder function
        :param z: latent layer z
        :return:
        """
        activation=tf.nn.relu
        magic = 24
        with tf.variable_scope("decoder", reuse=reuse):
            # Dense layers
            dec = tf.layers.dense(z, units=magic, activation=tf.nn.relu, name="dec_dense1")
            dec = tf.layers.dense(dec, units=magic*2+1, activation=tf.nn.relu, name="dec_dense2")
            # Deconv layers
            dec = tf.reshape(dec, [-1, 7, 7, self.cdim])
            dec = tf.layers.conv2d_transpose(dec, filters=64, kernel_size=4, strides=2, padding="same", activation=activation, name="dec_conv1")
            dec = tf.layers.conv2d_transpose(dec, filters=64, kernel_size=4, strides=2, padding="same", activation=activation, name="dec_conv2")
            dec = tf.reshape(dec, [self.batch_size, -1])
            # Generate picture
            dec = tf.layers.dense(dec, units=self.width*self.height, activation=tf.nn.sigmoid)
            img = tf.reshape(dec, shape=[-1, self.width, self.height])
            return img

    def compute_loss(self, logits, targets, mu, sigma):
        """
        Compute VAE loss as combinatoin of reconstruction loss and KL divergence
        :param logits: predictions
        :param targets: labels
        :return: loss
        """
        logits_reshaped = tf.reshape(logits, [-1, self.width*self.height])
        targets_reshaped = tf.reshape(targets, [-1, self.width*self.height])
        recon_loss = tf.reduce_sum(tf.squared_difference(logits_reshaped, targets_reshaped), 1)
        kl_div = -0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) -1, [1])
        loss = tf.reduce_mean(recon_loss + kl_div)
        return loss

    def model_fn(self):
        """
        Build the model graph
        """
        # Define placeholders for input and z
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + [self.height, self.width, self.cdim], name="input_img")
        #self.z = tf.placeholder(tf.float32, [self.batch_size, self.n_z])

        # Encoding and sampling
        z, mu, sigma = self.encoder(self.inputs, reuse=False)
        # Decoding
        self.out = self.decoder(z, reuse=False)

        # Loss
        self.loss = self.compute_loss(logits=self.out, targets=self.inputs, mu=mu, sigma=sigma)

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


    def train(self, data=None, num_epochs=100):
        # Init session
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        model = self.model_dir  + '/' + self.model_name
        self.writer = tf.summary.FileWriter(model, self.sess.graph)

        # Summary scalars
        tf.summary.scalar("loss", self.loss)
        merged_summary_op = tf.summary.merge_all()

        # Training loop
        for epoch in range(num_epochs):
            batch = [np.reshape(b, [self.width, self.height, self.cdim]) for b in data.train.next_batch(batch_size=self.batch_size)[0]]

            _, loss, summary = self.sess.run([self.optimizer, self.loss, merged_summary_op],
                                             feed_dict={self.inputs: batch})

            self.writer.add_summary(summary, epoch)
            if epoch % 5 == 0:
                print("Step: {}, loss: {:.5f}".format(epoch, loss))

        self.saver.save(self.sess, model)
        print("Training finished.")






