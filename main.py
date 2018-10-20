"""
Main funtion
"""

import tensorflow as tf
from vae import VAE
from tensorflow.examples.tutorials.mnist import input_data

def main():
    """
    Main function
    """

    with tf.Session() as sess:
        vae = VAE(sess=sess,
                  model_dir="/home/kevin/models/vae",
                  batch_size=32,
                  learning_rate=0.0001,
                  width=28,
                  height=28,
                  cdim=1,
                  n_z=8)

        # Build graph
        vae.model_fn()

        # Get Dataset
        mnist = input_data.read_data_sets('MNIST_data')

        # Training
        vae.train(data=mnist, num_epochs=500)


if __name__=='__main__':
    main()