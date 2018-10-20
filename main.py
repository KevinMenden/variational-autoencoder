"""
Main funtion
"""

import tensorflow as tf
from vae import VAE
from tensorflow.examples.tutorials.mnist import input_data
import imageio as io


if __name__=='__main__':

    with tf.Session() as sess:
        vae = VAE(sess=sess,
                  model_dir="/home/kevin/models/vae",
                  batch_size=64,
                  learning_rate=0.0001,
                  width=28,
                  height=28,
                  cdim=1,
                  n_z=64)

        # Build graph
        vae.model_fn()

        # Get Dataset
        mnist = input_data.read_data_sets('MNIST_data')

        # Training
        vae.train(data=mnist, num_epochs=2000)

        # Generate images
        images = vae.generate()
        outdir = "/home/kevin/test/"
        for i in range(images.shape[0]):
            io.imwrite(outdir + 'image_' + str(i) + '.jpg', images[i])
