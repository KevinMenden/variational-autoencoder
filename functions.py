"""
Wrapper functions to run the VAE
"""

import tensorflow as tf
from vae import VAE
import imageio as io

def generate_images(model_dir, num_images, out_dir="/home/kevin/test/"):
    """
    Generate new images with a trained model
    :param model_dir: trained VAE model
    :param num_images: number of images to generate
    :param save_dir: directory to save the images
    :return:
    """
    with tf.Session() as sess:
        vae = VAE(sess=sess,
                  model_dir=model_dir,
                  batch_size=num_images,
                  learning_rate=0.0001)
        vae.model_fn()
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        images = vae.generate()
        for i in range(images.shape[0]):
            io.imwrite(out_dir + 'image_' + str(i) + '.jpg', images[i])


def train_model(model_dir, num_steps, batch_size=64, learning_rate=0.0001):
    """
    Train a VAE model
    :param model_dir: model directory
    :param num_steps: number of training steps
    :param learning_rate: learning rate to use
    :return:
    """
    from tensorflow.examples.tutorials.mnist import input_data

    with tf.Session() as sess:
        vae = VAE(sess=sess,
                  model_dir=model_dir,
                  batch_size=batch_size,
                  learning_rate=learning_rate)

        # Build graph
        vae.model_fn()
        # Get Dataset
        mnist = input_data.read_data_sets('MNIST_data')
        # Training
        vae.train(data=mnist, num_epochs=num_steps)


