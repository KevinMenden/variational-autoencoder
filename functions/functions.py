"""
Wrapper functions to run the VAE
"""
import glob
import tensorflow as tf
from vae import VAE
import imageio as io
import numpy as np

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
                  learning_rate=0.0001,
                  height=28,
                  width=28,
                  cdim=1)

        mnist = load_mnist_data(batch_size=num_images)
        vae.model_fn(data=mnist)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        images = vae.generate()
        for i in range(images.shape[0]):
            io.imwrite(out_dir + 'image_' + str(i) + '.jpg', images[i])



def train_model_mnist(model_dir, num_steps, batch_size=64, learning_rate=0.0001):
    """
    Train a VAE model
    :param model_dir: model directory
    :param num_steps: number of training steps
    :param learning_rate: learning rate to use
    :return:
    """

    with tf.Session() as sess:
        vae = VAE(sess=sess,
                  model_dir=model_dir,
                  batch_size=batch_size,
                  learning_rate=learning_rate,
                  height=28,
                  width=28,
                  cdim=1)

        # Load data
        mnist = load_mnist_data(batch_size=batch_size)
        # Build graph
        vae.model_fn(data=mnist)

        # Training
        vae.train(data="placeholder", num_epochs=num_steps)

def train_model_cifar(model_dir, num_steps, batch_size=64, learning_rate=0.0005):
    """
    Train a VAE model
    :param model_dir:
    :param num_steps:
    :param batch_size:
    :param learning_rate:
    :return:
    """

    with tf.Session() as sess:
        vae = VAE(sess=sess,
                  model_dir=model_dir,
                  batch_size=batch_size,
                  learning_rate=learning_rate)

        # Load cifar dataset
        load_cifar_data(cifar_path="/home/kevin/deep_learning/cifar-10-python/cifar-10-batches-py/", batch_size=batch_size)

        # Build graph
        vae.model_fn(data=cifar)

        # Training
        vae.train(num_epochs=num_steps)


def load_cifar_data(cifar_path="/home/kevin/deep_learning/cifar-10-python/cifar-10-batches-py/", batch_size=64):
    """
    Load data from the CIFAR dataset and return as Dataset object
    :param cifar_path:
    :return: Dataset object
    """
    import pickle
    import numpy as np

    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    batches = []
    for i in range(1, 6):
        dict = unpickle(cifar_path + "data_batch_" + str(i))
        batches.append(np.asarray(dict[b'data']).astype("uint8"))
    X = np.concatenate(batches)
    X = np.asarray([np.reshape(b, [32, 32, 3]) for b in X])

    data = tf.data.Dataset.from_tensor_slices(X)
    data = data.shuffle(1000).repeat().batch(batch_size=batch_size)
    return data


def load_mnist_data(batch_size=64):
    """
    Load MNIST data
    :param batch_size:
    :return:
    """
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data')
    data = tf.data.Dataset.from_tensor_slices(mnist.train.images)
    data = data.shuffle(1000).repeat().batch(batch_size=batch_size)
    return data



"""
Functions for loading image data
"""

def _parse_function(filename):
    """
    Parse one image with a label
    :param filename:
    :param label:
    :return:
    """
    img_file = tf.read_file(filename)
    img = tf.image.decode_jpeg(img_file)
    img = tf.image.resize_images(img, size=(128, 128))
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    return img



def load_art_data(data_path="/home/kevin/deep_learning/art_dataset", batch_size=64):
    """
    Load the data
    :param data_path:
    :return: Dataset object
    """
    # Get filenames
    filenames = glob.glob(data_path + "/*.jpg")
    filenames = tf.constant(filenames)

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((filenames))
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat().batch(batch_size=batch_size)

    return dataset

def train_model_art(model_dir, num_steps, batch_size=64, learning_rate=0.0005):
    """
    Train a VAE on the art data
    :param model_dir:
    :param num_steps:
    :param batch_size:
    :param learning_rate:
    :return:
    """
    with tf.Session() as sess:
        vae = VAE(sess=sess,
                  model_dir=model_dir,
                  batch_size=batch_size,
                  learning_rate=learning_rate,
                  height=128,
                  width=128,
                  cdim=3,
                  n_z=64)

        # Load cifar dataset
        data = load_art_data(data_path="/home/kevin/deep_learning/cat-dataset/cats/CAT_00", batch_size=batch_size)

        # Training
        vae.train(data=data, num_epochs=num_steps)






