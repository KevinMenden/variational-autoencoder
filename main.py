"""
Main funtion
"""

import argparse
from functions.functions import *

if __name__=='__main__':

    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="Whether to train or generate images. One of [train | gen]", default="train")
    parser.add_argument("--model_dir", type=str, help="The model directory", default="./")
    parser.add_argument("--n_img", type=int, help="Number of images to generate. Default: 32", default=32)
    parser.add_argument("--steps", type=int, help="Number of training steps. Default: 1000", default=1000)
    parser.add_argument("--learning_rate", type=float, help="Learning rate. Default: 0.0001", default=0.0001)
    parser.add_argument("--batch_size", type=int, help="Batch size. Default: 64", default=64)
    parser.add_argument("--out", type=str, help="Directory for string results", default="./")

    # Parse arguments
    args = parser.parse_args()
    mode = args.mode
    model_dir = args.model_dir
    num_images = args.n_img
    num_steps = args.steps
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    out_dir = args.out

    if mode == "train":
        train_model_art(model_dir=model_dir,
                        num_steps=num_steps,
                        learning_rate=learning_rate,
                        batch_size=batch_size)

    # Training mode
    if mode == "train_cifar":
        train_model_cifar(model_dir=model_dir,
                    num_steps=num_steps,
                    learning_rate=0.005,
                    batch_size=batch_size)

    if mode == "train_mnist":
        train_model_mnist(model_dir=model_dir,
                          num_steps=num_steps,
                          learning_rate=0.0005,
                          batch_size=batch_size)

    # Generate images
    if mode == "gen":
        generate_images(model_dir=model_dir,
                        num_images=num_images,
                        out_dir=out_dir)


