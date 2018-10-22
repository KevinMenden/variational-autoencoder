"""
Main funtion
"""

import argparse
from functions import *

if __name__=='__main__':


    model_dir = "/home/kevin/models/vae"

    # model training
    train_model(model_dir=model_dir,
                num_steps=1000,
                learning_rate=0.0001,
                batch_size=64)

    # Generate images
    gen_image = False
    if gen_image:
        generate_images(model_dir="/home/kevin/models/vae",
                        num_images=32,
                        out_dir="/home/kevin/test/")



