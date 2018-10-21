"""
for testing
"""

from functions import *

model_dir = "/home/kevin/models/vae"


# model training
train_model(model_dir=model_dir,
            num_steps=500,
            learning_rate=0.0005,
            batch_size=64)

# image generation
generate_images(model_dir="/home/kevin/models/vae",
                num_images=10,
                out_dir="/home/kevin/test/")