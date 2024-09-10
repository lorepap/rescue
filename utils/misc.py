import os

import numpy as np
import tensorflow as tf
import pandas as pd
from utils.models import srcnn, vdsr, edsr  


def create_directory(path):
    """
    Create directory at the given path, checking for errors and if the directory
    already exists.
    """
    path = os.path.realpath(path)
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print(f"The syntax of the output file name, directory or volume is incorrect: {path}")
        else:
            print('\n# Created the output directory "{}"'.format(path))
    else:
        print('\n# The output directory "{}" already exists'.format(path))


def save_df_to_csv(data, path, index_name=None):
    df = pd.DataFrame(data=data)
    if index_name:
        df.index.name = index_name
        df.to_csv(path)
    else:
        df.to_csv(path, index=False)


# The function receive a value in [0, 1] and return a tuple (R, G, B)
def map_to_16M_colors(x):
    val_16M = int(np.rint(x * (2 ** 24 - 1)))
    print(val_16M)
    return [np.uint8(val_16M >> 16), np.uint8(val_16M << 8 >> 16), np.uint8(val_16M << 16 >> 16)]


# Function used to round a floating number to a specified number of decimals
def my_tf_round(x, decimals=0):
    multiplier = tf.constant(10 ** decimals, dtype=tf.float32)
    return tf.round(x * multiplier) / multiplier

def minmax_scale(images):
    # Assuming images is a 4D array with shape (N, 32, 32)
    min_val = np.min(images)
    max_val = np.max(images)
    
    scaled_images = (images - min_val) / (max_val - min_val)
    
    return scaled_images

def get_model(model_name, scale_factor):
    models = {
        'srcnn': srcnn(),
        'vdsr': vdsr(),
        'edsr': edsr(scale_factor)
    }
    return models[model_name]