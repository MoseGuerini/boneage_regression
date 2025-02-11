import argparse 
from loguru import logger
from hyperparameters import set_hyperp
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def hyperp_dict(conv_layers, conv_filters, dense_units, dense_depth, dropout_rate):
    """Creates dictionary containing user-selected hps keeping only unique values in each list 
    and sets it to be a global variable with set_hyperp"""
    hyperp_dict = {
            'conv_layers' : list(set(conv_layers)),
            'conv_filters': list(set(conv_filters)),
            'dense_units' : list(set(dense_units)),
            'dense_depth' : list(set(dense_depth)),
            'dropout_rate': list(set(dropout_rate))

    }
    set_hyperp(hyperp_dict)
    return hyperp_dict

def check_rate(value):
    if value < 0 or value > 1:
        raise argparse.ArgumentTypeError(f'Value must be between 0 and 1, input value:{value}')
    
def is_numeric(s):
    """Check if a given string represents a valid integer.

    :param s: The string to verify.
    :return: True if the string is an integer, False otherwise.
    """
    try:
        int(s)
        return True
    except ValueError:
        logger.warning(f"Value '{s}' is not valid. The image file name must be an integer.")
        return False
    
def sorting_and_preprocessing(image_files, target_size):
    images_rgb = []
    ids = []

    for img_path in image_files:
        img = plt.imread(img_path)
        img_id = int(img_path.stem)

        # Switch to RGB if needed (RGB are better from CNN point of view)
        if len(img.shape) == 2:  # BW images
            img = np.stack([img] * 3, axis=-1)
                
        # Assicuriamoci che i valori siano tra 0-255 (evitiamo problemi di visualizzazione)
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)  # Convertiamo in uint8
            
        # Ridimensioniamo l'immagine
        img_resized = tf.image.resize(img, target_size).numpy().astype(np.uint8)
            
        images_rgb.append(img_resized)
        ids.append(img_id)
    
    return images_rgb, ids