import argparse
from loguru import logger
from hyperparameters import set_hyperp
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from PIL import Image


def hyperp_dict(
    conv_layers, conv_filters, dense_units, dense_depth,
    dropout_rate
                ):
    """Creates dictionary containing user-selected hps keeping
    only unique values in each list and sets it to be a global
    variable with set_hyperp"""
    hyperp_dict = {
            'conv_layers': list(set(conv_layers)),
            'conv_filters': list(set(conv_filters)),
            'dense_units': list(set(dense_units)),
            'dense_depth': list(set(dense_depth)),
            'dropout_rate': list(set(dropout_rate))

    }
    set_hyperp(hyperp_dict)
    return hyperp_dict


def check_rate(value):
    """Check if the value is a float between 0 and 1"""
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid value, input type: {type(value)}. Expected float."
            )
    if not (0 <= value <= 1):
        raise argparse.ArgumentTypeError(
            f"Value out of range: {value}. Must be between 0 and 1."
            )

    return value


def is_numeric(s):
    """Check if a given string represents a valid integer.

    :param s: The string to verify.
    :return: True if the string is an integer, False otherwise.
    """
    try:
        int(s)
        return True
    except ValueError:
        logger.warning(
            f"Value '{s}' is not valid."
            f"The image file name must be an integer."
            )
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

        # Assert values to be in 0-255 range (avoiding visualization problem)
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)  # Convert to uint8

        # Resize the image to the target size
        img_resized = tf.image.resize(img, target_size).numpy()

        images_rgb.append(img_resized)
        ids.append(img_id)

    return images_rgb, ids


def str2bool(value):
    """Convert a string to a boolean value."""
    if isinstance(value, bool):
        return value
    if value.lower() in ['true', 't', 'yes', 'y', '1']:
        return True
    elif value.lower() in ['false', 'f', 'no', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid value '{value}' for boolean argument."
            f"Expected values: 'True', 'False', 'Yes', 'No', '1', '0'."
        )


def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        loss = predictions[:, 0]
        # Supponiamo che sia una rete di regressione/scoring ############

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = heatmap.numpy()[0]
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap)  # Normalizing between 0 and 1

    return heatmap


def overlay_heatmap(img, heatmap, alpha=0.4, colormap='jet'):
    # Resize the heatmap to adapt it to the original image size
    heatmap_resized = np.array(Image.fromarray(heatmap).resize(
        (img.shape[1], img.shape[0]), Image.BILINEAR)
                               )

    # Normalizing heatmap between 0 and 1
    heatmap_resized = (heatmap_resized - np.min(heatmap_resized))
    heatmap_resized = heatmap_resized / np.max(heatmap_resized)
    heatmap_resized = heatmap_resized - np.min(heatmap_resized) + 1e-8

    # Apply Matplotlib colormap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]
    # Consider onlt RGB channels

    img = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img

    # Combine immagine and heatmap using alpha blending
    superimposed_img = (1 - alpha) * img + alpha * heatmap_colored

    # Restore uint8 (0-255) format
    superimposed_img = np.clip(superimposed_img * 255, 0, 255).astype(np.uint8)

    return superimposed_img
