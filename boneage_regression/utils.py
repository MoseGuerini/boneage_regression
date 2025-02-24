import argparse
from loguru import logger
from hyperparameters import set_hyperp
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from PIL import Image
from pathlib import Path


def hyperp_dict(
    conv_layers, conv_filters, dense_units, dense_depth, dropout_rate
):
    """
    Creates a dictionary containing user-selected hyperparameters, ensuring
    only unique values in each list, and sets it as a global variable using
    `set_hyperp`.

    :param conv_layers: List of possible numbers of convolutional layers.
    :type conv_layers: list[int]
    :param conv_filters: List of possible numbers of filters per conv layer.
    :type conv_filters: list[int]
    :param dense_units: List of possible numbers of units in dense layers.
    :type dense_units: list[int]
    :param dense_depth: List of possible numbers of dense layers.
    :type dense_depth: list[int]
    :param dropout_rate: List of possible dropout rates.
    :type dropout_rate: list[float]

    :return: A dictionary containing the unique hyperparameter values.
    :rtype: dict[str, list]
    """
    hyperp_dict = {
        'conv_layers': list(set(conv_layers)),
        'conv_filters': list(set(conv_filters)),
        'dense_units': list(set(dense_units)),
        'dense_depth': list(set(dense_depth)),
        'dropout_rate': list(set(dropout_rate)),
    }
    set_hyperp(hyperp_dict)
    return hyperp_dict


def check_rate(value):
    """
    Validates if the given value is a float between 0 and 1.

    :param value: The input value to check.
    :type value: float or str

    :raises argparse.ArgumentTypeError: If the value is not a valid float
        or is outside the range [0, 1].

    :return: The validated float value.
    :rtype: float
    """
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
    """
    Check if a given string represents a valid integer.

    :param s: The string to verify.
    :type s: str

    :return: True if the string is an integer, False otherwise.
    :rtype: bool
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
    """
    Sort and preprocess images for model input.

    This function reads image files, converts grayscale images to RGB,
    normalizes pixel values, resizes images, and returns processed images
    along with their IDs.

    :param image_files: List of image file paths.
    :type image_files: list[pathlib.Path]
    :param target_size: Target size for resizing (height, width).
    :type target_size: tuple[int, int]

    :return: Processed images as NumPy arrays and their corresponding IDs.
    :rtype: tuple[list[np.ndarray], list[int]]
    """
    images_rgb = []
    ids = []

    for img_path in image_files:
        img = plt.imread(img_path)
        img_id = int(img_path.stem)

        # Switch to RGB if needed (RGB are better from CNN point of view)
        if len(img.shape) == 2:  # BW images
            img = np.stack([img] * 1, axis=-1)

        # Assert values to be in 0-255 range (avoiding visualization problem)
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)  # Convert to uint8

        # Resize the image to the target size
        img_resized = tf.image.resize(img, target_size).numpy()

        images_rgb.append(img_resized)
        ids.append(img_id)

    return images_rgb, ids


def str2bool(value):
    """
    Convert a string representation of a boolean to an actual boolean value.

    This function is useful for parsing boolean arguments from the command line.
    It accepts common string representations of boolean values.

    :param value: The string to convert.
    :type value: str or bool

    :return: The corresponding boolean value.
    :rtype: bool

    :raises argparse.ArgumentTypeError: If the input string is not a valid
        boolean representation.
    """
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
    """
    Retrieve the name of the last Conv2D layer in the given model.

    This function inspects the layers of the model in reverse order and returns
    the name of the first Conv2D layer encountered, which corresponds to the
    last Conv2D layer in the model.

    :param model: The Keras model to inspect.
    :type model: keras.Model

    :return: The name of the last Conv2D layer.
    :rtype: str

    :raises ValueError: If no Conv2D layer is found in the model.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    Generate a Grad-CAM heatmap for a given image, using the specified model
    and the last convolutional layer.

    This function uses a gradient-based technique to highlight the important
    regions of the input image for making predictions. The heatmap generated
    can be superimposed on the image to visualize which parts of the image
    were important for the model's decision.

    :param img_array: The input image(s) to process.
    :type img_array: np.ndarray
    :param model: The trained model used to generate the predictions.
    :type model: keras.Model
    :param last_conv_layer_name: The name of the last convolutional layer in the model.
    :type last_conv_layer_name: str

    :return: The Grad-CAM heatmap.
    :rtype: np.ndarray

    :raises ValueError: If the model doesn't contain the specified convolutional layer.
    """
    grad_model = keras.models.Model(
        model.inputs,
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
    heatmap /= (np.max(heatmap) + 1e-8)  # Avoid division by zero

    return heatmap


def overlay_heatmap(img, heatmap, alpha=0.4, colormap='jet'):
    """
    Overlay a heatmap on the original image using alpha blending.

    This function resizes the heatmap to match the input image size, normalizes
    it to the range [0, 1], applies a specified colormap, and then combines
    the heatmap with the original image using alpha blending. The resulting image
    will highlight the important regions according to the heatmap.

    :param img: The input image to overlay the heatmap on.
    :type img: np.ndarray
    :param heatmap: The heatmap to overlay on the image.
    :type heatmap: np.ndarray
    :param alpha: The blending factor between the image and the heatmap.
                  (Default is 0.4)
    :type alpha: float, optional
    :param colormap: The colormap to use for the heatmap visualization.
                     (Default is 'jet')
    :type colormap: str, optional

    :return: The image with the heatmap overlayed.
    :rtype: np.ndarray
    """
    # Resize the heatmap to adapt it to the original image size
    heatmap_resized = np.array(
        Image.fromarray(heatmap).resize((img.shape[1], img.shape[0]), Image.BILINEAR)
    )

    # Normalizing heatmap between 0 and 1
    heatmap_resized = (heatmap_resized - np.min(heatmap_resized))
    heatmap_resized = heatmap_resized / np.max(heatmap_resized)
    heatmap_resized = heatmap_resized - np.min(heatmap_resized) + 1e-8

    # Apply Matplotlib colormap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # Consider only RGB channels

    # If the image is in uint8, convert it to float32 for blending
    img = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img

    # Combine immagine and heatmap using alpha blending
    superimposed_img = (1 - alpha) * img + alpha * heatmap_colored

    # Restore uint8 (0-255) format
    superimposed_img = np.clip(superimposed_img * 255, 0, 255).astype(np.uint8)

    return superimposed_img

def save_image(file_name, folder_name = 'Grafici'):
    # Ottieni il percorso del folder corrente
    current_path = Path.cwd()
    
    # Torna indietro di due livelli
    parent_folder = current_path.parent
    
    # Crea il percorso del folder_name
    folder_path = parent_folder / folder_name
    
    # Crea la cartella se non esiste
    folder_path.mkdir(exist_ok=True)

    # Crea l'immagine e salvala
    image_path = folder_path / file_name
    
    # If a file with the same name is already present, delete it
    if image_path.exists():
        image_path.unlink()
    
    Path(file_name).rename(image_path)
