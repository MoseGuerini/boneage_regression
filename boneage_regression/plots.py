import matplotlib.pyplot as plt
import numpy as np
from utils import make_gradcam_heatmap, overlay_heatmap
from sklearn.metrics import mean_absolute_error, r2_score
from loguru import logger
from keras import layers


def plot_loss_metrics(history):
    """
    Plots training history metrics, including loss,
    mean absolute error (MAE) and R2 score.

    :param history: Training history object returned by Keras model.fit(),
                    containing loss and metric values.
    :type history: keras.callbacks.History

    :return: None (displays the plots)
    :rtype: None
    """

    # History data estraction
    mae = history.history['mean_absolute_error']
    val_mae = history.history['val_mean_absolute_error']
    r2 = history.history['r2_score']
    val_r2 = history.history['val_r2_score']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Create figure and subplots (3 whithin a row)
    plt.figure(figsize=(18, 6))

    # First subplot: Loss e Validation Loss
    plt.subplot(1, 3, 1)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Second subplot: MAE
    plt.subplot(1, 3, 2)
    plt.plot(mae, label='Mean Absolute Error')
    plt.plot(val_mae, label='Val. Mean Absolute Error')
    plt.title('Mean Absolute Error (MAE)')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    # Second subplot: R2 score
    plt.subplot(1, 3, 3)
    plt.plot(r2, label='R2 score')
    plt.plot(val_r2, label='Val. R2 score')
    plt.title('R2 score')
    plt.xlabel('Epochs')
    plt.ylabel('R2')
    plt.legend()
    plt.grid(True)

    # Show the figure
    plt.show(block=False)


def plot_predictions(y_true, y_pred):
    """
    Create a scatter plot of predictions vs true values.

    :param y_true: The true values (targets).
    :type y_true: numpy.array
    :param y_pred: The predicted values by the model.
    :type y_pred: numpy.array

    :return: None, displays the plot with predicted vs actual values.
    :rtype: None
    """
    # Calculate Mean Absolute Error and R2 Score
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Log MAE and R2 score values
    logger.info(f'Mean absolute error on predicted values: {mae:.1f}')
    logger.info(f'r2 score on predicted values: {r2:.1f}')

    # Create figure and axis
    plt.figure(figsize=(8, 6))

    # Scatter plot: predicted vs actual values
    plt.scatter(y_true, y_pred, color='blue', alpha=0.5,
                label=f'MAE: {mae:.1f} m.\nR² Score: {r2:.1f}')

    # Plotting y = x (ideal prediction line)
    lim = np.max([np.max(y_true), np.max(y_pred)])
    plt.plot([0, lim], [0, lim], color='red', label='y = x', linestyle='--')

    # Set axis labels, title, and grid
    plt.xlabel('Actual age [months]')
    plt.ylabel('Predicted age [months]')
    plt.title('Predicted vs actual age')
    plt.grid(True)

    # Show the legend and adjust the layout
    plt.legend()
    plt.tight_layout()

    # Show the figure
    plt.show(block=False)


def plot_gender(arr):
    """
    Creates a bar plot showing the distribution of gender values in the input
    array.

    :param arr: Array containing the gender values (0 for female, 1 for male).
    :type arr: numpy.array
    :return: None (displays the plot).
    :rtype: None
    """
    # Get occurrences of each unique value in the array
    unique, counts = np.unique(arr, return_counts=True)

    # Mapping gender values (0: Female, 1: Male)
    gender_labels = {0: 'Female', 1: 'Male'}
    unique_labels = [gender_labels[val] for val in unique]

    # Create and customize the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(unique_labels, counts, width=0.8, color=['b', 'r'])

    # Set axis labels, title, and grid
    plt.xlabel('Gender')
    plt.ylabel('Occurences')
    plt.title('Occurrences distributions')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.show(block=False)


def plot_boneage(arr):
    """
    Creates a bar plot showing the distribution of boneage values in the input
    array.

    :param arr: Array containing the boneage values.
    :type arr: numpy.array
    :return: None (displays the plot).
    :rtype: None
    """
    # Get occurrences of each unique value in the array
    unique, counts = np.unique(arr, return_counts=True)

    # Create and customize the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(unique, counts, width=0.8, color='b')

    # Set axis labels, title, ticks, and grid
    plt.xlabel('Values')
    plt.ylabel('Occurrences')
    plt.title('Distribution of Occurrences')
    plt.xticks(np.arange(0, 230, 10))
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.show(block=False)


def plot_accuracy_threshold(y_pred, y_test, threshold=5):
    """
    Plots the distribution of prediction errors.

    This function computes the absolute error between predictions and actual
    values, determines the percentage of predictions within the specified
    threshold, and visualizes the error distribution using a histogram.

    :param y_pred: np.ndarray
        Array of predicted values.
    :param y_test: np.ndarray
        Array of actual values.
    :param threshold: int or float, optional (default=5)
        The threshold (in months) within which a prediction is considered
        accurate.

    :return: None
    :rtype: None
    """
    # Compute absolute errors
    errors = np.abs(y_pred - y_test)

    # Compute accuracy within threshold
    correct_predictions = np.sum(errors <= threshold)
    total_predictions = len(y_test)
    accuracy = (correct_predictions / total_predictions) * 100

    print(f"Accuracy: {accuracy:.2f}%")

    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, color='blue', alpha=0.7)
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2,
                label=f"Threshold: {threshold} months")
    plt.title('Prediction Error Occurrences (Months)')
    plt.xlabel('Error (Months)')
    plt.ylabel('Occurrences')
    plt.legend()
    plt.show(block=False)


def visualize_gradcam_batch(trained_model, last_conv_layer_name, num_images=6):
    """
    Visualizes Grad-CAM heatmaps overlayed on 6 random images from the test set.

    This function selects `num_images` random images from the test set,
    generates Grad-CAM heatmaps for each image, and overlays them on the original
    image to visualize the areas of focus. The images are then displayed in a
    grid layout.

    :param trained_model: object
        The trained model to be used for generating Grad-CAM heatmaps. It should
        contain the attributes `X_test` and `X_gender_test` for input data.
    :param last_conv_layer_name: str
        The name of the last convolutional layer in the model. This layer is
        used to compute the Grad-CAM heatmaps.
    :param num_images: int, optional (default=6)
        The number of random images to visualize from the test set.

    :return: None
        This function only displays the Grad-CAM heatmap overlayed on images.
    """
    # Select `num_images` random indices from the test set
    indices = np.random.choice(len(trained_model.X_test),
                               num_images,
                               replace=False)

    # Create figure with subplots (2 rows x 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for i, idx in enumerate(indices):
        row, col = divmod(i, 3)

        # Prepare the image and auxiliary data for the model
        img_array = [
            np.expand_dims(trained_model.X_test[idx], axis=0),
            np.expand_dims(trained_model.X_gender_test[idx], axis=0)
        ]

        # Generate Grad-CAM heatmap
        heatmap = make_gradcam_heatmap(img_array,
                                       trained_model.trained_model,
                                       last_conv_layer_name)

        # Prepare the original image
        original_img = (trained_model.X_test[idx] * 255).astype(np.uint8)

        # Overlay the heatmap on the original image
        superimposed_img = overlay_heatmap(original_img, heatmap)

        # Mostra l'immagine nel subplot corrispondente
        axes[row, col].imshow(superimposed_img)
        axes[row, col].set_title(f"Sample {idx}")
        axes[row, col].axis("off")  # Rimuove gli assi per pulizia

    # Adjust the layout and show the figure
    plt.tight_layout()
    plt.show(block=False)


def get_last_conv_layer_name(model):
    """
    Inspects the model's layers and returns the name of the last 'Conv2D'
    layer.

    This function filters all layers of type 'Conv2D' and returns the name
    of the last such layer in the model. If no 'Conv2D' layers are found,
    a ValueError is raised.

    :param model: keras.Model
        The trained Keras model from which to extract the last Conv2D layer.

    :return: str
        The name of the last 'Conv2D' layer.
    :raises ValueError:
        If no 'Conv2D' layers are found in the model.
    """
    # Inspect all layers and filter those of type 'Conv2D'
    conv_layers = [layer for layer in model.layers
                   if isinstance(layer, layers.Conv2D)]

    # Return the name of the last 'Conv2D' layer
    if conv_layers:
        return conv_layers[-1].name
    else:
        raise ValueError("No Conv2D layer found in the model.")
