import matplotlib.pyplot as plt
import numpy as np
from utils import make_gradcam_heatmap, overlay_heatmap


def plot_loss_metrics(history):

    # Data estraction
    mae = history.history['mean_absolute_error']
    val_mae = history.history['val_mean_absolute_error']
    mse = history.history['mean_squared_error']
    val_mse = history.history['val_mean_squared_error']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Create figure and subplots (3 whithin a row)
    plt.figure(figsize=(12, 6))

    # First subplot: Loss e Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.legend()

    # Second subplot: MAE
    plt.subplot(1, 2, 2)
    plt.plot(mae, label='Mean Absolute Error')
    plt.plot(val_mae, label='Val. Mean Absolute Error')
    plt.title('Mean Absolute Error (MAE)')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    # Show the figure
    plt.show(block=False)


def plot_predictions(y_true, y_pred):
    """
    Crea un grafico delle predizioni vs i valori veri (true values),
    con una linea di riferimento y = x.

    Parameters
    ----------
    y_true : numpy.array
        I valori veri (target).
    y_pred : numpy.array
        I valori predetti dal modello.
    """

    # Create figure and axis
    plt.figure(figsize=(8, 6))

    # Plot: predicted vs real values
    plt.scatter(y_true, y_pred, color='blue', alpha=0.5, label='Predicted')

    # Plotting y = x (ideal prediction line)
    lim = np.max([np.max(y_true), np.max(y_pred)])  # y=x line limit
    plt.plot([0, lim], [0, lim], color='red', label='y = x', linestyle='--')

    # Adding labels and title
    plt.xlabel('Real Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs real values')

    # Adding grid
    plt.grid(True)

    # Adding legend
    plt.legend()

    # Show the plot
    plt.show(block=False)


def plot_gender(arr):
    """
    The function creates an histogram using gender real values.

    Parameters
    ----------
    arr : numpy.array
        Array con i valori di gender.
    """

    # Occurences of every values
    unique, counts = np.unique(arr, return_counts=True)

    # According to our data description: female = false = 0 and
    # male = true = 1)
    gender_labels = {0: 'Female', 1: 'Male'}
    unique_labels = [gender_labels[val] for val in unique]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.bar(unique_labels, counts, width=0.8, color=['b', 'r'])
    plt.xlabel('Gender')
    plt.ylabel('Occurences')
    plt.title('Occurrences distributions')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show(block=False)


def plot_boneage(arr):
    """
    The function creates an histogram using boneage real values.

    Parameters
    ----------
    arr : numpy.array
        Array con i valori di boneage.
    """
    # Occurences of every values
    unique, counts = np.unique(arr, return_counts=True)

    # Crea il grafico
    plt.figure(figsize=(10, 6))
    plt.bar(unique, counts, width=0.8, color='b')
    plt.xlabel('Values')
    plt.ylabel('Occurrences')
    plt.title('Distribution of Occurrences')
    plt.xticks(np.arange(0, 230, 10))  # Ticks from 0 to 230 (step of 10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show(block=False)


def visualize_gradcam(trained_model, img_idx, last_conv_layer_name):
    """
    This function visualizes the Grad-CAM heatmap overlayed on the original
    image.

    Parameters:
    - trained_model: An instance of the model after being trained.
    - img_idx: The index of the image to be visualized.
    - last_conv_layer_name: The name of the last convolutional layer of the
    model.

    Returns:
    - None (the function only visualizes the heatmap)
    """

    # Extract the image and corresponding input gender
    img_array = [
        np.expand_dims(trained_model.X_test[img_idx], axis=0),
        np.expand_dims(trained_model.X_gender_test[img_idx], axis=0)
                 ]

    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_array, trained_model.trained_model,
                                   last_conv_layer_name)

    # Quick original image processing (scale to 0-255)
    original_img = (trained_model.X_test[img_idx] * 255).astype(np.uint8)

    # Overlay the heatmap on the original image
    superimposed_img = overlay_heatmap(original_img, heatmap)

    # Display the original and superimposed images
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(original_img)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(superimposed_img)
    ax[1].set_title("Grad-CAM Overlay")
    ax[1].axis("off")

    plt.show(block=False)


def plot_accuracy_threshold(y_pred, y_test, threshold=5):
    """
    Plots the distribution of prediction errors and calculates the accuracy
    within a given threshold.

    This function computes the absolute error between predictions and actual
    values,
    determines the percentage of predictions within the specified threshold,
    and visualizes the error distribution using a histogram.

    :param y_pred: np.ndarray
        Array of predicted values.
    :param y_test: np.ndarray
        Array of actual values.
    :param threshold: int or float, optional (default=5)
        The threshold (in months) within which a prediction is considered
        accurate.

    :return: None
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
