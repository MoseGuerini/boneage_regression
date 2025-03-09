import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.metrics import mean_absolute_error, r2_score

from utils import save_image


def plot_loss_metrics(history, fold):
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
    plt.title(f'Fold {fold}: Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    # Second subplot: MAE
    plt.subplot(1, 3, 2)
    plt.plot(mae, label='Mean Absolute Error')
    plt.plot(val_mae, label='Val. Mean Absolute Error')
    plt.title(f'Fold {fold}: MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    # Second subplot: R2 score
    plt.subplot(1, 3, 3)
    plt.plot(r2, label='R2 score')
    plt.plot(val_r2, label='Val. R2 score')
    plt.title(f'Fold {fold}: R2 score')
    plt.xlabel('Epochs')
    plt.ylabel('R2')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the image
    image_name = f'fold{fold}_loss.png'
    save_image(image_name)
    plt.close()


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
                label=f'MAE: {mae:.1f} m.\nR² Score: {r2:.2f}')

    # Plotting y = x (ideal prediction line)
    lim = np.max([np.max(y_true), np.max(y_pred)])
    plt.plot([0, lim], [0, lim], color='red', label='y = x', linestyle='--')

    # Set axis labels, title, and grid
    plt.xlabel('Actual age [months]', fontsize=14)
    plt.ylabel('Predicted age [months]', fontsize=14)
    plt.title('Predicted vs actual age', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=14)

    # Save the image
    save_image('predictions.png')
    plt.close()


def plot_error_distribution(y_pred, y_test, threshold=5):
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
    percentage_below_th = (correct_predictions / total_predictions) * 100

    logger.info(f"Prediction below threshold: {percentage_below_th:.2f}%")

    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, color='blue', alpha=0.7)
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2,
                label=f"Threshold: {threshold} months")
    plt.title('Prediction Error Occurrences', fontsize=14)
    plt.xlabel('Error [months]', fontsize=14)
    plt.ylabel('Occurrences', fontsize=14)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)

    # Save the image
    save_image('error_distribution.png')
    plt.close()
