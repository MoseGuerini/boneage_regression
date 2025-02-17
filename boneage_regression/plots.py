import matplotlib.pyplot as plt
import numpy as np
from utils import make_gradcam_heatmap, overlay_heatmap
from sklearn.metrics import mean_absolute_error, r2_score
from loguru import logger
from keras import layers

def plot_loss_metrics(history):

    # Data estraction
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

    # Second subplot: r2 score
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
    Crea un grafico delle predizioni vs i valori veri (true values),
    con una linea di riferimento y = x.

    Parameters
    ----------
    y_true : numpy.array
        I valori veri (target).
    y_pred : numpy.array
        I valori predetti dal modello.
    """
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    logger.info(f'Mean absolute error on predicted values: {mae:.1f}')
    logger.info(f'r2 score on predicted values: {r2:.1f}')

    # Create figure and axis
    plt.figure(figsize=(8, 6))

    # Plot: predicted vs real values
    plt.scatter(y_true, y_pred, color='blue', alpha=0.5,
                    label=f'MAE: {mae:.1f} m.\nR² Score: {r2:.1f}')

    # Plotting y = x (ideal prediction line)
    lim = np.max([np.max(y_true), np.max(y_pred)])  # y=x line limit
    plt.plot([0, lim], [0, lim], color='red', label='y = x', linestyle='--')

    # Adding labels and title
    plt.xlabel('Actual age [months]')
    plt.ylabel('Predicted age [months]')
    plt.title('Predicted vs actual age')

    # Adding grid
    plt.grid(True)

    # Adding legend
    plt.legend()
    plt.tight_layout()
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

def visualize_gradcam_batch(trained_model, last_conv_layer_name, num_images=6):
    """
    Visualizza le heatmap Grad-CAM sovrapposte su 6 immagini casuali del test set.

    Parameters:
    - trained_model: Modello addestrato.
    - last_conv_layer_name: Nome dell'ultimo layer convoluzionale.
    - num_images: Numero di immagini da visualizzare (default: 6).

    Returns:
    - None (visualizza le immagini con Grad-CAM overlay).
    """

    # Seleziona `num_images` indici casuali dal test set
    indices = np.random.choice(len(trained_model.X_test), num_images, replace=False)

    # Crea la figura con sottografici (2 righe x 3 colonne)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for i, idx in enumerate(indices):
        row, col = divmod(i, 3)  # Determina posizione nella griglia
        
        # Prepara l'immagine e il dato ausiliario per il modello
        img_array = [
            np.expand_dims(trained_model.X_test[idx], axis=0),
            np.expand_dims(trained_model.X_gender_test[idx], axis=0)
        ]
        
        # Genera la heatmap Grad-CAM
        heatmap = make_gradcam_heatmap(img_array, trained_model.trained_model, last_conv_layer_name)

        # Prepara l'immagine originale
        original_img = (trained_model.X_test[idx] * 255).astype(np.uint8)

        # Sovrappone la heatmap all'immagine originale
        superimposed_img = overlay_heatmap(original_img, heatmap)

        # Mostra l'immagine nel subplot corrispondente
        axes[row, col].imshow(superimposed_img)
        axes[row, col].set_title(f"Sample {idx}")
        axes[row, col].axis("off")  # Rimuove gli assi per pulizia
    
    # Aggiunge uno spazio tra i subplot
    plt.tight_layout()
    plt.show(block=False)


def get_last_conv_layer_name(model):
    # Ispeziona tutti i layer e filtra quelli di tipo 'Conv2D'
    conv_layers = [layer for layer in model.layers if isinstance(layer, layers.Conv2D)]
    
    # Prendi il nome dell'ultimo layer di tipo 'Conv2D'
    if conv_layers:
        return conv_layers[-1].name
    else:
        raise ValueError("No Conv2D layer found in the model.")