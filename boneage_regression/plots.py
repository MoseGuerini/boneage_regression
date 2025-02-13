import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import matplotlib.cm as cm
from utils import make_gradcam_heatmap, overlay_heatmap
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_loss_metrics(history):

    # Estrai i dati
    mae = history.history['mean_absolute_error']
    mse = history.history['mean_squared_error']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Crea la figura e i subplot (3 in una riga)
    plt.figure(figsize=(18, 6))

    # Primo subplot: Loss e Validation Loss
    plt.subplot(1, 3, 1)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.legend()

    # Secondo subplot: MAE
    plt.subplot(1, 3, 2)
    plt.plot(mae, label='Mean Absolute Error')
    plt.title('Mean Absolute Error (MAE)')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    
    # Terzo subplot: MSE
    plt.subplot(1, 3, 3)
    plt.plot(mse, label='Mean Squared Error')
    plt.title('Mean Squared Error (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()

    # Mostra il grafico
    plt.show(block=False)
    plt.pause(0.1)

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
    
    # Creare la figura e l'asse
    plt.figure(figsize=(8, 6))

    # Plot delle predizioni contro i valori veri
    plt.scatter(y_true, y_pred, color='blue', alpha=0.5, label='Predizioni')

    # Aggiungere la linea y = x (riferimento)
    lim = np.max([np.max(y_true), np.max(y_pred)])  # Limiti per la linea y=x
    plt.plot([0, lim], [0, lim], color='red', label='y = x', linestyle='--')

    # Aggiungere etichette e titolo
    plt.xlabel('Valori Veri (True Values)')
    plt.ylabel('Valori Predetti (Predicted Values)')
    plt.title('Predizioni vs Valori Veri')

    # Aggiungere una griglia
    plt.grid(True)

    # Aggiungere la legenda
    plt.legend()

    # Mostrare il grafico
    plt.show(block=False)
    plt.pause(0.1)
    
def plot_gender(arr):
    """
    The function creates an histogram using gender real values.

    Parameters
    ----------
    arr : numpy.array
        Array con i valori di gender.
    """
    
    # Conta le occorrenze di ciascun valore
    unique, counts = np.unique(arr, return_counts=True)
    
    # Mappa i valori numerici ai rispettivi labels (nel nostro caso abbiamo femmina = false = 0 è maschio = true = 1)
    gender_labels = {0: 'Female', 1: 'Male'}
    unique_labels = [gender_labels[val] for val in unique]
    
    # Crea il grafico
    plt.figure(figsize=(10, 6))
    plt.bar(unique_labels, counts, width=0.8, color='skyblue')
    plt.xlabel('Gender')
    plt.ylabel('Occurences')
    plt.title('Occurrences distributions')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_boneage(arr):
    """
    The function creates an histogram using boneage real values.

    Parameters
    ----------
    arr : numpy.array
        Array con i valori di boneage.
    """
    # Conta le occorrenze di ciascun valore
    unique, counts = np.unique(arr, return_counts=True)
    
    # Crea il grafico
    plt.figure(figsize=(10, 6))
    plt.bar(unique, counts, width=0.8, color='skyblue')
    plt.xlabel('Valori')
    plt.ylabel('Occorrenze')
    plt.title('Distribuzione delle Occorrenze')
    plt.xticks(np.arange(0, 230, 10))  # Etichette sull'asse X da 1 a 216
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
import numpy as np
import matplotlib.pyplot as plt

def visualize_gradcam(trained_model, img_idx, last_conv_layer_name):
    """
    The functions visualizes the heatmap Grad-CAM overlayed on the original image.

    Parametri:
    - trained_model: model instance after haveing been trained.
    - img_idx: image index which has to be visualized.
    - last_conv_layer_name: name of the very last model convolutional layer.

    Ritorna:
    - None (the funztion only visualizes the heatmap)
    """

    # Prendi l'immagine di test e il corrispondente input di genere
    img_array = [np.expand_dims(trained_model.X_test[img_idx], axis=0),
                 np.expand_dims(trained_model.X_gender_test[img_idx], axis=0)]

    # Genera la heatmap Grad-CAM
    heatmap = make_gradcam_heatmap(img_array, trained_model.trained_model, last_conv_layer_name)

    # Prepara l'immagine originale (se normalizzata tra 0 e 1, scala a 0-255)
    original_img = (trained_model.X_test[img_idx] * 255).astype(np.uint8)

    # Sovrapponi la heatmap
    superimposed_img = overlay_heatmap(original_img, heatmap)

    # Mostra le immagini originali e con heatmap sovrapposta
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(original_img)
    ax[0].set_title("Immagine Originale")
    ax[0].axis("off")

    ax[1].imshow(superimposed_img)
    ax[1].set_title("Grad-CAM Overlay")
    ax[1].axis("off")

    plt.show()
    

def plot_accuracy_threshold(y_pred, y_test, threshold = 5):
    # Calcola l'errore assoluto tra la previsione e il valore reale
    errors = np.abs(y_pred - y_test)
    
    # Controlla quante predizioni sono dentro la soglia (5 mesi)
    correct_predictions = np.sum(errors <= threshold)
    total_predictions = len(y_test)
    accuracy = correct_predictions / total_predictions * 100

    print(f"Accuratezza: {accuracy:.2f}%")

    # Mostra l'errore per ogni previsione
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50)  # Usa un singolo colore per il dataset
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f"Threshold: {threshold} month")
    plt.title('Prediction errors occurences (month)')
    plt.xlabel('Errors (month)')
    plt.ylabel('Occurences')
    plt.legend()
    plt.show()
    