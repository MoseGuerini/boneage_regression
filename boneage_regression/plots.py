import matplotlib.pyplot as plt
import numpy as np

def plot_loss_metrics(history):

    mae = history.history['mean_absolute_error']
    mse = history.history['mean_squared_error']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Crea la figura e i subplot
    plt.figure(figsize=(14, 6))

    # Primo subplot: Loss e Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.legend()

    # Secondo subplot: MAE e MSE
    plt.subplot(1, 2, 2)
    plt.plot(mae, label='Mean Absolute Error')
    plt.plot(mse, label='Mean Squared Error')
    plt.title('MAE and MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()

    # Aggiungi tight_layout per evitare sovrapposizioni
    plt.tight_layout()

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