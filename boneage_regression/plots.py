import matplotlib.pyplot as plt

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
    plt.show()  