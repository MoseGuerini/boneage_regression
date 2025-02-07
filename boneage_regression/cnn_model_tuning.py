from keras import layers, models, optimizers, callbacks
import keras_tuner as kt
from keras_tuner import Hyperband
from sklearn.model_selection import KFold
import numpy as np
from loguru import logger 
import sys
import matplotlib.pyplot as plt
import os

from utils import return_dataset, preprocessing_image

# Setting logger configuration 
logger.remove()  
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO"
)

def build_model(hp):
    
    """
    Builds a CNN model for regression using hyperparameter tuning.

    :param hp: Hyperparameter tuning instance from Keras Tuner, used to define model parameters.
    :type hp: keras_tuner.HyperParameters
    :param input_shape: Shape of the input images for the model (excluding batch size), e.g., (128, 128, 3).
    :type input_shape: tuple[int]

    :return: Uncompiled Keras model built with the specified hyperparameters.
    :rtype: keras.Model
    """
    
    # First Branch (images features)
    input_image = layers.Input(shape=(256, 256, 3))
    x = input_image

    # Number of convolutional layers (hyperparameter)
    hp_num_conv_layers = hp.Int('num_conv_layers', min_value=3, max_value=5, step=1)
    hp_filters = hp.Int(f'filters', min_value=16, max_value=64, step=16)

    for i in range(hp_num_conv_layers):
        x = layers.Conv2D(hp_filters*(i+1), (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)

    # Dense layer before concatenation
    x = layers.Dense(hp.Int('dense_units', min_value=32, max_value=128, step=32), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Second Branch (gender features)
    input_gender = layers.Input(shape=(1,))
    y = input_gender

    # Concatenate two branches
    concatenated = layers.concatenate([x, y])

    # Fully connected layers
    num_dense = hp.Int('dense_units_2', min_value=32, max_value=128, step=32)
    x = layers.Dense(num_dense, activation='relu')(concatenated)
    x = layers.Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1))(x)
    x = layers.Dense(int(num_dense/2), activation='relu')(x)

    # Output layer for regression
    output = layers.Dense(1, activation='linear')(x)

    # Create model
    model = models.Model(inputs=[input_image, input_gender], outputs=output)

    # Compile with hyperparameter tuning for learning rate
    model.compile(
        optimizer=optimizers.Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
        loss='mean_squared_error',
        metrics=['mae']
    )

    return model

def run_hyperparameter_tuning(x_train, 
                              x_gender_train, 
                              y_train, epochs=10, 
                              batch_size=32): 

    """
    Executes hyperparameter tuning using RandomSearch for a CNN regression model.

    :param x_train: Array or tensor containing the training images.
    :type x_train: numpy.ndarray or tensorflow.Tensor
    :param x_gender_train: Array or tensor containing the training gender data.
    :type x_gender_train: numpy.ndarray or tensorflow.Tensor
    :param y_train: Array containing the training labels (e.g., ages).
    :type y_train: numpy.ndarray
    :param epochs: Number of training epochs for each configuration, defaults to 10.
    :type epochs: int, optional
    :param batch_size: Batch size for training, defaults to 32.
    :type batch_size: int, optional

    :raises ValueError: If the dimensions of `x_train`, `x_gender_train`, and `y_train` do not match.

    :return: A tuple containing the best model built with the optimal hyperparameters and the optimal hyperparameters object.
    :rtype: tuple(keras.Model, keras_tuner.HyperParameters)
    """

    # Definizione del tuner
    tuner = Hyperband(
        build_model,
        objective='val_mae',  # Obiettivo: minimizzare l'errore assoluto medio di validazione
        max_epochs=10,
        hyperband_iterations=3,  # Numero di esecuzioni per configurazione
        directory='../tuner_results',
        project_name='cnn_regression_tuning'
    )
    
    logger.info('Starting tuning')

    #Early stopping set up
    stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=3)

    # Esecuzione del tuning
    tuner.search([x_train, x_gender_train], y_train,
                 epochs=epochs,
                 validation_split=0.2,
                 batch_size=batch_size,
                 callbacks=[stop_early])
    
    # Recupero dei migliori iperparametri trovati
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Costruzione del modello con i migliori iperparametri
    best_model = build_model(best_hps)
    
    logger.info("Best Hyperparameters:")
    for param, value in best_hps.values.items():
        logger.info(f"{param}: {value}")

    # Printing model summary
    logger.info('Summary of the best network architecture:')
    best_model.summary()

    return best_model, best_hps

def k_fold_validation(model_fn, x_train, x_gender_train, y_train, best_hps, k=5, epochs=10, batch_size=32):
    """
    Esegue la validazione incrociata con k-fold sul modello per stimare la sua performance.
    
    :param model_fn: Funzione per costruire il modello (ad esempio `build_model`).
    :type model_fn: function
    :param x_train: Dati di input (immagini).
    :type x_train: numpy.ndarray o tensorflow.Tensor
    :param x_gender_train: Dati di genere.
    :type x_gender_train: numpy.ndarray o tensorflow.Tensor
    :param y_train: Etichette di addestramento (ad esempio, età).
    :type y_train: numpy.ndarray
    :param best_hps: I migliori iperparametri ottenuti dal tuning.
    :type best_hps: keras_tuner.HyperParameters
    :param k: Numero di fold per la validazione incrociata, defaults to 5.
    :type k: int, optional
    :param epochs: Numero di epoche per l'addestramento in ogni fold, defaults to 10.
    :type epochs: int, optional
    :param batch_size: Dimensione del batch per l'addestramento, defaults to 32.
    :type batch_size: int, optional
    
    :return: Il MAE medio sui k fold.
    :rtype: float
    """
    
    logger.info(f'Inizio validazione k-fold con k={k}, epochs={epochs}, batch_size={batch_size}')
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)  # KFold per suddividere i dati in k fold
    mae_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_train), start=1):
        logger.info(f'Fold {fold}/{k}: Creazione partizioni di training e validazione')
        
        # Creazione delle partizioni per il training e la validazione
        x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
        x_gender_train_fold, x_gender_val_fold = x_gender_train[train_idx], x_gender_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        logger.info(f'Fold {fold}: Costruzione modello')
        model = model_fn(best_hps) 
        
        logger.info(f'Fold {fold}: Addestramento modello')
        model.fit([x_train_fold, x_gender_train_fold], y_train_fold,
                  epochs=epochs, batch_size=batch_size, verbose=0)
        
        logger.info(f'Fold {fold}: Valutazione modello')
        _, mae = model.evaluate([x_val_fold, x_gender_val_fold], y_val_fold, verbose=0)
        mae_scores.append(mae)
        logger.info(f'Fold {fold}: MAE = {mae:.4f}')
    
    mean_mae = np.mean(mae_scores)
    logger.info(f'Validazione k-fold completata. MAE medio: {mean_mae:.4f}')
    
    return mean_mae


def train_and_plot(model, x_train, x_gender_train, y_train, x_test,x_gender_test, y_test, epochs=2, batch_size=32, save_model=False, model_dir='trained_models'):
    """
    Funzione per allenare il modello, monitorare la training loss e la validation loss e testare sul set di test.
    
    :param model: il modello Keras da allenare.
    :param x_train: i dati di input di training.
    :param y_train: i target di training.
    :param x_val: i dati di input di validazione.
    :param y_val: i target di validazione.
    :param x_test: i dati di input di test (usati per la valutazione finale).
    :param y_test: i target di test (usati per la valutazione finale).
    :param epochs: numero di epoche di allenamento (default 50).
    :param batch_size: dimensione del batch per l'allenamento (default 32).
    """
    # Early stopping per evitare overfitting
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Allenamento del modello
    history = model.fit(
        [x_train, x_gender_train], y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    # Tracciamento della loss durante il training
    plt.ion()
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Valutazione finale sul set di test
    test_loss, test_mae = model.evaluate([x_test, x_gender_test], y_test, verbose=0)
    logger.info(f"Test Loss: {test_loss}")
    logger.info(f"Test MAE: {test_mae}")

    # Se il flag save_model è True, salva il modello
    if save_model:
        # Crea la directory se non esiste
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Salva il modello con un nome che include la data corrente (per esempio)
        model_filename = os.path.join(model_dir, 'best_model.h5')
        model.save(model_filename)
        print(f"Model saved to {model_filename}")

    return model  # restituisce il modello allenato


def plot_predictions_vs_actuals(model, x_test,x_gender_test, y_test):
    """
    Visualizza le predizioni del modello confrontandole con i target reali,
    creando un grafico in cui idealmente i punti devono allinearsi lungo la retta y = x.

    :param model: il modello Keras allenato.
    :param x_test: i dati di input di test.
    :param y_test: i target reali di test.
    """
    # Ottieni le predizioni dal modello
    y_pred = model.predict([x_test, x_gender_test])

    # Crea il grafico
    plt.ion()
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, label="Predictions", alpha=0.7)
    
    # Disegna la retta ideale y = x
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal line (y = x)')
    
    # Etichetta e titolo
    plt.title('Predictions vs Actuals (Test Data)')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.show()

def run():

    x, y, x_gender = return_dataset('/Users/moseguerini/Desktop/Test_dataset/Validation', '/Users/moseguerini/Desktop/Test_dataset/Validation_Dataset.csv')
    x_train = preprocessing_image(x[:60])
    x_val = preprocessing_image(x[60:80])
    x_test = preprocessing_image(x[80:100])
    y_train = (y[:60])
    y_val = (y[60:80])
    y_test = (y[80:100])
    x_gender_train = (x_gender[:60])
    x_gender_val = (x_gender[60:80])
    x_gender_test = (x_gender[80:100])
    # Esegui la ricerca degli iperparametri
    best_model, best_hps = run_hyperparameter_tuning(x_val, x_gender_val, y_val, epochs=10, batch_size=32)
    train_and_plot(best_model, x_train, x_gender_train, y_train, x_test, x_gender_test, y_test)
    logger.info("Training completed")
    plot_predictions_vs_actuals(best_model, x_test,x_gender_test, y_test)
    logger.info("Plotting done")
    # Esegui la validazione incrociata con k-fold
    #mean_mae = k_fold_validation(build_model, x_train, x_gender_train, y_train, best_hps, k=5, epochs=10, batch_size=32)

if __name__ == "__main__":
    run()

