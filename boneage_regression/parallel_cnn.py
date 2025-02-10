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
    input_image = layers.Input(shape=(64, 64, 3))
    x = input_image

    # Number of convolutional layers (hyperparameter)
    hp_num_conv_layers = hp.Int('num_conv_layers', min_value=3, max_value=3, step=1)
    hp_filters = hp.Int(f'filters', min_value=32, max_value=32, step=32)

    for i in range(hp_num_conv_layers):
        x = layers.Conv2D(hp_filters*(i+1), (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)

    # Dense layer before concatenation
    x = layers.Dense(hp.Int('dense_units', min_value=32, max_value=32, step=32), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Second Branch (gender features)
    input_gender = layers.Input(shape=(1,))
    y = input_gender

    # Concatenate two branches
    concatenated = layers.concatenate([x, y])

    # Fully connected layers
    num_dense = hp.Int('dense_units_2', min_value=32, max_value=32, step=32)
    x = layers.Dense(num_dense, activation='relu')(concatenated)
    x = layers.Dropout(hp.Float('dropout', min_value=0.1, max_value=0.1, step=0.1))(x)
    x = layers.Dense(int(num_dense/2), activation='relu')(x)

    # Output layer for regression
    output = layers.Dense(1, activation='linear')(x)

    # Create model
    model = models.Model(inputs=[input_image, input_gender], outputs=output)

    # Compile with hyperparameter tuning for learning rate
    model.compile(
        optimizer=optimizers.Adam(hp.Float('learning_rate', min_value=1e-2, max_value=1e-2, sampling='log')),
        loss='mean_squared_error',
        metrics=['mae']
    )

    return model

def run_hyperparameter_tuning_parallel(x_train, x_gender_train, y_train, epochs=1, batch_size=64): 
    tuner = Hyperband(
        build_model,
        objective='val_mae',
        max_epochs=1,
        executions_per_trial=2,  # Esegui più prove per trial in parallelo
        directory='../tuner_results',
        project_name='cnn_regression_tuning'
    )
    
    logger.info('Starting tuning')

    # Setup per EarlyStopping
    stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=3)

    # Esegui la ricerca degli iperparametri in parallelo
    tuner.search([x_train, x_gender_train], y_train,
                 epochs=epochs,
                 validation_split=0.2,
                 batch_size=batch_size,
                 callbacks=[stop_early])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hps

from concurrent.futures import ProcessPoolExecutor
import numpy as np

def k_fold_validation_parallel(model_fn, x_train, x_gender_train, y_train, best_hps, k=5, epochs=10, batch_size=32):
    """
    Esegue la validazione incrociata con k-fold in parallelo per ogni fold.
    """
    logger.info(f'Inizio validazione k-fold con k={k}, epochs={epochs}, batch_size={batch_size}')
    
    kf = KFold(n_splits=k, shuffle=True, random_state=5)  # KFold per suddividere i dati in k fold
    mae_scores = []
    
    def train_fold(train_idx, val_idx, fold):
        """
        Funzione per allenare e valutare il modello per un singolo fold.
        """
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
        return mae

    # Usa un ProcessPoolExecutor per parallelizzare i fold
    with ProcessPoolExecutor() as executor:
        futures = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(x_train), start=1):
            futures.append(executor.submit(train_fold, train_idx, val_idx, fold))

        for future in futures:
            mae = future.result()
            mae_scores.append(mae)
            logger.info(f'Fold {fold}: MAE = {mae:.4f}')
    
    mean_mae = np.mean(mae_scores)
    logger.info(f'Validazione k-fold completata. MAE medio: {mean_mae:.4f}')
    
    return mean_mae



def train_and_plot(model, x_train, x_gender_train, y_train, x_val, y_val, x_test,x_gender_test, y_test, epochs=5, batch_size=64, save_model=False, model_dir='trained_models'):
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
        validation_data=(x_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )

    # Tracciamento della loss durante il training
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

    x, y, x_gender = return_dataset(r'C:\Users\nicco\Desktop\Preprocessed_dataset_prova\Preprocessed_foto', r'C:\Users\nicco\Desktop\Preprocessed_dataset_prova\train.csv')
    print(f"x shape: {x.shape}, y shape: {y.shape}")
    """
    x = x[..., np.newaxis]  # Aggiunge un asse di dimensione 1
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int32)  # Se le etichette sono numeriche
    print(f"x shape after expansion: {x.shape}")  # Dovrebbe essere (18, 128, 128, 1)
    for i, img in enumerate(x):
        print(f"Image {i} shape: {img.shape}")
    """
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
    best_model, best_hps = run_hyperparameter_tuning_parallel(x_val, x_gender_val, y_val, epochs=10, batch_size=9)
    train_and_plot(best_model, x_train, x_gender_train, y_train, x_test, y_test)
    plot_predictions_vs_actuals(best_model, x_test,x_gender_test, y_test)
    # Esegui la validazione incrociata con k-fold
    #mean_mae = k_fold_validation(build_model, x_train, x_gender_train, y_train, best_hps, k=5, epochs=10, batch_size=32)

if __name__ == "__main__":
    run()

