from keras import layers, models
from keras import optimizers
import keras_tuner as kt
from keras_tuner import RandomSearch
from sklearn.model_selection import KFold
import numpy as np
from loguru import logger 
import sys

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
    
    # Input layers
    input_image = layers.Input(shape=(128, 128, 3))
    input_gender = layers.Input(shape=(1,))

    x = input_image

    # Number of convolutional layers (hyperparameter)
    num_conv_layers = hp.Int('num_conv_layers', min_value=2, max_value=4, step=1)

    for i in range(num_conv_layers):
        filters = hp.Int(f'filters_{i}', min_value=32, max_value=128, step=32)
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)

    # Dense layer before concatenation
    x = layers.Dense(hp.Int('dense_units', min_value=32, max_value=128, step=32), activation='relu')(x)

    # Concatenate gender input
    concatenated = layers.concatenate([x, input_gender])

    # Fully connected layers
    x = layers.Dense(hp.Int('dense_units_2', min_value=32, max_value=128, step=32), activation='relu')(concatenated)
    x = layers.Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1))(x)
    x = layers.Dense(32, activation='relu')(x)

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
    tuner = RandomSearch(
        build_model,
        objective='val_mae',  # Obiettivo: minimizzare l'errore assoluto medio
        max_trials=10,        # Numero di configurazioni da provare
        executions_per_trial=1,  # Numero di esecuzioni per configurazione
        directory='tuner_results',
        project_name='cnn_regression_tuning'
    )
    
    logger.info('Starting tuning')

    # Esecuzione del tuning
    tuner.search([x_train, x_gender_train], y_train,
                 epochs=epochs,
                 validation_split=0.2,
                 batch_size=batch_size)
    
    # Recupero dei migliori iperparametri trovati
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Costruzione del modello con i migliori iperparametri
    best_model = build_model(best_hps)
    
    # Printing model summary
    logger.info('Summary of the best network:')
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


def run():

    x_train, y_train, x_gender_train = return_dataset('/Users/moseguerini/Desktop/Test_dataset/Validation', '/Users/moseguerini/Desktop/Test_dataset/Validation_Dataset.csv')
    x_train = preprocessing_image(x_train)
    # Esegui la ricerca degli iperparametri
    best_model, best_hps = run_hyperparameter_tuning(x_train, x_gender_train, y_train, epochs=10, batch_size=32)

    # Esegui la validazione incrociata con k-fold
    mean_mae = k_fold_validation(build_model, x_train, x_gender_train, y_train, best_hps, k=5, epochs=10, batch_size=32)

if __name__ == "__main__":
    run()

