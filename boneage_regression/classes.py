import os
import numpy as np
import matplotlib.pyplot as plt

from loguru import logger
from keras import layers, models, optimizers, callbacks
import keras_tuner as kt

from hyperparameters import build_model

class Model:
    """
    Classe per la regressione dell'età da radiografie della mano tramite CNN.

    Gestisce il tuning degli iperparametri, l'allenamento e la valutazione del modello.

    Attributi
    ---------
    x : np.ndarray
        Array contenente le immagini preprocessate (dimensione 256x256x3).
    y : np.ndarray
        Array contenente le età dei pazienti (target).
    x_gender : np.ndarray
        Array contenente le feature relative al genere.
    max_trials : int
        Numero massimo di tentativi per l'hyperparameter tuning.
    overwrite : bool
        Se True forza il tuning completo (cioè non utilizza iperparametri pre-salvati).
    k : int
        Numero di fold per eventuali validazioni incrociate (non utilizzato esplicitamente qui).
    batch_size : int
        Batch size per l'allenamento.
    tuner_dir : str
        Directory in cui salvare i risultati del tuning.
    model_dir : str
        Directory in cui salvare il modello allenato.
    best_model : keras.Model
        Modello ottenuto con i migliori iperparametri.
    best_hps : keras_tuner.HyperParameters
        Iperparametri ottimali trovati.
    _selected_model : str
        Path del modello salvato (per il successivo caricamento).
    """
    def __init__(self, x, y, x_gender, fast=True, max_trials=10, k=5, batch_size=32,
                 tuner_dir='tuner_results', model_dir='trained_models'):
        self.x = x
        self.y = y
        self.x_gender = x_gender
        self.max_trials = max_trials
        self.overwrite = not fast  # Se fast=True, si evita il tuning completo
        self.model_builder = build_model
        self.k = k
        self.batch_size = batch_size
        self.tuner_dir = tuner_dir
        self.model_dir = model_dir
        self._selected_model = None

    def hyperparameter_tuning(self, epochs=10):
        """
        Esegue l'hyperparameter tuning e
        impiegando un validation_split interno per la valutazione.
        """
        tuner = kt.BayesianOptimization(
            self.build_model,
            objective='val_mae',
            max_epochs=epochs,
            hyperband_iterations=3,
            directory=self.tuner_dir,
            project_name='cnn_regression_tuning'
        )
        stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=3)
        tuner.search([self.x, self.x_gender], self.y,
                     epochs=epochs,
                     validation_split=0.2,
                     batch_size=self.batch_size,
                     callbacks=[stop_early])
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        best_model = tuner.get_best_models()[0]

        logger.info("Best Hyperparameters:")
        for param, value in best_hps.values.items():
            logger.info(f"{param}: {value}")

        # Printing model summary
        logger.info('Summary of the best network architecture:')
        best_model.summary()

        return best_model, best_hps

    def train_model(self, epochs=10, save_model=True):
        """
        Allena il modello (definito dai migliori iperparametri) sui dati completi,
        utilizzando un validation_split interno. Al termine, mostra il grafico della loss,
        valuta il modello e, se richiesto, lo salva su disco.
        """
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = self.best_model.fit(
            [self.x, self.x_gender],
            self.y,
            epochs=epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )

        # Plot delle curve di loss
        plt.figure(figsize=(10,6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training e Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        loss, mae = self.best_model.evaluate([self.x, self.x_gender], self.y, verbose=0)
        print(f"Valutazione sul dataset completo: Loss = {loss:.4f}, MAE = {mae:.4f}")

        if save_model:
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            model_path = os.path.join(self.model_dir, 'best_model.h5')
            self.best_model.save(model_path)
            self._selected_model = model_path
            print(f"Modello salvato in {model_path}")

    def train(self, epochs=10):
        """
        Esegue in sequenza il tuning degli iperparametri e l'allenamento del modello.
        """
        self.hyperparameter_tuning(epochs=epochs)
        self.train_model(epochs=epochs, save_model=True)

    def get_predictions(self, x=None, x_gender=None):
        """
        Restituisce le predizioni del modello per i dati in input.
        Se x o x_gender non sono specificati, usa i dati completi memorizzati in self.x e self.x_gender.
        """
        if x is None or x_gender is None:
            x = self.x
            x_gender = self.x_gender
        return self.best_model.predict([x, x_gender])

    @property
    def selected_model(self):
        """
        Restituisce il path del modello salvato. Se il modello non è stato ancora allenato,
        solleva un'eccezione.
        """
        if self._selected_model is None:
            raise ValueError("Il modello non è ancora stato allenato. Esegui train() prima.")
        return self._selected_model

