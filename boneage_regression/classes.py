import sys
import pathlib
from loguru import logger

import keras
from keras import callbacks
import keras_tuner as kt
from keras.models import load_model

from hyperparameters import build_model
from plots import plot_loss_metrics, plot_predictions

# Setting logger configuration 
logger.remove()  
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO"
)

class CNN_Model:

    def __init__(self, data, overwrite=False, max_trials=10):
        self._X = data.x
        self._X_gender = data.X_gender
        self._y = data.y
        self.max_trials = max_trials
        self.overwrite = overwrite
        self.model_builder = build_model
        self._trained_model = None

    @property
    def X(self):
        return self._X
    
    @X.setter
    def X(self, X_new):
        logger.warning('X dataset cannot be modified!')
    
    @property
    def X_gender(self):
        return self._X_gender
    
    @X_gender.setter
    def X_gender(self, X_gender_new):
        logger.warning('X_gender dataset cannot be modified!')
    
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, y_new):
        logger.warning('y dataset cannot be modified!')

    @property
    def trained_model(self):
        if self._trained_model is None:
            raise ValueError("Il modello non è ancora stato allenato. Esegui train() prima.")
        return self._trained_model
    
    @trained_model.setter
    def trained_model(self, model):
        """
        Set the trained model. This setter can include checks to validate the model.
        """
        if not isinstance(model, keras.Model):
            raise ValueError("The model must be an instance of keras.Model.")
        self._trained_model = model
    
    def train(self):
        """
        Esegue in sequenza il tuning degli iperparametri e l'allenamento del modello, plottando le predizioni.
        """
        self.hyperparameter_tuning()
        self.train_model()
        self.predict()

    def hyperparameter_tuning(self, X_val, X_gender_val, y_val, model_builder, epochs=50, batch_size=64):
        """
        Esegue l'hyperparameter tuning e
        impiegando un validation_split interno per la valutazione.
        """

        tuner_dir = pathlib.Path(__file__).resolve().parent.parent / 'tuner'
        tuner_dir.mkdir(parents=True, exist_ok=True)

        if self.overwrite:
            project_name = 'tuner_new'
        else:
            project_name = 'tuner_old'

        tuner = kt.BayesianOptimization(
            model_builder,
            objective='val_mae',
            max_trials = self.max_trials,
            overwrite = self.overwrite,
            directory=tuner_dir,
            project_name=project_name
        )

        stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=3)

        tuner.search([X_val, X_gender_val], y_val,
                     epochs=epochs,
                     validation_split=0.2,
                     batch_size=batch_size,
                     callbacks=[stop_early])
        
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        best_model = tuner.get_best_models()[0]

        logger.info("Best Hyperparameters:")
        for param, value in best_hps.values.items():
            logger.info(f"Parameter: {param}, Value: {value}")

        # Printing model summary
        logger.info('Summary of the best network architecture:')
        best_model.summary()

        return best_model, best_hps

    def train_model(self, epochs=10):
        """
        Allena il modello (definito dai migliori iperparametri) sui dati completi,
        utilizzando un validation_split interno. Al termine, mostra il grafico della loss,
        valuta il modello e, se richiesto, lo salva su disco.
        """
        X_val, X_gender_val, y_val = 0, 0, 0
        X_train, X_gender_train, y_train = 0, 0, 0
        X_test, X_gender_test, y_test = 0, 0, 0

        _, best_model = self.hyperparameter_tuning(X_val, X_gender_val, y_val, self.model_builder)

        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        history = best_model.fit(
            [X_train, X_gender_train],
            y_train,
            epochs=epochs,
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=2
        )

        plot_loss_metrics(history)

        loss, mae, mse = best_model.evaluate([X_test, X_gender_test], y_test, verbose=2)
        logger.info(f"Evaluation on the complete dataset: Loss = {loss:.4f}, MAE = {mae:.4f}, MSE = {mse:.4f}")

        model_dir = pathlib.Path(__file__).resolve().parent.parent / 'model'
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "best_model.h5"

        best_model.save(model_path)
        logger.info(f"Modello saved at {model_path}")
        self._trained_model = best_model

    def predict(self, model = None):
        """
        Restituisce le predizioni del modello per i dati in input.
        Se x o x_gender non sono specificati, usa i dati completi memorizzati in self.x e self.x_gender.
        """
        model = model if model is not None else self._trained_model

        if model is None:
            raise ValueError("No model available for prediction.")
    
        X_test, X_gender_test, y_test = 0, 0, 0

        # Ottieni le predizioni dal modello
        y_pred = model.predict([X_test, X_gender_test])

        plot_predictions(y_test, y_pred)

        return y_pred

    def load_trained_model(self, model_path):
        """
        Load a trained model from a file.
        """
        self._trained_model = load_model(model_path)
        logger.info(f"Model loaded from {model_path}")

