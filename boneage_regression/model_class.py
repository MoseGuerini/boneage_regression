import sys
import pathlib
from loguru import logger

import keras
from keras import callbacks, models
import keras_tuner as kt
from sklearn.model_selection import train_test_split

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

    def __init__(self, data_train, data_test, overwrite=False, max_trials=10):

        self._X_train = data_train.X
        self._X_gender_train = data_train.X_gender
        self._y_train = data_train.y

        self._X_test = data_test.X
        self._X_gender_test = data_test.X_gender
        self._y_test = data_test.y

        self.max_trials = max_trials
        self.overwrite = overwrite
        self.model_builder = build_model
        self._trained_model = None

    @property
    def X_train(self):
        return self._X_train
    
    @X_train.setter
    def X_train(self, X_train_new):
        logger.warning('X_train dataset cannot be modified!')
    
    @property
    def X_gender_train(self):
        return self._X_gender_train
    
    @X_gender_train.setter
    def X_gender_train(self, X_gender_train_new):
        logger.warning('X_gender_train dataset cannot be modified!')
    
    @property
    def y_train(self):
        return self._y_train
    
    @y_train.setter
    def y_train(self, y_train_new):
        logger.warning('y_train dataset cannot be modified!')

    @property
    def X_test(self):
        return self._X_test
    
    @X_test.setter
    def X_test(self, X_test_new):
        logger.warning('X_test dataset cannot be modified!')
    
    @property
    def X_gender_test(self):
        return self._X_gender_test
    
    @X_gender_test.setter
    def X_gender_test(self, X_gender_test_new):
        logger.warning('X_gender_test dataset cannot be modified!')
    
    @property
    def y_test(self):
        return self._y_test
    
    @y_test.setter
    def y_test(self, y_test_new):
        logger.warning('y_test dataset cannot be modified!')
    
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
        self.train_model()
        self.predict()

    def hyperparameter_tuning(self, X_val, X_gender_val, y_val, model_builder, epochs=10, batch_size=64):
        """
        Esegue l'hyperparameter tuning e
        impiegando un validation_split interno per la valutazione.
        """

        tuner_dir = pathlib.Path(__file__).resolve().parent.parent / 'tuner'
        tuner_dir.mkdir(parents=True, exist_ok=True)

        if self.overwrite:
            project_name = 'new_tuner'
        else:
            project_name = 'best_tuner'

        tuner = kt.BayesianOptimization(
            model_builder,
            objective='val_mean_absolute_error',
            max_trials = self.max_trials,
            overwrite = self.overwrite,
            directory=tuner_dir,
            project_name=project_name
        )

        stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=5)

        tuner.search([X_val, X_gender_val], y_val,
                     epochs=epochs,
                     validation_split=0.2,
                     batch_size=batch_size,
                     callbacks=stop_early)
        
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        best_model = tuner.get_best_models()[0]

        logger.info("Best Hyperparameters:")
        for param, value in best_hps.values.items():
            logger.info(f"Parameter: {param}, Value: {value}")

        # Printing model summary
        logger.info('Summary of the best network architecture:')
        best_model.summary()

        return  best_hps, best_model

    def train_model(self, epochs=50):
        """
        Allena il modello (definito dai migliori iperparametri) sui dati completi,
        utilizzando un validation_split interno. Al termine, mostra il grafico della loss,
        valuta il modello e, se richiesto, lo salva su disco.
        """
        X_train, X_val, X_gender_train, X_gender_val, y_train, y_val  = train_test_split(self.X_train, self.X_gender_train, self.y_train, test_size=0.2, random_state=1)

        _, best_model = self.hyperparameter_tuning(X_val, X_gender_val, y_val, self.model_builder)

        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = best_model.fit(
            [X_train, X_gender_train],
            y_train,
            epochs=epochs,
            batch_size=64,
            validation_split=0.2,
            callbacks=early_stop,
            verbose=2
        )

        plot_loss_metrics(history)

        loss, mae, mse = best_model.evaluate([self.X_test, self.X_gender_test], self.y_test, verbose=2)
        logger.info(f"Evaluation on the complete dataset: Loss = {loss:.4f}, MAE = {mae:.4f}, MSE = {mse:.4f}")

        self._trained_model = best_model

        self.save_model()


    def save_model(self, filename="best_model.h5"):
        """Salva il modello addestrato."""
        model_dir = pathlib.Path(__file__).resolve().parent.parent / 'model'
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / filename
        self._trained_model.save(model_path)
        logger.info(f"Modello salvato in {model_path}")


    def predict(self, model = None):
        """
        Restituisce le predizioni del modello per i dati in input.
        Se x o x_gender non sono specificati, usa i dati completi memorizzati in self.x e self.x_gender.
        """
        model = model if model is not None else self._trained_model

        if model is None:
            raise ValueError("No model available for prediction.")

        # Ottieni le predizioni dal modello
        y_pred = model.predict([self.X_test, self.X_gender_test])

        plot_predictions(self.y_test, y_pred)

        return y_pred

    def load_trained_model(self, model_path):
        """
        Load a trained model from a file.
        """
        self._trained_model = models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")

