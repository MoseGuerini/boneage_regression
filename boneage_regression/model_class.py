import sys
import pathlib
from loguru import logger

import keras
from keras import callbacks, models
import keras_tuner as kt
from sklearn.model_selection import train_test_split

from hyperparameters import build_model
from plots import plot_loss_metrics, plot_predictions, plot_accuracy_threshold

# Setting logger configuration
logger.remove()
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO"
)


class CNN_Model:
    """
    A class that builds, trains, and evaluates a Convolutional Neural Network (CNN)
    for regression tasks using Keras.

    This class allows for:

    - Hyperparameter tuning via Bayesian optimization.

    - Model training with early stopping.

    - Model evaluation and prediction.

    - Saving and loading the trained model.

    Attributes:
    -----------
    X_train, X_gender_train, y_train : numpy.ndarray or pandas.DataFrame
        Training features and labels.
    X_test, X_gender_test, y_test : numpy.ndarray or pandas.DataFrame
        Test features and labels.
    model_builder : function
        Function to build the CNN model (usually `build_model`).
    trained_model : keras.Model or None
        The trained model.
    max_trials : int
        Max number of hyperparameter tuning trials (default: 10).
    overwrite : bool
        Flag to overwrite tuner results (default: False).

    Methods:
    --------
    train() :
        Performs hyperparameter tuning and training, followed by predictions.

    hyperparameter_tuning(X_val, X_gender_val, y_val, model_builder) :
        Performs Bayesian optimization for hyperparameter tuning.

    train_model(epochs=100) :
        Trains the model using the best hyperparameters.

    save_model(filename="best_model.keras") :
        Saves the trained model to a file.

    predict(model=None) :
        Generates predictions using the trained model or provided model.

    load_trained_model(model_path) :
        Loads a pre-trained model from a file.
    """
    def __init__(self, data_train, data_test, overwrite=False, max_trials=10):
        """Initialize the CNN_Model instance with training and testing datasets.

        This method sets up the training and testing datasets, along with
        configuration options for hyperparameter tuning.

        :param data_train: The training data object containing the features
                        and labels for training.
        :type data_train: DataClass
        :param data_test: The testing data object containing the features
                        and labels for testing.
        :type data_test: DataClass
        :param overwrite: Whether to overwrite existing tuning results.
                        Defaults to False.
        :type overwrite: bool, optional
        :param max_trials: The maximum number of trials for hyperparameter tuning.
                        Defaults to 10.
        :type max_trials: int, optional
        """
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
        """Return the trained model if available.

        :raises ValueError: If the model has not been trained yet.
        """
        if self._trained_model is None:
            raise ValueError(
                "The model has not been trained yet. Run `train()` first."
            )
        return

    @trained_model.setter
    def trained_model(self, model):
        """
        Set the trained model.
        """
        if not isinstance(model, keras.Model):
            raise ValueError("The model must be an instance of keras.Model.")
        self._trained_model = model

    def train(self):
        """
        Performs hyperparameter tuning followed by training of the model,
        and then generates predictions.

        This method first calls `train_model` to perform hyperparameter tuning
        and train the model, then calls `predict` to make predictions using the
        trained model.

        :return: None
        """
        self.train_model()
        self.predict()

    def hyperparameter_tuning(
            self, X_val, X_gender_val, y_val, model_builder,
            epochs=1, batch_size=64
    ):
        """
        Performs hyperparameter tuning using Bayesian optimization with an
        internal validation split for evaluation.

        :param X_val: Validation data features.
        :type X_val: numpy.ndarray or pandas.DataFrame
        :param X_gender_val: Validation data for gender features.
        :type X_gender_val: numpy.ndarray or pandas.DataFrame
        :param y_val: True labels for the validation set.
        :type y_val: numpy.ndarray or pandas.Series
        :param model_builder: Function to build the model, used by Keras Tuner.
        :type model_builder: function
        :param epochs: The number of epochs to train the model during tuning.
        :type epochs: int, optional, default is 1
        :param batch_size: The batch size to use during training.
        :type batch_size: int, optional, default is 64

        :return: A tuple of the best hyperparameters and the
                 best model found during tuning.
        :rtype: tuple
        """
        # Set directory for tuner results
        tuner_dir = pathlib.Path(__file__).resolve().parent.parent / 'tuner'
        tuner_dir.mkdir(parents=True, exist_ok=True)

        # Set project name for the tuner
        if self.overwrite:
            project_name = 'new_tuner'
        else:
            project_name = 'new_tuner'      # change later to 'best_tuner'

        # Initialize the BayesianOptimization tuner
        tuner = kt.BayesianOptimization(
            model_builder,
            objective='val_mean_absolute_error',
            max_trials=self.max_trials,
            overwrite=self.overwrite,
            directory=tuner_dir,
            project_name=project_name
        )

        stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=5)

        # Perform the hyperparameter search
        tuner.search(
            [X_val, X_gender_val], y_val,
            epochs=epochs,
            validation_split=0.2,
            batch_size=batch_size,
            callbacks=[stop_early]
        )

        # Print summary of tuner results
        tuner.results_summary()

        # Get the best hyperparameters and model
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.get_best_models()[0]

        # Log the best hyperparameters
        logger.info("Best Hyperparameters:")
        for param, value in best_hps.values.items():
            logger.info(f"Parameter: {param}, Value: {value}")

        # Log the summary of the best model
        logger.info('Summary of the best network architecture:')
        best_model.summary()

        return best_hps, best_model

    def train_model(self, epochs=70):
        """
        Trains the model using the best hyperparameters on the complete dataset,
        with an internal validation split. After training, the loss curve is plotted,
        and the model is evaluated. If requested, the model is saved to disk.

        :param epochs: The number of epochs to train the model. Defaults to 100.
        :type epochs: int

        :return: None
        :rtype: None
        """
        # Splitting training data into training and validation
        X_train, X_val, X_gender_train, X_gender_val, y_train, y_val = (
            train_test_split(self.X_train, self.X_gender_train, self.y_train,
                             test_size=0.12, random_state=1)
        )

        # Perform hyperparameters tuning
        _, best_model = self.hyperparameter_tuning(
            X_val, X_gender_val, y_val, self.model_builder
        )

        # Set up early stop
        #early_stop = callbacks.EarlyStopping(
        #    monitor='val_loss',
        #    patience=15,
        #    restore_best_weights=True,
        #    start_from_epoch=20
        #)

        # Training the model
        history = best_model.fit(
            [X_train, X_gender_train],
            y_train,
            epochs=epochs,
            batch_size=64,
            validation_split=0.1, ##rimetti callback=[early_stop]
            verbose=2
        )

        # Plot training metrics
        plot_loss_metrics(history)

        # Evaluate the model on the test dataset and log the results
        loss, mae, r2 = best_model.evaluate(
            [self.X_test, self.X_gender_test], self.y_test, verbose=2
        )

        logger.info(
            f"Evaluation on the complete dataset: Loss = {loss:.4f}"
            f"MAE = {mae:.4f}, r2 = {r2:.4f}"
        )

        # Save the trained model
        self._trained_model = best_model
        self.save_model()

    def save_model(self, filename="best_model.keras"):
        """
        Saves the trained model to a specified file.

        The model is saved in a directory named 'model', which is created if it
        does not already exist. The file is saved with the specified `filename`.

        :param filename: The name of the file where the model will be saved.
                         Defaults to "best_model.keras".
        :type filename: str

        :return: None
        :rtype: None
        """
        # Set model directory and path
        model_dir = pathlib.Path(__file__).resolve().parent.parent / 'model'
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / filename

        # Save the model and log the path
        self._trained_model.save(model_path)
        logger.info(f"Model saved in {model_path}")

    def predict(self, model=None):
        """
        Returns predictions from the model for the input data.

        If `model` is not provided, it uses the trained model stored in
        `self._trained_model`. If the required input data
        (`X_test` and `X_gender_test`) are not provided, it defaults
        to using the stored data in `self`.

        :param model: The model to use for prediction. If `None`, the trained
                      model stored in `self._trained_model` will be used.
        :type model: keras.Model

        :raises ValueError: If no model is available for prediction.

        :return: Predicted values for the test data.
        :rtype: numpy.ndarray
        """
        model = model if model is not None else self._trained_model

        if model is None:
            raise ValueError("No model available for prediction.")

        # Get predictions
        y_pred = model.predict([self.X_test, self.X_gender_test])
        y_pred = y_pred.flatten()

        # Plot prediction and error distribution
        plot_predictions(self.y_test, y_pred)
        plot_accuracy_threshold(y_pred, self.y_test)

        return y_pred

    def load_trained_model(self, model_path):
        """
        Load a trained model from a file.

        This method loads a pre-trained model from the specified file path
        and stores it in the instance variable `_trained_model`.

        :param model_path: Path to the file containing the trained model.
        :type model_path: str

        :raises ValueError: If the model cannot be loaded due to an
                            invalid path or corrupted file.

        :return: None
        :rtype: None
        """
        self._trained_model = models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
