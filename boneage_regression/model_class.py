import pathlib
import sys

import keras
import keras_tuner as kt
import numpy as np
import matplotlib.pyplot as plt
from keras import callbacks, models
from loguru import logger
from sklearn.model_selection import KFold

from hyperparameters import build_model
from plots import plot_loss_metrics, plot_predictions, plot_error_distribution
from utils import (
    make_gradcam_heatmap,
    overlay_heatmap,
    save_image,
    log_training_summary,
    get_last_conv_layer_name,
)


# Setting logger configuration
logger.remove()
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO"
)


def readonly_property(attr_name: str) -> property:
    """
    Creates a read-only property for a specified attribute.

    Attempts to modify the property will log a warning but will not
    update the attribute.

    :param attr_name: The name of the attribute to create a read-only
                    property for.
    :type attr_name: str

    :return: A property that can be used to access the specified attribute
             but prevents modification.
    :rtype: property
    """
    def getter(self):
        return getattr(self, f"_{attr_name}")

    def setter(self, value):
        logger.warning(f"{attr_name} dataset cannot be modified!")
    return property(getter, setter)


class CnnModel:
    """
    A class that represents a Convolutional Neural Network (CNN) model
    for training, evaluation, and prediction tasks, with support for
    hyperparameter tuning using Keras Tuner and cross-validation.

    This class enables the user to train and evaluate a CNN model,
    perform hyperparameter tuning, generate predictions, and visualize results
    such as Grad-CAM heatmaps. The model is trained using k-fold
    cross-validation, and the best model is selected based on performance
    metrics like MSE, MAE and R-squared (R²).

    Attributes:
        _X_train: The feature data for training.

        _X_gender_train: The gender-specific feature data for training.

        _y_train: The target labels for training.

        _X_test: The feature data for testing.

        _X_gender_test: The gender-specific feature data for testing.

        _y_test: The target labels for testing.

        max_trials: The maximum number of trials for hyperparameter tuning.

        overwrite_tuner: Whether to overwrite existing tuner results.

        overwrite_model: Whether to overwrite existing trained models.

        model_builder: A function that builds the model.

        _trained_model: The best-trained model selected after cross-validation.

    Methods:
        __init__(data_train, data_test, overwrite=False, max_trials=10):
            Initializes the CNN_Model instance.

        trained_model:
            Property that returns the trained model if available,
            raising an exception if not trained yet.

        train():
            Trains the model using hyperparameter tuning and
            generates predictions.

        hyperparameter_tuning(
            X_train, X_gender_train, y_train,
            X_val, X_gender_val, y_val, model_builder,
            fold, epochs=50, batch_size=64
        ):
            Performs hyperparameter tuning using Bayesian optimization
            with an internal validation split.

        train_on_fold(fold, train_idx, val_idx):
            Trains and evaluates the model for a single fold in
            cross-validation, saving the model.

        train_model(k=5):
            Trains the model using k-fold cross-validation and selects
            the best model based on lowest MAE.

        save_model(model=None, filename="best_model.keras"):
            Saves the trained model to a specified file.

        predict(model=None):
            Generates predictions for the test data using the specified
            or trained model.

        visualize_gradcam_batch():
            Visualizes Grad-CAM heatmaps overlaid on test images with
            the lowest and highest prediction errors.
    """
    def __init__(
            self,
            data_train,
            data_test,
            overwrite_tuner=False,
            overwrite_model=False,
            max_trials=10):
        """
        Initialize the CNN_Model instance with training and testing datasets.

        This method sets up the training and testing datasets, along with
        configuration options for hyperparameter tuning.

        :param data_train: The training data object containing the features
                        and labels for training.
        :type data_train: DataLoader
        :param data_test: The testing data object containing the features
                        and labels for testing.
        :type data_test: DataLoader
        :param overwrite_tuner: Whether to overwrite existing tuning results.
                        Default: False.
        :type overwrite: bool, optional
        :param overwrite_model: Whether to overwrite existing treained models.
                        Default: False.
        :type overwrite: bool, optional
        :param max_trials: The maximum number of trials for hyperparameter
                        tuning. Defaults: 10.
        :type max_trials: int, optional
        """
        self._X_train = data_train.X
        self._X_gender_train = data_train.X_gender
        self._y_train = data_train.y

        self._X_test = data_test.X
        self._X_gender_test = data_test.X_gender
        self._y_test = data_test.y

        self.max_trials = max_trials
        self.overwrite_tuner = overwrite_tuner
        self.overwrite_model = overwrite_model
        self.model_builder = build_model
        self._trained_model = None

    # Making dataset attributes read-only
    X_train = readonly_property("X_train")
    X_gender_train = readonly_property("X_gender_train")
    y_train = readonly_property("y_train")
    X_test = readonly_property("X_test")
    X_gender_test = readonly_property("X_gender_test")
    y_test = readonly_property("y_test")

    @property
    def trained_model(self):
        """
        Get the trained model if available.

        :raises ValueError: If the model has not been trained yet.
        :return: The trained Keras model.
        :rtype: keras.Model
        """
        if self._trained_model is None:
            raise ValueError(
                "The model has not been trained yet. Run `train()` first."
            )
        return self._trained_model

    @trained_model.setter
    def trained_model(self, model):
        """
        Set the trained model.

        :param model: The trained Keras model to set.
        :type model: keras.Model
        :raises ValueError: If the provided model is not an instance of
                            keras.Model.
        """
        if not isinstance(model, keras.Model):
            raise ValueError("The model must be an instance of keras.Model.")
        self._trained_model = model

    def train(self):
        """
        Performs hyperparameters tuning, model training using k-fold
        cross-validation and generates Grad-CAM visualizations.

        :return: None
        """
        self.train_model()
        self.visualize_gradcam_batch()

    def hyperparameter_tuning(
            self, X_train, X_gender_train, y_train,
            X_val, X_gender_val, y_val,
            model_builder, fold, epochs=50, batch_size=64
    ):
        """
        Performs hyperparameter tuning using Bayesian optimization with an
        internal validation split for evaluation.

        :param X_train: Training data features.
        :type X_train: numpy.ndarray

        :param X_gender_train: Training gender features.
        :type X_gender_train: numpy.ndarray

        :param y_train: True labels for the training set.
        :type y_train: numpy.ndarray

        :param X_val: Validation data features.
        :type X_val: numpy.ndarray

        :param X_gender_val: Validation gender features.
        :type X_gender_val: numpy.ndarray

        :param y_val: True labels for the validation set.
        :type y_val: numpy.ndarray

        :param model_builder: Function to build the model, used by Keras Tuner.
        :type model_builder: function

        :param epochs: The number of epochs to train the model during tuning.
        :type epochs: int, optional, default is 50

        :param batch_size: The batch size to use during training.
        :type batch_size: int, optional, default is 64

        :return: A tuple of the best hyperparameters found during tuning
                 and the model built using this hyperparameters.
        :rtype: tuple
        """
        # Set directory for tuner results
        base_path = pathlib.Path(__file__).resolve().parent.parent
        tuner_subdir = f'tuner_{fold}'
        tuner_dir = base_path / 'Tuner' / tuner_subdir
        tuner_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the BayesianOptimization tuner
        tuner = kt.BayesianOptimization(
            model_builder,
            objective='val_loss',
            max_trials=self.max_trials,
            overwrite=self.overwrite_tuner,
            directory=tuner_dir,
            project_name='new_tuner'
        )

        # Set up early stop
        stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=10)

        # Perform the hyperparameter search
        tuner.search(
            [X_train, X_gender_train], y_train,
            epochs=epochs,
            validation_data=([X_val, X_gender_val], y_val),
            batch_size=batch_size,
            verbose=1,
            callbacks=[stop_early]
        )

        # Print summary of tuner results
        tuner.results_summary()

        # Get the best hyperparameters and model
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = model_builder(best_hps)

        return best_hps, best_model

    def train_on_fold(self, fold, train_idx, val_idx):
        """
        Trains and evaluates the model for a single fold of cross-validation.

        This method splits the training data for the current fold, performs
        hyperparameter tuning, trains the best model, evaluates it on the
        test set, and saves the model.
        The method also logs the loss, mean absolute error (MAE),
        and R-squared (R²) for the fold.

        :param fold: The current fold number.
        :type fold: int

        :param train_idx: Indices of the training data for this fold.
        :type train_idx: numpy.ndarray

        :param val_idx: Indices of the validation data for this fold.
        :type val_idx: numpy.ndarray

        :return: A tuple containing the best hyperparameters, the best model,
                the loss, mean absolute error (MAE), and R-squared (R²) for the
                fold evaluated on the test set.
        :rtype: tuple
        """
        # Split training dataset for the current fold
        X_train_fold = self.X_train[train_idx]
        X_val_fold = self.X_train[val_idx]

        X_gender_train_fold = self.X_gender_train[train_idx]
        X_gender_val_fold = self.X_gender_train[val_idx]

        y_train_fold = self.y_train[train_idx]
        y_val_fold = self.y_train[val_idx]

        # Hyperparameter tuning for this fold
        logger.info(f"Performing hyperparameter search for fold {fold}")
        best_hps, best_model = self.hyperparameter_tuning(
            X_train_fold, X_gender_train_fold, y_train_fold,
            X_val_fold, X_gender_val_fold, y_val_fold,
            self.model_builder, fold
        )

        # Train the best model on this fold
        logger.info(f"Performing training for fold {fold}")
        history = best_model.fit(
            [X_train_fold, X_gender_train_fold],
            y_train_fold,
            epochs=300,
            batch_size=64,
            validation_data=([X_val_fold, X_gender_val_fold], y_val_fold),
            verbose=1
        )

        # Plot the training loss and metrics
        plot_loss_metrics(history, fold=fold)

        # Evaluate the model on the test set and log the results
        loss, mae, r2 = best_model.evaluate(
            [self._X_test, self._X_gender_test], self._y_test, verbose=2
        )
        logger.info(
            f"Evaluation on fold {fold}: Loss = {loss:.4f} "
            f"MAE = {mae:.4f}, R2 = {r2:.4f}"
        )

        # Save the model for this fold
        self.save_model(best_model, filename=f"model_fold{fold}.keras")

        return best_hps, best_model, loss, mae, r2

    def train_model(self, k=5):
        """
        Trains the model using k-fold cross-validation.

        The method performs k-fold cross-validation on the training data.
        It loops over each fold, calling the `train_on_fold` method for
        training and evaluation. The results, including the best
        hyperparameters, model, loss, mean absolute error (MAE), and
        R-squared (R²) for each fold, are stored.
        After training on all folds, the model with the lowest loss (MSE)
        is selected as the final model.

        :param k: The number of folds for cross-validation. Defaults to 5.
        :type k: int, optional

        :return: None
            This method does not return anything. It trains the model and
            stores the best model in `self._trained_model`.
        """
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        best_hps_list = []
        model_list = []
        loss_list = []
        mae_list = []
        r2_list = []
        fold = 1

        # Loop over each fold and train using the train_on_fold method
        for train_idx, val_idx in kf.split(self._X_train):
            logger.info(f"Entering fold {fold}/{k}")

            model_filename = (
                pathlib.Path(__file__).parent.parent / f"Models/model_fold{fold}.keras"
            )

            # If overwrite_model = False and model exist, skip training
            if model_filename.exists() and not self.overwrite_model:
                # Log and load pre-trained model
                logger.info(
                    f"Model for fold {fold} already exists. "
                    "Skipping training."
                )
                best_model = models.load_model(model_filename, compile=True)

                # Evaluate the model on the test dataset
                loss, mae, r2 = best_model.evaluate(
                    [self._X_test, self._X_gender_test], self._y_test, verbose=2
                )

                # Store and log results
                lists = [loss_list, mae_list, r2_list, model_list]
                values = [loss, mae, r2, best_model]

                for lst, val in zip(lists, values):
                    lst.append(val)

                logger.info(
                    f"Evaluation on fold {fold}: "
                    f"Loss = {loss:.4f}, MAE = {mae:.4f}, r2 = {r2:.4f}"
                )
            else:
                # Train new model
                (best_hps, best_model, loss, mae, r2) = (
                    self.train_on_fold(fold, train_idx, val_idx)
                )

                # Evaluate the model on the test dataset
                loss, mae, r2 = best_model.evaluate(
                    [self._X_test, self._X_gender_test], self._y_test, verbose=2
                )

                # Store and log results
                lists = [best_hps_list, loss_list, mae_list, r2_list, model_list]
                values = [best_hps, loss, mae, r2, best_model]

                for lst, val in zip(lists, values):
                    lst.append(val)
                logger.info(
                    f"Evaluation on fold {fold}: "
                    f"Loss = {loss:.4f}, MAE = {mae:.4f}, r2 = {r2:.4f}"
                )

            fold += 1

        logger.info("Training completed for all folds, logging summary:")

        log_training_summary(best_hps_list, loss_list, mae_list, r2_list)

        # Finding the model with the minimum MSE
        min_mse_index = np.argmin(loss_list)
        self._trained_model = model_list[min_mse_index]
        logger.info(
            f"Selected model for predictions from fold {min_mse_index + 1} "
            f"with MSE = {loss_list[min_mse_index]:.2f}"
        )

    def predict(self, model=None):
        """
        Generates predictions for the test data using the specified model.

        If no model is provided, the method uses the trained model stored in
        `self._trained_model`.

        :param model: The model to use for prediction. If `None`, the trained
                    model stored in `self._trained_model` will be used.
        :type model: keras.Model, optional

        :raises ValueError: If no model is available for prediction.

        :return: Predicted values for the test set.
        :rtype: numpy.ndarray
        """
        model = model if model is not None else self._trained_model

        if model is None:
            raise ValueError("No model available for prediction.")

        # Get predictions
        y_pred = model.predict([self._X_test, self._X_gender_test])
        y_pred = y_pred.flatten()

        # Plot prediction and error distribution
        plot_predictions(self._y_test, y_pred)
        plot_error_distribution(y_pred, self._y_test)

        return y_pred

    def visualize_gradcam_batch(self):
        """
        Visualizes Grad-CAM heatmaps overlaid on the 5 test images with
        the lowest and highest prediction errors.

        This method selects 5 images with the smallest prediction errors and
        5 images with the largest prediction errors from the test set.
        Grad-CAM heatmaps are generated and overlaid on the original
        images to highlight areas of focus.
        The images are displayed in a 2x5 grid layout.

        The resulting image is saved as 'heat_map.png' locally.

        :return: None
            This method does not return any value. It displays and saves
            the Grad-CAM visualizations.
        """
        last_conv_layer_name = get_last_conv_layer_name(self._trained_model)

        y_pred = self.predict()

        # Computes error between actual and predicted values
        errors = np.abs(y_pred - self._y_test)

        # Selecting the images based on the prediction error
        sorted_indices = np.argsort(errors)
        best_indices = sorted_indices[:5]   # Get 5 best images
        worst_indices = sorted_indices[-5:]  # Get 5 worse images
        indices = np.concatenate([best_indices, worst_indices])
        errors = errors[indices]

        # Create figure with subplots (2 rows x 3 columns)
        fig, axes = plt.subplots(2, 5, figsize=(15, 7))

        for i, idx in enumerate(indices):
            row, col = divmod(i, 5)

            # Prepare the image and auxiliary data for the model
            img_array = [
                np.expand_dims(self._X_test[idx], axis=0),
                np.expand_dims(self._X_gender_test[idx], axis=0)
            ]

            # Generate Grad-CAM heatmap
            heatmap = make_gradcam_heatmap(
                img_array,
                self._trained_model,
                last_conv_layer_name)

            # Prepare the original image
            original_img = (self._X_test[idx] * 255).astype(np.uint8)

            # Overlay the heatmap on the original image
            superimposed_img = overlay_heatmap(original_img, heatmap)

            # Show images in corresponding subplots
            axes[row, col].imshow(superimposed_img)
            axes[row, col].set_title(
                f"True = {self._y_test[idx]} m. Pred = {y_pred[idx]:.1f} m."
            )
            axes[row, col].axis("off")  # Remove axes

        # Adjust the layout and show the figure
        plt.tight_layout()

        # Save the image
        save_image('heat_map.png')
        plt.close()

    def save_model(self, model=None, filename="best_model.keras"):
        """
        Saves the trained model to a specified file.

        The model is saved in a directory named 'models',
        which is created if it does not already exist. The model is saved with
        the specified `filename`.

        :param model: The model to save. If `None`, the trained model stored in
                    `self._trained_model` will be used.
        :type model: keras.Model, optional

        :param filename: The name of the file where the model will be saved.
                        Defaults to "best_model.keras".
        :type filename: str, optional

        :return: None
            This method does not return anything. It saves the model to a file.
        """
        model = model if model is not None else self._trained_model
        # Set model directory and path
        model_dir = pathlib.Path(__file__).resolve().parent.parent / 'Models'
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / filename

        # Save the model and log the path
        model.save(model_path)
        logger.info(f"Model saved in {model_path}")
