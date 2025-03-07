"""Hypermodel builder and hps setter"""
from keras import layers, models, optimizers

img_size = (256, 256, 1)


def set_hyperp(hyperp_dict):
    """
    Sets the hyperparameters dictionary as a global variable, making it
    accessible to the hypermodel.

    :param hyperp_dict: Dictionary containing hyperparameters.
    :type hyperp_dict: dict
    """
    global hyperp
    hyperp = hyperp_dict


def hyperp_space_size(hyperp):
    """
    Calculates the total number of possible hyperparameter combinations
    based on the user-defined hyperparameter space.

    :return: The total number of hyperparameter combinations.
    :rtype: int
    """
    size = 1
    for key in hyperp:
        size *= len(hyperp[key])
    return size


def build_model(hp):
    """
    Builds a Convolutional Neural Network (CNN) model for regression using
    hyperparameter tuning.

    The model consists of two branches:
    - An image-processing branch with convolutional layers.
    - A gender feature branch that is concatenated with the extracted
    image features.

    The final output is a regression prediction.

    :param hp: Hyperparameter tuning instance from Keras Tuner, used to define
               model parameters.
    :type hp: keras_tuner.HyperParameters

    :return: Compiled Keras model built with the specified hyperparameters.
    :rtype: keras.Model
    """
    # First Branch (images features)
    input_image = layers.Input(shape=img_size)
    x = input_image

    # Hyperparameters definitions
    hp_num_conv_layers = hp.Choice('conv_layers', hyperp['conv_layers'])
    hp_filters = hp.Choice('conv_filters', hyperp['conv_filters'])
    hp_dropout = hp.Choice('dropout_rate', hyperp['dropout_rate'])
    hp_dense_depth = hp.Choice('dense_depth', hyperp['dense_depth'])
    num_dense = 256

    # Variable number of convolutional layers
    for i in range(hp_num_conv_layers):
        x = layers.Conv2D(
            hp_filters * (i + 1),
            (3, 3),
            activation='relu',
            padding='same'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

    # Flattening
    x = layers.Flatten()(x)

    # Dense layer before concatenation
    x = layers.Dense(num_dense, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Second Branch (gender features)
    input_gender = layers.Input(shape=(1,))

    # Concatenate two branches
    x = layers.concatenate([x, input_gender])

    # Fully connected layers
    for i in range(hp_dense_depth):
        x = layers.Dense(int(num_dense/(i+1)), activation='relu')(x)
        x = layers.Dropout(hp_dropout)(x)

    # Output layer for regression
    output = layers.Dense(1, activation='linear')(x)

    # Create model
    model = models.Model(inputs=[input_image, input_gender], outputs=output)

    # Compile with hyperparameter tuning for learning rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='mean_squared_error',
        metrics=['mean_absolute_error', 'r2_score']
    )

    return model
