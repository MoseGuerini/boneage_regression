"""Hypermodel builder and hps setter"""
from keras import layers, models, optimizers

img_size=(256,256,3)


def set_hyperp(hyperp_dict):
    """Hps dictionary is set to be a global variable so that it is accessible to the hypermodel as well.
    ...
    Parameters
    ----------
    hyperp_dict: dict
        hps dictionary"""
    global hyperp
    hyperp  = hyperp_dict

def hyperp_space_size():
    """Calculate hyperparameters space size (based on the user-selected combinations)."""
    size = 1
    for key in hyperp:
        size *= len(hyperp[key])
    return size

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
    input_image = layers.Input(shape=img_size)
    x = input_image

    # Hyperparameters definitions
    hp_num_conv_layers = hp.Choice('conv_layers', hyperp['conv_layers'])
    hp_filters = hp.Choice('conv_filters', hyperp['conv_filters'])
    hp_dropout = hp.Choice('dropout_rate', hyperp['dropout_rate'])
    hp_dense_units = hp.Choice('dense_units', hyperp['dense_units'])
    hp_dense_depth = hp.Choice('dense_depth', hyperp['dense_depth'])
    num_dense = 256

    # Variable number of convolutional layers
    for i in range(hp_num_conv_layers):
        x = layers.Conv2D(hp_filters*(i+1), (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)

    # Dense layer before concatenation
    x = layers.Dense(hp_dense_units, activation='relu')(x)
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
        metrics=['mean_absolute_error', 'mean_squared_error']
    )

    return model