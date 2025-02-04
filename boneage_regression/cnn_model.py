from keras import layers, models
from loguru import logger

def model_cnn(input_shape,
               dropout,
               optimizer = 'adam',
               loss='mean_squared_error',
               metrics=['mae']):
    
    '''
    Creates and compiles a Convolutional Neural Network (CNN) model for regression
    using both image features and a gender input feature (0 for female, 1 for male).

    :param input_shape: The shape of the input images (e.g., (128, 128, 3))
    :type input_shape: tuple
    :param dropout: The dropout rate to apply to the dense layers
    :type dropout: float
    :param optimizer: The optimizer to use for the model. Defaults to 'adam'
    :type optimizer: str, optional
    :param loss: The loss function to use for the model. Defaults to 'mean_squared_error'
    :type loss: str, optional
    :param metrics: The list of metrics to calculate during training. Defaults to ['mae']
    :type metrics: list, optional

    :raises ValueError: If input_shape is not a valid shape or optimizer is not recognized.
    
    :return: The compiled Keras model ready for training.
    :rtype: keras.Model

    :Example:

    >>> model = model_cnn(input_shape=(128, 128, 3), dropout=0.3)
    '''
    
    # Input for image features
    input_image = layers.Input(shape=input_shape)

    # First convolutional layer
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_image)
    x = layers.MaxPooling2D((2, 2))(x)

    # Second convolutional layer
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Third convolutional layer
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Flattening and adding a dense layer
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)

    # Input for gender features (0 for female or 1 for male)
    input_gender = layers.Input(shape=(1,))

    # Features concatenation (images and gender)
    concatenated = layers.concatenate([x, input_gender])

    # Adding more dense layers
    x = layers.Dense(64, activation='relu')(concatenated)
    x = layers.Dropout(dropout)(x)  
    x = layers.Dense(32, activation='relu')(x)

    # Output layer with linear activation for a regression problem
    output = layers.Dense(1, activation='linear')(x)

    # Model creation
    full_model = models.Model(inputs=[input_image, input_gender], outputs=output)

    # Model compilation
    full_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Printing model summary
    logger.info('Summary of the network:')
    full_model.summary()

    return full_model

def train_cnn():
    pass
