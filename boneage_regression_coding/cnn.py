from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization, ReLU, Dense
from keras.models import Model

def cnn():
    inputs = Input(shape=(128, 128, 1))  # Modifica in base al numero di canali (1 per scala di grigi, 3 per RGB)
    
    for _ in range(5):
        # Primo layer di convoluzione (3x3x1 o 3x3x3, in base ai canali)
        inputs = Conv2D(32, (3, 3), strides=1, padding='same')(inputs)
        inputs = ReLU()(inputs)
        
        # Secondo layer di convoluzione (3x3)
        inputs = Conv2D(64, (3, 3), strides=1, padding='same')(inputs)
        inputs = BatchNormalization()(inputs)
        inputs = ReLU()(inputs)
        
        # MaxPooling (2x2)
        inputs = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(inputs)
    
    # Flatten e layer fully connected (Dense)
    inputs = Flatten()(inputs)
    inputs = Dense(128, activation='relu')(inputs)
    inputs = Dense(64, activation='relu')(inputs)
    outputs = Dense(1, activation='linear')(inputs)  # Usa sigmoid per classificazione binaria
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # Impostato per classificazione binaria
    
    model.summary()

print('finito')
