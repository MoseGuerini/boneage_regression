from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization, ReLU, Dense
from keras.models import Model

def conv_nn():
    inputs = Input(shape=(128, 128, 3))  # Modifica la forma dell'input a (128, 128, 3)
    '''
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
<<<<<<< HEAD
    outputs = Dense(1, activation='linear')(inputs)  # Usa sigmoid per classificazione binaria
=======
    outputs = Dense(1)(inputs)  # Usa attivazione lineare per regressione
     '''
     

    hidden=  Conv2D(5,(5,5), activation='relu')(inputs)
    hidden= MaxPooling2D((3,3))(hidden)
    hidden=  Conv2D(3,(3,3), activation='relu')(hidden)
    hidden= MaxPooling2D((3,3))(hidden)
    #hidden=  Conv2D(3,(3,3), activation='relu')(hidden)
    hidden= Flatten()(hidden)
    hidden=  Dense(50, activation='relu')(hidden)
    hidden=  Dense(20, activation='relu')(hidden)
    hidden=  Dense(20, activation='relu')(hidden)
    #hidden=  Dense(40, activation='relu')(hidden)
    #hidden=  Dense(30, activation='relu')(hidden)
    outputs = Dense(1)(hidden)  # Usa attivazione lineare per regressione
>>>>>>> d3e5aba19785e2fc73cf584663b89df7af12cdbd
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])  # MSE per regressione e MAE per errore assoluto medio
    
    model.summary()
    
    return model

