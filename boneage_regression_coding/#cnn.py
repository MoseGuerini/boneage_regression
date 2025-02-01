#cnn.py
from keras.layers import Input,Conv2D,MaxPooling2D,Flatten,BatchNormalization

def cnn ():
    
    inputs=Input(shape=(128,128,1,))
    hidden=  Conv2D(5,(5,5), activation='relu')(inputs)     
    hidden= MaxPooling2D((3,3))(hidden)
    hidden=  Conv2D(3,(3,3), activation='relu')(hidden)
    hidden= MaxPooling2D((3,3))(hidden)
    hidden=  Conv2D(3,(3,3), activation='relu')(hidden)
    hidden= MaxPooling2D((3,3))(hidden)
    hidden=  Conv2D(3,(3,3), activation='relu')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden= MaxPooling2D((3,3))(hidden)
    hidden=  Conv2D(3,(3,3), activation='relu')(hidden)
    hidden= MaxPooling2D((3,3))(hidden)
    hidden= Flatten()(hidden)
    
    for _ in range(5):
        # Primo layer di convoluzione (3x3x3)
        inputs = Conv3D(5, (3, 3), strides=1, padding='same')(inputs)
        inputs = ReLU()(inputs)
    
        # Secondo layer di convoluzione (3x3x3)
        inputs = Conv3D(3, (3, 3), strides=1, padding='same')(inputs)
        inputs = BatchNormalization()(inputs)
        inputs = ReLU()(inputs)
    
        # MaxPooling (2x2x2)
        inputs = MaxPooling3D(pool_size=(2, 2), strides=2, padding='same')(inputs)
        
    # Flatten e layer fully connected (Dense)
    inputs = Flatten()(inputs)
    inputs = Dense(128, activation='relu')(inputs)
    inputs = Dense(64, activation='relu')(inputs)
    outputs = Dense(1, activation='sigmoid')(inputs)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss, optimizer='adam',metrics=['accuracy'])

    model.summary()