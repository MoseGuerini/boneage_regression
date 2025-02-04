from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dense, ReLU
from keras.models import Model
from matplotlib import pyplot as plt
from utils import run_preliminary_test, load_images, load_labels

# Definizione input
inputs = Input(shape=(128, 128, 1))

# Costruzione della CNN
x = Conv2D(5, (5, 5), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2))(x)  # Ridotto a (2,2) per evitare shrinking eccessivo
x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

# Controllo dimensioni prima di Flatten
model_temp = Model(inputs=inputs, outputs=x)
model_temp.summary()  # Stampa la dimensione delle feature map

# Seconda parte della CNN
for _ in range(3):  # Ridotto da 5 a 3 per evitare shrinking eccessivo
    x = Conv2D(5, (3, 3), strides=1, padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(3, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

# Flatten e Fully Connected
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(1, activation='linear')(x)

# Creazione del modello
model = Model(inputs=inputs, outputs=outputs)

# Compilazione
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Stampa il sommario del modello
model.summary()

# Caricamento dati
data = load_images('../Test_dataset/Training')
labels = load_labels('../Test_dataset/training.csv')

# Training del modello
history = model.fit(data, labels, validation_split=0.5, epochs=100)

# Plot della loss
plt.plot(history.history["val_loss"], label='Validation Loss')
plt.plot(history.history["loss"], label='Training Loss')
plt.yscale('log')
plt.legend()
plt.show()



import numpy as np

sample_input = np.random.rand(1, 128, 128, 1)  # Crea un'immagine casuale di input
prediction = model.predict(sample_input)

print("Prediction:", prediction)
print("Type of prediction:", type(prediction))
print("Shape of prediction:", prediction.shape)
