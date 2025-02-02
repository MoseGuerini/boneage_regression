#trying to build a CNN

from utils import run_preliminary_test, preprocessing_image
from cnn import conv_nn
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

image_train, real_age_train, gender_train = run_preliminary_test() 
image_train_resized = preprocessing_image(image_train)

# Verifica la forma
print(f"Shape delle immagini preprocessate: {image_train_resized.shape}")

#a sprinkle of preprocessing with matlab
'''Not done yet, too many tears and too little results'''

model = conv_nn()

'''
features.shape  # (num_immagini, height, width, 3)  se RGB
                # (num_immagini, height, width)     se grayscale

labels.shape    # (num_immagini,)  # Bone age (int o float)

gender.shape    # (num_immagini,)  # 0 (femmina) / 1 (maschio) (int32)
'''
history = model.fit(image_train_resized, real_age_train, validation_split=0.8, epochs=10)

'''
model.fit(
    image_train, real_age_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)
'''
