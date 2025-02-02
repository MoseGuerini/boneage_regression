#trying to build a CNN

from utils import run_preliminary_test, preprocessing_image
from cnn import conv_nn
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

image_train, real_age_train, gender_train = run_preliminary_test() 

#a sprinkle of preprocessing with matlab
'''Not done yet, too many tears and too little results. Let's use python instead'''
image_train_resized = preprocessing_image(image_train)

# Verifica la forma
print(f"Shape delle immagini preprocessate: {image_train_resized.shape}")
'''
model = conv_nn()
history = model.fit(image_train_resized, real_age_train, validation_split=0.8, epochs=10)

# Supponiamo che tu abbia X_test e y_test
boneage_prediction = model.predict(image_train_resized)

# Stampiamo i primi 10 risultati
for true_age, pred_age in zip(real_age_train[:10], boneage_prediction[:10]):
    print(f"Età corretta: {true_age} - Età inferita: {pred_age[0]}")
'''
import matlab.engine
eng = matlab.engine.start_matlab()
print(eng.sqrt(4))  # Test: dovrebbe stampare 2.0
eng.quit()