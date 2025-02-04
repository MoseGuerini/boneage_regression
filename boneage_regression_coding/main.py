#trying to build a CNN

from cnn import conv_nn
import numpy as np
import sys
from pathlib import Path
import tensorflow as tf
from matplotlib import pyplot as plt

utils_path = Path(__file__).resolve().parent.parent / "boneage_regression"
sys.path.insert(0, str(utils_path))

from utils import run_preliminary_test, preprocessing_image
'''
image_train, real_age_train, gender_train = run_preliminary_test() 

#a sprinkle of preprocessing with matlab
Not done yet, too many tears and too little results. Let's use python instead
image_train_resized = preprocessing_image(image_train) # python preprocessing

# Verifica la forma
print(f"Shape delle immagini preprocessate: {image_train_resized.shape}")

model = conv_nn()
history = model.fit(image_train_resized, real_age_train, validation_split=0.8, epochs=10)

# Supponiamo che tu abbia X_test e y_test
boneage_prediction = model.predict(image_train_resized)

# Stampiamo i primi 10 risultati
for true_age, pred_age in zip(real_age_train[:10], boneage_prediction[:10]):
    print(f"Età corretta: {true_age} - Età inferita: {pred_age[0]}")
'''
import matlab.engine

# Avvia MATLAB Engine
eng = matlab.engine.start_matlab()

# Percorsi delle cartelle con i dati non processati e della cartella che conterrà i dati processati
input_folder_char = r'C:\Users\nicco\boneage_regression\Test_dataset\Training'
output_folder_char = r'C:\Users\nicco\boneage_regression\Preprocessed_images'
print(f"Input folder: {input_folder_char}")  # Verifica se il percorso è corretto
print(f"Output folder: {output_folder_char}")  # Verifica se il percorso è corretto
#converti la lista di caratteri in stringhe
input_folder_str = ''.join(input_folder_char)
output_folder_str = ''.join(output_folder_char)
# Chiama la funzione MATLAB
eng.matlab_images_preprocessing(input_folder_str, output_folder_str, nargout=0)
# Chiudi MATLAB Engine
eng.quit()

