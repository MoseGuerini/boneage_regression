from utils import return_dataset, load_images, load_labels
import numpy as np
import tensorflow as tf

# Controlla se una GPU è disponibile
if tf.config.list_physical_devices('GPU'):
    print("GPU disponibile")
else:
    print("GPU non disponibile")

image_path = r'C:\Users\nicco\Desktop\Preprocessed_dataset_prova\Preprocessed_foto'  # Inserisci il percorso alla cartella delle immagini
labels_path = r'C:\Users\nicco\Desktop\Preprocessed_dataset_prova\training.csv'  # Inserisci il percorso al file delle etichette

num_images = 30; #specifica quante immagini correi che fossero caricate

# Esegui la funzione
try:
    # Carica una parte delle immagini e i nomi
    features, names = load_images(image_path)
    feature, labels_filtered, gender_filtered = return_dataset(image_path, labels_path)
    print(f"Names: {type(labels_filtered)}; genders: {type(gender_filtered)}")
    print(f"feature type: {type(feature)}, labels type: {type(labels_filtered)}")
    print([type(x) for x in labels_filtered[:10]])  # Controlla i primi 10 elementi
    print([type(x) for x in gender_filtered[:10]])
    print([type(x) for x in feature[:10]])
    """
    gender_filtered = np.array(gender_filtered)  # Conversione in array NumPy
    labels_filtered = np.array(labels_filtered)  # Converte in array NumPy
    """
    print(feature.dtype, gender_filtered.dtype, labels_filtered.dtype)
    #print(f"Features: {features}")
    print(f"Names: {names}")
    print(f"Ages: {labels_filtered}")
    print(f"Gender: {gender_filtered}")
except ValueError as e:
    print(f"Error: {e}")
