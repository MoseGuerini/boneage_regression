import pathlib
from loguru import logger
from  matplotlib import pyplot as plt
import numpy as np 
import pandas as pd
import tensorflow as tf

import pathlib
import numpy as np
import matplotlib.pyplot as plt

def load_images(image_path, num_images=None):
    """
    Load a subset of images and return them with their corresponding IDs from a specified directory.

    :param image_path: Path to the directory containing image files.
    :type image_path: str or pathlib.Path
    :param num_images: The number of images to load. If None, all images are loaded. Default is None.
    :type num_images: int or None
    :raises FileNotFoundError: If the specified directory does not exist.
    :return: A tuple containing a NumPy array of loaded images and a NumPy array of image IDs.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    
    path = pathlib.Path(image_path)

    if not path.is_dir():
        raise FileNotFoundError(f'No such file or directory: {path}')
    
    # Elenco delle immagini nella cartella, ordinate per nome
    names = list(path.iterdir())
    names_sorted = sorted(names, key=lambda x: int(x.stem))

    # Se è stato specificato un numero massimo di immagini, carica solo quello
    if num_images is not None:
        names_sorted = names_sorted[:num_images]

    logger.info(f'Read images from the dataset')
    
    # Carica le immagini
    images = [plt.imread(name) for name in names_sorted]
    
    # Estrai gli ID dalle immagini
    id = [name.stem for name in names_sorted]

    return np.array(images, dtype=object), np.array(id, dtype=np.int32)


def load_labels(labels_path, num_path=None):
    """
    Load and return labels from a CSV file. The CSV must contain 'id', 'boneage', and 'male' columns.

    :param labels_path: Path to the CSV file containing label data.
    :type labels_path: str or pathlib.Path
    :raises FileNotFoundError: If the specified CSV file does not exist.
    :raises ValueError: If the CSV file does not contain the required columns.
    :return: A tuple containing three NumPy arrays: IDs, bone age values, and gender labels (1 for male, 0 for female).
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """

    req_columns = ['id', 'boneage', 'male']
    path = pathlib.Path(labels_path)

    # Verifica che il file esista prima di cercare di leggerlo
    if not path.is_file():
        print(f"File {labels_path} does not exist")
        return None, None, None

    try:
        logger.info(f'Read labels from csv file')
        df = pd.read_csv(path)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None, None, None

    # Normalizza i nomi delle colonne per evitare problemi con maiuscole/minuscole
    df.columns = df.columns.str.lower()

    # Verifica che tutte le colonne richieste siano presenti
    miss_cols = [col for col in req_columns if col not in df.columns]
    if miss_cols:
        raise ValueError(f'The file must contain {miss_cols} column(s)')

    # Estrai le informazioni necessarie dal dataframe
    id = df['id'].to_numpy()
    boneage = df['boneage'].to_numpy()
    gender = df['male'].astype(int).to_numpy()  # 1 if True (male), 0 if False (female)

    return id, boneage, gender

    
def return_dataset(image_path, labels_path, num_images=None):
    '''Return arrays containing features and labels for training the CNN, 
    loading only a subset of images.

    :param num_images: Number of images to load, default is 200.
    :type num_images: int
    '''

    # Carica tutte le immagini e i nomi
    feature, names = load_images(image_path, num_images)

    # Carica tutte le etichette e i generi
    id, labels, gender = load_labels(labels_path)

    # Verifica che le etichette siano state caricate correttamente
    if id is None or labels is None or gender is None:
        raise ValueError("Failed to load labels. Please check the labels file.")

    # Assicurati che il numero di immagini caricate sia sufficiente
    if num_images is not None and len(feature) < num_images:
        raise ValueError(f"Only {len(feature)} images are available, but you requested {num_images}.")


    # Filtra le etichette per mantenere solo quelle che corrispondono ai nomi delle immagini caricate
    id_filtered = [i for i in id if i in names]
    labels_filtered = [labels[i] for i, label in enumerate(id) if id[i] in names]
    gender_filtered = [gender[i] for i, lab in enumerate(id) if id[i] in names]

    # Verifica che il numero di etichette filtrate corrisponda al numero di immagini caricate
    if len(id_filtered) != len(names):
        raise ValueError("Number of labels does not match the number of images after filtering.")

    # Verifica che i nomi delle immagini corrispondano agli ID delle etichette
    if not np.array_equal(names, id_filtered):
        diff = np.where(names != id_filtered)
        raise ValueError(f'''Image dataset does not correspond to labels dataset: 
                         image[{diff[0]}] = {names[diff[0]]} while labels[{diff[0]}] = {id_filtered[diff[0]]}!''')

    # Restituisci i dati filtrati
    return feature, labels_filtered, gender_filtered



def run_preliminary_test():

    current_path = pathlib.Path.cwd()

    while current_path.name != 'boneage_regression':
        current_path = current_path.parent

    return return_dataset('../Test_dataset/Training', '../Test_dataset/training.csv')

def preprocessing_image(images):
    '''Hopefully we will use matlab soon'''
    
    print(f"Shape delle immagini prima del preprocessing: {images.shape}")
    target_size = (256, 256)  # Imposta una dimensione fissa

    # Assicurati che le immagini siano in scala di grigi e poi espandi a 3 canali
    image_rgb = []
    for img in images:
        if len(img.shape) == 2:  # Se l'immagine è in scala di grigi (1 canale)
            img_rgb = np.stack([img] * 3, axis=-1)  # Crea 3 canali duplicati
        else:
            img_rgb = img  # Se già ha 3 canali (RGB), la lascio così com'è
        image_rgb_resized = tf.image.resize(img_rgb, target_size).numpy()
        image_rgb.append(image_rgb_resized)

    # Converti in un array NumPy
    image_rgb = np.array(image_rgb, dtype=np.float32)
    
    # Verifica la forma
    print(f"Shape delle immagini preprocessate: {image_rgb.shape}")
    
    return image_rgb