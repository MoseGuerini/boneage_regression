import pathlib
from loguru import logger
from  matplotlib import pyplot as plt
import numpy as np 
import pandas as pd
import tensorflow as tf

def load_images(image_path):
    """
    Load and return images and their corresponding IDs from a specified directory.

    :param image_path: Path to the directory containing image files.
    :type image_path: str or pathlib.Path
    :raises FileNotFoundError: If the specified directory does not exist.
    :return: A tuple containing a NumPy array of loaded images and a NumPy array of image IDs.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    
    path = pathlib.Path(image_path)

    if not (path).is_dir():
            raise FileNotFoundError(f'No such file or directory {path}') 
        
    names = list((path).iterdir())
    names_sorted = sorted(names, key=lambda x: int(x.stem))
    logger.info(f'Read images from the dataset')
    images = [plt.imread(name) for name in names_sorted]
    id = [name.stem for name in names_sorted]
        
    return np.array(images, dtype=object), np.array(id, dtype=np.int32) #altrimenti np.array(img, dtype=np.float32) / 255.0 se vogliamo normalizzare i pixel tra 0 e 1 per un modello di Machine Learning

def load_labels(labels_path):
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

    try:
        logger.info(f'Read labels from csv file')
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"File {labels_path} does not exist")
        return None

    df.columns = df.columns.str.lower()
    miss_cols = [col for col in req_columns if col not in df.columns]

    if miss_cols:
        raise ValueError(f'The file must contain {miss_cols} column(s)')
    
    id = df['id'].to_numpy()
    boneage = df['boneage'].to_numpy()
    gender = df['male'].astype(int).to_numpy() #1 if True () 0 if False

    return id, boneage, gender
    
def return_dataset(image_path, labels_path):
    '''Return arrays containing features and labels for training the CNN'''

    feature, names = load_images(image_path)
    id, labels, gender = load_labels(labels_path)

    if not np.array_equal(names, id):
        diff = np.where(names != id)
        raise ValueError(f'''Image dataset does not correspond to labels dataset: 
                         image[{diff[0]}] = {names[diff[0]]} while labels[{diff[0]}] = {id[diff[0]]}!''')
    
    return feature, labels, gender

def run_preliminary_test():

    current_path = pathlib.Path.cwd()

    while current_path.name != 'boneage_regression':
        current_path = current_path.parent

    return return_dataset('../Test_dataset/Training', '../Test_dataset/training.csv')

def preprocessing_image(images):
    '''Hopefully we will use matlab soon'''
    
    print(f"Shape delle immagini prima del preprocessing: {images.shape}")
    target_size = (128, 128)  # Imposta una dimensione fissa

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