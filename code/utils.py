import pathlib
from loguru import logger
from  matplotlib import pyplot as plt
import numpy as np 
import pandas as pd

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
    names_sorted = sorted(names)
    logger.info(f'Read images from the dataset')
    images = [plt.imread(name) for name in names_sorted]
    id = [name.stem for name in names_sorted]
        
    return np.array(images, dtype=object), np.array(id, dtype=np.int32) #altrimenti np.array(img, dtype=np.float32) / 255.0 se vogliamo normalizzare i pixel tra 0 e 1 per un modello di Machine Learning

def load_labels(labels_path):
    '''Function reading all the labels in a csv file, columns must be ID, boneage, male (True/False)'''

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
    """
    Load and return labels from a CSV file. The CSV must contain 'id', 'boneage', and 'male' columns.

    :param labels_path: Path to the CSV file containing label data.
    :type labels_path: str or pathlib.Path
    :raises FileNotFoundError: If the specified CSV file does not exist.
    :raises ValueError: If the CSV file does not contain the required columns.
    :return: A tuple containing three NumPy arrays: IDs, bone age values, and gender labels (1 for male, 0 for female).
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """

    feature, names = load_images(image_path)
    id, labels, gender = load_labels(labels_path)

    if not np.array_equal(names, id):
        raise ValueError('Image dataset does not correspond to labels dataset!')
    
    return feature, labels, gender

#images, labels = load_images('/Users/moseguerini/Desktop/Test_dataset/Training')

#id, boneage, gender = load_labels('/Users/moseguerini/Desktop/Dataset/Bone_Age_Validation_Set/Validation_Dataset.csv')

return_dataset('/Users/moseguerini/Desktop/Test_dataset/Training','/Users/moseguerini/Desktop/Test_dataset/training.csv')