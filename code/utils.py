import pathlib
from loguru import logger
from  matplotlib import pyplot as plt
import numpy as np 
import pandas as pd

def load_images(image_path):
    '''Function reading all the images in a directory and returns sorted images and id's'''
    
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
    '''Function that returns the dataset as Y array (labels) and X array (features) 
       and checks that they correspond with each other'''

    feature, names = load_images(image_path)
    id, labels, gender = load_labels(labels_path)

    if not np.array_equal(names, id):
        raise ValueError('Image dataset does not correspond to labels dataset!')
    
    return feature, labels, gender

#images, labels = load_images('/Users/moseguerini/Desktop/Test_dataset/Training')

#id, boneage, gender = load_labels('/Users/moseguerini/Desktop/Dataset/Bone_Age_Validation_Set/Validation_Dataset.csv')

return_dataset('/Users/moseguerini/Desktop/Test_dataset/Training','/Users/moseguerini/Desktop/Test_dataset/training.csv')