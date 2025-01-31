import pathlib
from loguru import logger
from  matplotlib import pyplot as plt
import numpy as np 
import pandas as pd

def load_images(image_path):
    '''Function reading all the images in a directory and returns sorted images and names'''
    
    path = pathlib.Path(image_path)

    if not (path).is_dir():
            raise FileNotFoundError(f'No such file or directory {path}') 
        
    names = list((path).iterdir())
    names_sorted = sorted(names)
    logger.info(f'Read images from the dataset')
    images = [plt.imread(name) for name in names_sorted]
    labels = [name.stem for name in names_sorted]
        
    return np.array(images, dtype=object), np.array(labels, dtype=np.int32) #altrimenti np.array(img, dtype=np.float32) / 255.0 se vogliamo normalizzare i pixel tra 0 e 1 per un modello di Machine Learning

def load_labels(labels_path):
    '''Function reading all the labels in a csv file'''
    
    path = pathlib.Path(labels)

    df = pd.read_csv(path)

    pass

images, labels = load_images('/Users/moseguerini/Desktop/Test_dataset/Training')

imm = np.asarray(images[1], dtype=np.float32)
plt.imshow(imm)
plt.title(labels[1])
plt.show()