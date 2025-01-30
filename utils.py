import pathlib
from loguru import logger
from  matplotlib import pyplot as plt
import numpy as np 

def load_images(image_path, subdirs):
    '''Function reading all the images in a directory containing two subdirectories'''
    
    images = []
    labels = []
    for sdir in subdirs:
        sdir = str(sdir)
        if not (pathlib.Path(image_path) / sdir).is_dir():
               raise FileNotFoundError(f'No such file or directory {pathlib.Path(image_path) / sdir}') 
        
        names = list((pathlib.Path(image_path) / sdir).iterdir())
        logger.info(f'Read images from the {sdir} dataset')
        images.extend([plt.imread(name) for name in names])
        labels.extend([name.stem for name in names])
        
    return np.array(images, dtype=object), np.array(labels, dtype=np.int32)  #altrimenti np.array(img, dtype=np.float32) / 255.0 se vogliamo normalizzare i pixel tra 0 e 1 per un modello di Machine Learning

def load_labels(labels_path):
    '''Function reading all the labels in a csv file'''
    pass

load_images('/Users/moseguerini/Desktop/Test_dataset', ['Training', 'Validation'])
