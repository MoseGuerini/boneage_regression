import pathlib
from loguru import logger
from  matplotlib import pyplot as plt
import numpy as np 

def load_images(image_path, subdirs):
    '''Function reading all the images in a directory containing two subdirectories'''
    
    images = []
    labels = []
    for sdir in subdirs:
        if not (pathlib.Path(image_path) / str(sdir)).is_dir():
               raise FileNotFoundError(f'No such file or directory {pathlib.Path(image_path) / str(sdir)}') 
        
        names = list((pathlib.Path(image_path) / str(sdir)).iterdir())
        logger.info(f'Read images from the {sdir} dataset')
        images += [plt.imread(name) for name in names]
        labels += [name.stem for name in names]
    
    return 

def load_labels(labels_path):
    '''Function reading all the labels in a csv file'''
    pass

