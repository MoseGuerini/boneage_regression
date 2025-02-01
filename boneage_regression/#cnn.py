#cnn.py

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

from keras.layers import Input,Conv2D,MaxPooling2D,Flatten,BatchNormalization


inputs=Input(shape=(128,128,1,))
hidden=  Conv2D(5,(5,5), activation='relu')(inputs)     
hidden= MaxPooling2D((3,3))(hidden)
hidden=  Conv2D(3,(3,3), activation='relu')(hidden)
hidden= MaxPooling2D((3,3))(hidden)
hidden=  Conv2D(3,(3,3), activation='relu')(hidden)
hidden= MaxPooling2D((3,3))(hidden)
hidden=  Conv2D(3,(3,3), activation='relu')(hidden)
hidden = BatchNormalization()(hidden)
hidden= MaxPooling2D((3,3))(hidden)
hidden=  Conv2D(3,(3,3), activation='relu')(hidden)
hidden= MaxPooling2D((3,3))(hidden)
hidden= Flatten()(hidden)
    
for _ in range(5):
    # Primo layer di convoluzione (3x3x3)
    inputs = Conv3D(5, (3, 3), strides=1, padding='same')(inputs)
    inputs = ReLU()(inputs)

    # Secondo layer di convoluzione (3x3x3)
    inputs = Conv3D(3, (3, 3), strides=1, padding='same')(inputs)
    inputs = BatchNormalization()(inputs)
    inputs = ReLU()(inputs)

    # MaxPooling (2x2x2)
    inputs = MaxPooling3D(pool_size=(2, 2), strides=2, padding='same')(inputs)
        
# Flatten e layer fully connected (Dense)
inputs = Flatten()(inputs)
inputs = Dense(128, activation='relu')(inputs)
inputs = Dense(64, activation='relu')(inputs)
outputs = Dense(1, activation='sigmoid')(inputs)
    
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss=loss, optimizer='adam',metrics=['accuracy'])

model.summary()

data = load_images('C:\\Users\\nicco\\Desktop\\Training examples')
label = load_labels('C:\\Users\\nicco\\Desktop\\train_example')
    
history=model.fit(data,labels,validation_split=0.5,epochs=100)