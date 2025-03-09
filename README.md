# Bone age Regression

[![Documentation Status](https://readthedocs.org/projects/boneage-regression/badge/?version=latest)](https://boneage-regression.readthedocs.io/en/latest/?badge=latest)
![GitHub repo size](https://img.shields.io/github/repo-size/MoseGuerini/boneage_regression)
![CircleCI](https://circleci.com/gh/MoseGuerini/boneage_regression/tree/main.svg?style=shield)

The aim of this reporsitory is to build and train a convolutional neural network (CNN) for a deep learning based regression. Starting from hands x-rays the CNN will be able to infer patiences' ages. This neaural network will be developed using both Python and Matlab.
# Table of contents
+ [Data](#Data)
  + [Preprocessing](#Preprocessing)
+ [Neural Network](#Neural_Network)
  + [Classes](#Classes)
+ [Results](#Results)
  + [Heat Map](#Heat_Map)
+ [Usage](#Usage)  


# Data
The dataset is composed of 14233 images, coming from patiences whose age range from 0 to 216 months. The 46% of the patinces are female (label "0") and 54% are male (label "1") and the mean age is 127 months.
You are able to download data yourself using the following link: https://www.rsna.org/rsnai/ai-image-challenge/RSNA-Pediatric-Bone-Age-Challenge-2017

## Preprocessing
The image are on greyscale but the size and the pixels intensity are different from one image to another. To stardize the dataset we renormalize the images from 0 and 1 exploiting to features which are always present in the images: the background, darker than the hand (and then set to "0"); a white letter, lighter than the hand (and then set to "1"). 
Secondly we cut as much background as we can in order to center the hand. 
Thirdly we padded images in order to obtain squared ones.
Lastly we resized them from whatever their dimension was to 256x256 in order to be able to pass them to the CNN.
The preprocessing was implemented using Matlab.
Some examples of processed and unprocessed fotos follows.

<div align="center">

| **Unpreprocessed** | **Preprocessed** |
|--------------------|------------------|
| <img src="Example_images/No_preprocessing/1378.png" alt="No Preprocessing" width="200"> | <img src="Example_images/Preprocessing/1378.png" alt="Preprocessing" width="200"> |

</div>

<div align="center">

| **Unpreprocessed** | **Preprocessed** |
|--------------------|------------------|
| <img src="Example_images/No_preprocessing/1478.png" alt="No Preprocessing" width="200"> | <img src="Example_images/Preprocessing/1478.png" alt="Preprocessing" width="200"> |

</div>

<div align="center">

| **Unpreprocessed** | **Preprocessed** |
|--------------------|------------------|
| <img src="Example_images/No_preprocessing/1399.png" alt="No Preprocessing" width="200"> | <img src="Example_images/Preprocessing/1399.png" alt="Preprocessing" width="200"> |

</div>

<div align="center">

| **Unpreprocessed** | **Preprocessed** |
|--------------------|------------------|
| <img src="Example_images/No_preprocessing/1418.png" alt="No Preprocessing" width="200"> | <img src="Example_images/Preprocessing/1418.png" alt="Preprocessing" width="200"> |

</div>

# Neural Network
The hypermodel consists of a variable number of convolutional blocks (Conv2D, BatchNormalization and MaxPooling), followed by a Flatten and a Dense (plus BatchNormalization) layer. The output is then concatenated with the gender features. Following there are a variable number of Dense plus Dropout layers and a final Dense layer with linear activation.
## Hypermodel
The user can insert custom values for the hyperparameters tuning. Namely the hyperparameters are: <br>
• `Conv. layers`: number of convolutional layers of the network. <br>
• `Conv. filters`: number of convolutional filters in the first conv. layer. The number of conv. filters grows linearly each successive layer.<br>
• `Dense depth`: number of dense layer(s) after feature concatenation. <br>
• `Dropout rate`: dropout rate of the final dense layer(s). <br>

The default hyperparameters values are shown in the following table: <br>

| Hyperparameters   |     Values      | 
| ----------------  | ----------------|
| `conv_layers`     |  3, 4, 5        | 
| `conv_filters`    |  8, 16, 32      | 
| `dense_depth`     |  1, 3, 5        | 
| `dropout_rate`    |  0.1, 0.2, 0.3  | 

## Classes
In order to improve readability by performing encapsulation we build up two classes: one to handle data and another to handle the model.
- Data: this class is designed to combine each image with its label, discard images without labels and viceversa and perform preprocessing (this function could be deactivated);
- Model: on the other hand this class takes input data and train a model (whose hyperparameters can also be searched).

# Results 
## Heat Map
As part of the analysis, we include the possibility to "visualize" what the model has learnt using a heat map, which highlights the regions of input images which are relevant in the decision making process.
Here are some examples:

![Descrizione dell'immagine](Example_images/heat_map.png)

# Usage
Simply download this repository and run using default parameters.
```python
cd boneage_regression\boneage_regression
python main.py
```
In case you are running this code for the first time remember to install the requirements.
```python
pip install -r requirements.txt
```

In order to select custom parameters, refer to the help:
```python
python main.py -h

usage: main.py [-h] [-fp] [-p] [-o] [-cl  [...]] [-cf  [...]] [-dd  [...]] [-dr  [...]] [-sf]

This script performs bone age prediction using a machine learning regression model. It accepts input parameters for model configuration and the
dataset folder path. If you pass a dataset folder remember that it must contain: 1) Two separate folders for training and test images named 'Training'
and 'Test'. 2) Two .csv files with the corresponding labels named 'training.csv' and 'test.csv'. Each CSV file must contain three columns named 'ID',
'boneage', 'male'.

options:
  -h, --help            show this help message and exit
  -fp , --folder_path   Path to the directory containing training and test images as well as csv files with the labels. Default: Preprocessed_images
  -p , --preprocessing  If False avoid image preprocessing. Default: False
  -o , --overwrite      If False avoid hyperparameters search and use the pre-saved hyperpar. Default: False
  -cl  [ ...], --conv_layers  [ ...]
                        List of values for the hypermodel number of conv2d layers Default [3, 4, 5]
  -cf  [ ...], --conv_filters  [ ...]
                        List of values for the hypermodel first conv2d number of filters. Default [8, 16, 32]
  -dd  [ ...], --dense_depth  [ ...]
                        List of values for the hypermodel depth of final dense layers Default [1, 2, 3]
  -dr  [ ...], --dropout_rate  [ ...]
                        List of values for the dropout rate of the final dense layers Default [0.1, 0.2, 0.3]
  -sf , --searching_fraction 
                        Fraction of the hyperparameters space explored during hypermodel search. Default: 0.25
```

Plese note that in order to be able to preprocess the images you must have **`Matlab`** and the `Python` module **`matlabengine`** installed (they are **not** included in the project's requirements). If you want to run the code without installing **`Matlab`** and **`matlabengine`**, simply run the script using the default values for `folder_path` (`Preprocessed_images`) and `preprocessing` (`False`), this will skip the preprocessing using already preprocessed images in the Preprocessed_images folder.

In order to pass a list of hyperparameters, separate them with a space like this:
```python
python main.py -cl 2 3 -cf 32 -dd 1 2 -dr 0.2 0.3
```
## Attribution

This work builds upon the research presented in the following paper:

- **Halabi SS, Prevedello LM, Kalpathy-Cramer J, et al.** (2018). The RSNA Pediatric Bone Age Machine Learning Challenge. *Radiology*, 290(2), 498-503. [DOI](https://doi.org/10.1148/radiol.2018180736).

For more details, please refer to the original paper.

The original dataset can be downloaded at https://www.rsna.org/rsnai/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017





