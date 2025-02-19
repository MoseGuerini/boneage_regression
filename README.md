# Boneage Regression

[![Documentation Status](https://readthedocs.org/projects/boneage-regression/badge/?version=latest)](https://boneage-regression.readthedocs.io/en/latest/?badge=latest)
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

## Preprocessing
The image are on greyscale but the size and the pixels intensity change according to the image. To stardize the dataset we renormalize the images from 0 and 1 exploiting to features which are always present in the images: the background, darker than the hand (and then set to "0"); a white letter, lighter than the hand (and then set to "1"). 
Secondly we cut as much background as we can in order to center and point out the hand respect to the background. 
Thirdly we padded images in order to obtain squared ones.
Lastly we resized them from whatever their dimension was to 256x256 in order to be able to pass them to the CNN.
The preprocessing was implemented using Matlab.
Some examples of processed and unprocessed fotos follows.

<div align="center">

| **Unpreprocessed** | **Preprocessed** |
|--------------------|------------------|
| <img src="example_images/no_preprocessing/1378.png" alt="No Preprocessing" width="200"> | <img src="example_images/preprocessing/1378.png" alt="Preprocessing" width="200"> |

</div>

<div align="center">

| **Unpreprocessed** | **Preprocessed** |
|--------------------|------------------|
| <img src="example_images/no_preprocessing/1478.png" alt="No Preprocessing" width="200"> | <img src="example_images/preprocessing/1478.png" alt="Preprocessing" width="200"> |

</div>

<div align="center">

| **Unpreprocessed** | **Preprocessed** |
|--------------------|------------------|
| <img src="example_images/no_preprocessing/1399.png" alt="No Preprocessing" width="200"> | <img src="example_images/preprocessing/1399.png" alt="Preprocessing" width="200"> |

</div>

<div align="center">

| **Unpreprocessed** | **Preprocessed** |
|--------------------|------------------|
| <img src="example_images/no_preprocessing/1418.png" alt="No Preprocessing" width="200"> | <img src="example_images/preprocessing/1418.png" alt="Preprocessing" width="200"> |

</div>

# Neural Network

## Hypermodel

## Classes
In order to improve readability by performing encapsulation we build up two classes: one to handle data and another to handle the model.
- Data: this class is designed to combine each image with its label, discard images without labels and viceversa and perform preprocessing (this function could be deactivated);
- Model: on the other hand this class takes input data and train a model (whose hyperparameter can also be searched).

# Results 
## Heat Map
As part of the analysis, we included the possibility to "visualize" what the model has learnt using a heat map, which highlights the regions of an input image  which are relevant in the decision making process.
Here are some examples:

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




