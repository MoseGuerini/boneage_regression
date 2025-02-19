# Boneage Regression

[![Documentation Status](https://readthedocs.org/projects/boneage-regression/badge/?version=latest)](https://boneage-regression.readthedocs.io/en/latest/?badge=latest)
![CircleCI](https://circleci.com/gh/MoseGuerini/boneage_regression/tree/main.svg?style=shield)

The aim of this reporsitory is to build and train a convolutional neural network (CNN) for a deep learning based regression. Starting from hands x-rays the CNN will be able to infer patiences' ages. This neaural network will be developed using
# Table of contents
+ [Data](#Data)
  + [Preprocessing](#Preprocessing)
+ [Neural Network](#Neural_Network)


## Data
The dataset is composed of 14233 images, coming from patiences whose age range from 0 to 216 months. The 46% of the patinces are female (label "0") and 54% are male (label "1") and the mean age is 127 months. 
The image are on greyscale but the size and the pixels intensity change according to the image. To stardize the dataset we renormalize the images from 0 and 1 exploiting to features which are always present in the images: the background, darker than the hand (and then set to "0"); a white letter, lighter than the hand (and then set to "1"). 
Secondly we cut as much background as we can in order to center and point out the hand respect to the background. 
Thirdly we padded images in order to obtain squared ones.
Lastly we resized them from whatever their dimension was to 256x256 in order to be able to pass them to the CNN.
Some examples of processed and unprocessed fotos follows.

![Descrizione dell'immagine](images/mia_immagine.png)

## Neural Network
First of all we tried to find the best combinations of our network hyperparameters. Due to the high number of the hyperparameters combinations (METTERE QUNANTE SARANNO ALLA FINE) we preferred permorming bayesian search instead of grid search. This meant that we did not explore all the possible combination but, once the target number of trial is set, we followed a gradient-discendent method to reach a local minimum.
After that we trained a CNN composed according to the best hyperparameters found. The available dataset was composed of 15k images which we splitted into two subgroups: about 10% for testing and the remaining 90% for training and validation.

