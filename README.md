# boneage_regression
This is the Computing Methods for Experimental Physics and Data Analysis examination project. Tha aim of this reporsitory is handling medical images and using them to train a CNN. The CNN would then be able to infer patients' age looking at their x-rays.
# Table of contents
1. [Preprocessing](#Preprocessing)
2. [NN training](#NN-training)


## Preprocessing
The image were hihgly different one from another. This is the reason why we did a little preprocessing before giving the images to the CNN.
First of all we increase the contrast by normalizing the grey-scale of each image using the dark background and a clear letter always present in these images as reference points for 0 (black) and 1 (white).
After that we discarded as much background as we can cutting the darkest part of the images.
Lastely we padded the images to make them squared and resized them to obtain a set of 256x256 preprocessed images.

QUI  METTEREI FOTO DEI VARI STEP DEL PREPROCESSING

## NN training
First of all we tried to find the best combinations of our network hyperparameters. Due to the high number of the hyperparameters combinations (METTERE QUNANTE SARANNO ALLA FINE) we preferred permorming bayesian search instead of grid search. This meant that we did not explore all the possible combination but, once the target number of trial is set, we followed a gradient-discendent method to reach a local minimum.
[![Documentation Status](https://readthedocs.org/projects/boneage-regression/badge/?version=latest)](https://boneage-regression.readthedocs.io/en/latest/?badge=latest)

![CircleCI](https://circleci.com/gh/MoseGuerini/boneage_regression/tree/main.svg?style=shield)
