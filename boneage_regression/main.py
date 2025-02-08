"""Main"""
import numpy as np
import argparse
import os
import time
import keras
import shutup
import warnings 
#from utils import wave_dict, hyperp_dict, str2bool, rate, delete_directory
from hyperparameters import hyperp_space_size
from classes import  Model

warnings.filterwarnings('ignore')
shutup.please()
if __name__=='__main__':
    start = time.time()
    os.chdir('..')
    parser = argparse.ArgumentParser(
        description="Bone Age Regressor"
    )
    
    parser.add_argument(
        "-fast",
        "--fast_execution",
        metavar="",
        type=bool,      #aggiungere controllo ad esempio stringa to bool per poter immettere stringhe
        help="If True avoid hyperparameters search and use the pre-saved hyperpar. Default: False",
        default=False,
    )
    
    parser.add_argument(
        "-cl",
        "--conv_layers",
        metavar="",
        nargs='+',
        type=int,
        help="List of values for the hypermodel's number of conv2d layers",
        default=[3,4,5],
    )
    
    
    parser.add_argument(
        "-cf",
        "--conv_filters",
        metavar="",
        nargs='+',
        type=int,
        help="List of values for the hypermodel's first conv2d number of filters",
        default=[8,16,32],
    )
    
    parser.add_argument(
        "-du",
        "--dense_units",
        metavar="",
        nargs='+',
        type=float,     
        help="List of values for the hypermodel's dense units",
        default=[64, 128, 256],
    )

    parser.add_argument(
        "-dd",
        "--dense_depth",
        metavar="",
        nargs='+',
        type=float,     
        help="List of values for the hypermodel's depth of final dense layers",
        default=[1, 2, 3],
    )
    parser.add_argument(
        "-dp",
        "--dropout_rate",
        metavar="",
        nargs='+',
        type=float,     #aggiungere controllo in modo che sia solo tra 0 e 1
        help="List of values for the hypermodel's dropout rate",
        default=[0.1, 0.2, 0.3],
    )

    parser.add_argument(
        "-sf",
        "--searching_fraction",
        metavar="",
        type=float,     #aggiungere controllo in modo che sia solo tra 0 e 1
        help="Fraction of the hyperparamiters space explored during hypermodel search. Default: 0.25",
        default=0.25,
    )

    args = parser.parse_args()

    # Dataset part

    #2. set chosen hyperparameters and get number of trials
    hyperp_dict=hyperp_dict(args.conv_layers, args.conv_filters, args.dropout_rate, args.dense_units, args.dense_depth)
    space_size = hyperp_space_size()
    max_trials = np.rint(args.searching_fraction*space_size)

    #3. create and train the model
    model = Model(data=data, fast=args.fast_execution, max_trials=max_trials)
    model.train()

    #4. check what the most reliable model has learnt using gradCAM
    best_model = keras.models.load_model(model.selected_model)
    num_images = args.gradcam
    if num_images > 25:
        print('Showing 25 images using gradCAM')
        num_images = 25
    rand_images, _ = data.get_random_images(size=num_images, classes=[1])
    preds = best_model.predict(rand_images)
    get_gcam_images(rand_images, best_model)
    gCAM_show(preds=preds)