"""Main"""
import numpy as np
import argparse
import pathlib
import matplotlib.pyplot as plt

#from utils import wave_dict, hyperp_dict, str2bool, rate, delete_directory
from hyperparameters import hyperp_space_size
from classes import  CNN_Model
from class_to_read_data import DataLoader

from utils import hyperp_dict


if __name__=='__main__':

    parser = argparse.ArgumentParser(
        description="Bone Age Regressor"
    )
    
    parser.add_argument(
        "-o",
        "--overwrite",
        metavar="",
        type=bool,      #aggiungere controllo ad esempio stringa to bool per poter immettere stringhe
        help="If False avoid hyperparameters search and use the pre-saved hyperpar. Default: False",
        default=True,
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

    #1. Dataset part
    test_data_dir = pathlib.Path(__file__).resolve().parent.parent / 'Test_dataset'
    train_data = test_data_dir / 'Training'
    train_csv = test_data_dir / 'training.csv'
    test_data = test_data_dir / 'Validation'
    test_csv = test_data_dir / 'Validation_dataset.csv'

    data_train = DataLoader(train_data, train_csv)
    data_test = DataLoader(test_data, test_csv)

    #2. set chosen hyperparameters and get number of trials
    hyperp_dict=hyperp_dict(args.conv_layers, args.conv_filters, args.dense_units, args.dense_depth, args.dropout_rate)
    space_size = hyperp_space_size()
    max_trials = 3
    #max_trials = np.rint(args.searching_fraction*space_size)

    #3. create and train the model
    model = CNN_Model(data_train=data_train, data_test=data_test, overwrite=args.overwrite, max_trials=max_trials)
    model.train()
    plt.show()
