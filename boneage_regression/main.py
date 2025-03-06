"""Main"""
import sys
import argparse
import pathlib

import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from tensorflow.keras import mixed_precision

from hyperparameters import hyperp_space_size
from model_class import CnnModel
from data_class import DataLoader
from utils import hyperp_dict, check_rate, check_folder, str2bool

# Setting logger configuration
logger.remove()
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO"
)

# Setting default data folder
data_dir = (
    pathlib.Path(__file__).resolve().parent.parent / 'Preprocessed_images'
)

if __name__ == '__main__':

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    parser = argparse.ArgumentParser(
        description=(
            """
            This script performs bone age prediction using a machine learning
            regression model. It accepts input parameters for model
            configuration and the dataset folder path.

            If you pass a dataset folder remember that it must contain:

            1) Two separate folders for training and test images named
                'Training' and 'Test'.
            2) Two .csv files with the corresponding labels named
                'training.csv' and 'test.csv'.

            Each CSV file must contain three columns named 'ID', 'boneage',
            'male'.
            """
        )
    )

    parser.add_argument(
        "-fp",
        "--folder_path",
        metavar="",
        type=check_folder,
        help="Path to the directory containing training "
        "and test images as well as csv files with the labels. "
        "Default: Preprocessed_images",
        default=data_dir,
    )

    parser.add_argument(
        "-p",
        "--preprocessing",
        metavar="",
        type=str2bool,
        help="If False avoid image preprocessing. "
        "Default: False",
        default=False,
    )

    parser.add_argument(
        "-o",
        "--overwrite",
        metavar="",
        type=str2bool,
        help="If False avoid hyperparameters search "
        "and use the pre-saved hyperpar. Default: False",
        default=False,
    )

    parser.add_argument(
        "-cl",
        "--conv_layers",
        metavar="",
        nargs='+',
        type=int,
        help="List of values for the hypermodel number of conv2d layers"
        " Default [3, 4, 5]",
        default=[3, 4, 5],
    )

    parser.add_argument(
        "-cf",
        "--conv_filters",
        metavar="",
        nargs='+',
        type=int,
        help="List of values for the hypermodel "
        "first conv2d number of filters. Default [8, 16, 32]",
        default=[8, 16, 32],
    )

    parser.add_argument(
        "-dd",
        "--dense_depth",
        metavar="",
        nargs='+',
        type=int,
        help="List of values for the hypermodel depth of final dense layers"
        " Default [1, 2, 3]",
        default=[1, 2, 3],
    )
    parser.add_argument(
        "-dr",
        "--dropout_rate",
        metavar="",
        nargs='+',
        type=check_rate,
        help="List of values for the dropout rate of the final dense layers"
        " Default [0.1, 0.2, 0.3]",
        default=[0.1, 0.2, 0.3],
    )

    parser.add_argument(
        "-sf",
        "--searching_fraction",
        metavar="",
        type=check_rate,
        help="Fraction of the hyperparameters space explored "
        "during hypermodel search. Default: 0.25",
        default=0.25,
    )

    args = parser.parse_args()

    # 1. Dataset loading and optional preprocessing
    train_data = args.folder_path / 'Training'
    train_csv = args.folder_path / 'training.csv'
    test_data = args.folder_path / 'Test'
    test_csv = args.folder_path / 'test.csv'

    data_train = DataLoader(
        train_data,
        train_csv,
        preprocessing=args.preprocessing
        )

    data_test = DataLoader(
        test_data,
        test_csv,
        preprocessing=args.preprocessing
        )

    # 2. Set chosen hyperparameters and get number of trials for tuning
    hyperp_dict = hyperp_dict(
        args.conv_layers,
        args.conv_filters,
        args.dense_depth,
        args.dropout_rate
        )

    space_size = hyperp_space_size(hyperp_dict)

    max_trials = np.rint(args.searching_fraction*space_size)

    # 3. Create and train the model
    model = CnnModel(
        data_train=data_train,
        data_test=data_test,
        overwrite=args.overwrite,
        max_trials=max_trials
        )

    model.train()

    plt.show()
