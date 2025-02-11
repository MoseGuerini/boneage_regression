import argparse 

from hyperparameters import set_hyperp

def hyperp_dict(conv_layers, conv_filters, dense_units, dense_depth, dropout_rate):
    """Creates dictionary containing user-selected hps keeping only unique values in each list 
    and sets it to be a global variable with set_hyperp"""
    hyperp_dict = {
            'conv_layers' : list(set(conv_layers)),
            'conv_filters': list(set(conv_filters)),
            'dense_units' : list(set(dense_units)),
            'dense_depth' : list(set(dense_depth)),
            'dropout_rate': list(set(dropout_rate))

    }
    set_hyperp(hyperp_dict)
    return hyperp_dict

def check_rate(value):
    if value < 0 or value > 1:
        raise argparse.ArgumentTypeError(f'Value must be between 0 and 1, input value:{value}')
    
