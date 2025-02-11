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
    """Check if the value is a float between 0 and 1"""
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value, input type: {type(value)}. Expected float.")
    
    if not (0 <= value <= 1):
        raise argparse.ArgumentTypeError(f"Value out of range: {value}. Must be between 0 and 1.")
    
    return value

def str2bool(value):
    """Convert a string to a boolean value."""
    if isinstance(value, bool):
        return value
    if value.lower() in ['true', 't', 'yes', 'y', '1']:
        return True
    elif value.lower() in ['false', 'f', 'no', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid value '{value}' for boolean argument. Expected values: 'True', 'False', 'Yes', 'No', '1', '0'."
        )
