import unittest
import pathlib
import numpy as np
import pandas as pd
import matlab.engine
from unittest.mock import patch, MagicMock, mock_open
from class_to_read_data import DataLoader, is_numeric  # Sostituisci con il modulo corretto
import matplotlib.pyplot as plt
from unittest.mock import patch
import tempfile

class TestDataLoader(unittest.TestCase):
    
    # Test for is_numeric function
    def test_is_numeric_valid(self):
        """Verifico che la funzione accetti un numero intero"""
        valid_string = "123"
        self.assertTrue(is_numeric(valid_string))
        
    def test_is_numeric_invalid(self):
        """Verifico che la funzione non accetti ciò che è diverso da un numero intero"""
        invalid_string = "abc"
        self.assertFalse(is_numeric(invalid_string))
    
    def test_image_path_validation_valid(self):
        """Verifico che la funzione proceda se vengono inseriti un path per le immagini e uno per le labels validi"""
        valid_image_path = r'C:\Users\nicco\Desktop\Test_folder\test_image_path\valid_path'
        valid_labels_path = r'C:\Users\nicco\Desktop\Test_folder\test_label_path\valid_path\train.csv'
        loader = DataLoader(image_path=valid_image_path, labels_path=valid_labels_path)
        self.assertEqual(loader.image_path, pathlib.Path(valid_image_path))
        self.assertEqual(loader.labels_path, pathlib.Path(valid_labels_path))
        
    def test_image_path_validation_invalid(self):
        """Verifico che la funzione si arresti se vengono inseriti un path per le immagini e\o uno per le labels invalidi"""
        valid_image_path = r'C:\Users\nicco\Desktop\unexisting_path'
        valid_labels_path = r'C:\Users\nicco\Desktop\unexisting_path'
        loader = DataLoader(image_path=valid_image_path, labels_path=valid_labels_path)
        self.assertEqual(loader.image_path, pathlib.Path(valid_image_path))
        self.assertEqual(loader.labels_path, pathlib.Path(valid_labels_path))






            
if __name__ == '__main__':
    unittest.main()