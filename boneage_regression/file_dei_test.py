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
    def test_is_numeric(self):
        """Verifico che la funzione accetti un numero intero"""
        valid_string = "123"
        self.assertTrue(is_numeric(valid_string))
        """Verifico che la funzione non accetti ciò che è diverso da un numero intero"""
        invalid_string = "abc"
        self.assertFalse(is_numeric(invalid_string))
    
    def test_paths(self):
        """Verifico che la funzione proceda se vengono inseriti un path per le immagini e uno per le labels validi"""
        valid_image_path = r'C:\Users\nicco\Desktop\Test_folder\test_image_path\valid_path'
        valid_labels_path = r'C:\Users\nicco\Desktop\Test_folder\test_label_path\valid_path\train.csv'
        loader = DataLoader(image_path=valid_image_path, labels_path=valid_labels_path)
        self.assertEqual(loader.image_path, pathlib.Path(valid_image_path))
        self.assertEqual(loader.labels_path, pathlib.Path(valid_labels_path))
        """Verifico che la funzione si arresti se vengono inseriti un path per le immagini e/o uno per le labels invalidi"""
        invalid_labels_path = r'C:\Users\nicco\Desktop\unexisting_path'
        with self.assertRaises(FileNotFoundError):  
            DataLoader(image_path=valid_image_path, labels_path=invalid_labels_path)
        invalid_image_path = r'C:\Users\nicco\Desktop\unexisting_path'
        # Il test ora si aspetta un FileNotFoundError
        with self.assertRaises(FileNotFoundError):  
            DataLoader(image_path=invalid_image_path, labels_path=valid_labels_path)    

    def test_target_size(self):
        """Verifico che la funzione accetti solo una coppia intera come target_size"""
        valid_image_path = r'C:\Users\nicco\Desktop\Test_folder\test_image_path\valid_path'
        valid_labels_path = r'C:\Users\nicco\Desktop\Test_folder\test_label_path\valid_path\train.csv'
        valid_target_size = (256,256)
        loader = DataLoader(image_path=valid_image_path, labels_path=valid_labels_path, target_size=valid_target_size)
        self.assertEqual(loader.target_size, valid_target_size)
        invalid_target_size = 256
        with self.assertRaises(ValueError):  
            DataLoader(image_path=valid_image_path, labels_path=valid_labels_path, target_size=invalid_target_size) 
        invalid_target_size = 'abc'
        with self.assertRaises(ValueError):  
            DataLoader(image_path=valid_image_path, labels_path=valid_labels_path, target_size=invalid_target_size) 
        invalid_target_size = (0.1,0.1)
        with self.assertRaises(ValueError):  
            DataLoader(image_path=valid_image_path, labels_path=valid_labels_path, target_size=invalid_target_size) 
        invalid_target_size = (128,'abc')
        with self.assertRaises(ValueError):  
            DataLoader(image_path=valid_image_path, labels_path=valid_labels_path, target_size=invalid_target_size)
        invalid_target_size = (128,-128)
        with self.assertRaises(ValueError):  
            DataLoader(image_path=valid_image_path, labels_path=valid_labels_path, target_size=invalid_target_size)
        invalid_target_size = (128,64)
        with self.assertRaises(ValueError):  
            DataLoader(image_path=valid_image_path, labels_path=valid_labels_path, target_size=invalid_target_size)
            
    def test_num_images(self) :
        """Verifico che la funzione accetti solo un numero intero come num_images"""
        valid_image_path = r'C:\Users\nicco\Desktop\Test_folder\test_image_path\valid_path'
        valid_labels_path = r'C:\Users\nicco\Desktop\Test_folder\test_label_path\valid_path\train.csv'
        valid_num_imges = 10
        loader = DataLoader(image_path=valid_image_path, labels_path=valid_labels_path, num_images=valid_num_imges)
        self.assertEqual(loader.num_images, valid_num_imges)
        valid_num_imges = None
        loader = DataLoader(image_path=valid_image_path, labels_path=valid_labels_path, num_images=valid_num_imges)
        self.assertEqual(loader.num_images, valid_num_imges)
        invalid_num_images = (256,1)
        with self.assertRaises(ValueError):  
            DataLoader(image_path=valid_image_path, labels_path=valid_labels_path, num_images=invalid_num_images) 
        invalid_num_images = 'abc'
        with self.assertRaises(ValueError):  
            DataLoader(image_path=valid_image_path, labels_path=valid_labels_path,num_images=invalid_num_images) 
        invalid_num_images = 0.1
        with self.assertRaises(ValueError):  
            DataLoader(image_path=valid_image_path, labels_path=valid_labels_path, num_images=invalid_num_images) 
        invalid_num_images = -10
        with self.assertRaises(ValueError):  
            DataLoader(image_path=valid_image_path, labels_path=valid_labels_path, num_images=invalid_num_images)     
            
    def test_preprocessing(self) :
        """Verifico che la funzione accetti un numero intero come preprocessing"""
        valid_image_path = r'C:\Users\nicco\Desktop\Test_folder\test_image_path\valid_path'
        valid_labels_path = r'C:\Users\nicco\Desktop\Test_folder\test_label_path\valid_path\train.csv'
        valid_preprocessing = True
        valid_num_workers = 8
        valid_target_size = (64,64)
        loader = DataLoader(image_path=valid_image_path, labels_path=valid_labels_path, preprocessing=valid_preprocessing, num_workers=valid_num_workers, target_size=valid_target_size)
        self.assertEqual(loader.preprocessing, valid_preprocessing)
        valid_preprocessing = False
        loader = DataLoader(image_path=valid_image_path, labels_path=valid_labels_path, preprocessing=valid_preprocessing, num_workers=valid_num_workers, target_size=valid_target_size)
        self.assertEqual(loader.preprocessing, valid_preprocessing)
        invalid_preprocessing = (256,1)
        with self.assertRaises(ValueError):  
            DataLoader(image_path=valid_image_path, labels_path=valid_labels_path, preprocessing=invalid_preprocessing, num_workers=valid_num_workers, target_size=valid_target_size)
        invalid_preprocessing = 'abc'
        with self.assertRaises(ValueError):  
            DataLoader(image_path=valid_image_path, labels_path=valid_labels_path, preprocessing=invalid_preprocessing, num_workers=valid_num_workers, target_size=valid_target_size)
        invalid_preprocessing = 0.1
        with self.assertRaises(ValueError):  
            DataLoader(image_path=valid_image_path, labels_path=valid_labels_path, preprocessing=invalid_preprocessing, num_workers=valid_num_workers, target_size=valid_target_size)
        invalid_preprocessing = -10
        with self.assertRaises(ValueError):  
            DataLoader(image_path=valid_image_path, labels_path=valid_labels_path, preprocessing=invalid_preprocessing, num_workers=valid_num_workers, target_size=valid_target_size)      
        invalid_preprocessing = 1
        with self.assertRaises(ValueError):  
            DataLoader(image_path=valid_image_path, labels_path=valid_labels_path, preprocessing=invalid_preprocessing, num_workers=valid_num_workers, target_size=valid_target_size)
      
if __name__ == '__main__':
    unittest.main()