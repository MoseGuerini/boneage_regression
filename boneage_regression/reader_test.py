import unittest
import pathlib
import numpy as np
import pandas as pd
import matlab.engine
from unittest.mock import patch, MagicMock, mock_open
from data_class import DataLoader, is_numeric  # Sostituisci con il modulo corretto
import matplotlib.pyplot as plt
from unittest.mock import patch

class TestDataLoader(unittest.TestCase):
    
    # Test for is_numeric function
    def test_is_numeric_valid(self):
        valid_string = "123"
        self.assertTrue(is_numeric(valid_string))
        
    def test_is_numeric_invalid(self):
        invalid_string = "abc"
        self.assertFalse(is_numeric(invalid_string))

    # Test for initialization
    @patch("pathlib.Path.exists", return_value=True)
    def test_initialization(self, mock_exists):
        image_path = "/mock/path/images"
        labels_path = "/mock/path/labels.csv"
        target_size = (128, 128)
        num_images = 100
        preprocessing = True
        num_workers = 4
        
        loader = DataLoader(image_path, labels_path, target_size, num_images, preprocessing, num_workers)
        
        # Test if attributes are set correctly
        self.assertEqual(loader.image_path, pathlib.Path(image_path))
        self.assertEqual(loader.labels_path, pathlib.Path(labels_path))
        self.assertEqual(loader.target_size, target_size)
        self.assertEqual(loader.num_images, num_images)
        self.assertEqual(loader.preprocessing, preprocessing)
        self.assertEqual(loader.num_workers, num_workers)

    # Test for load_images function
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.iterdir")
    @patch("matplotlib.pyplot.imread")
    def test_load_images_success(self, mock_imread, mock_iterdir, mock_exists):
        mock_iterdir.return_value = [pathlib.Path(f"mock_image_{i}.jpg") for i in range(5)]
        mock_imread.return_value = np.zeros((128, 128, 3))  # Mock image of size 128x128
        
        loader = DataLoader('/mock/path/images', '/mock/path/labels.csv', num_images=5)
        
        # Test loading images
        images, ids, labels = loader.load_images()
        
        # Test if 5 images are loaded
        self.assertEqual(images.shape, (5, 128, 128, 3))
        self.assertEqual(len(ids), 5)
        
    @patch("pathlib.Path.exists", return_value=False)
    def test_load_images_path_not_found(self, mock_exists):
        loader = DataLoader('/invalid/path/images', '/mock/path/labels.csv')
        with self.assertRaises(FileNotFoundError):
            loader.load_images()

    # Test for preprocess_images function
    @patch('matlab.engine.start_matlab')
    def test_preprocess_images(self, mock_start_matlab):
        mock_eng = MagicMock()
        mock_start_matlab.return_value = mock_eng
        
        loader = DataLoader('/mock/path/images', '/mock/path/labels.csv', preprocessing=True)
        
        # Preprocessing should be called
        loader.preprocess_images()
        
        # Test if preprocessing function was called
        mock_eng.preprocessing.assert_called_once_with(
            '/mock/path/images', '/mock/path/processed_images', 12, 128, nargout=0
        )
        
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pandas.read_csv')  # Mock del CSV
    @patch('builtins.open', new_callable=mock_open)  # Mock dell'apertura file
    @patch('matlab.engine.start_matlab')  # Se serve il MATLAB Engine
    def test_preprocessing_flag(self, mock_start_matlab, mock_read_csv, mock_open_file, mock_exists):
        mock_eng = MagicMock()
        mock_start_matlab.return_value = mock_eng
        mock_read_csv.return_value = pd.DataFrame()
        
        loader = DataLoader('/mock/path/images', '/mock/path/labels.csv', preprocessing=True)

        # Preprocessing è attivo
        self.assertTrue(loader.preprocessing)

        # Esegui il preprocessing
        loader.preprocess_images()

        # Controlla che il preprocessing sia stato disattivato dopo l'esecuzione
        self.assertFalse(loader.preprocessing)

    # Test for load_labels function
    @patch('pandas.read_csv')
    def test_load_labels_success(self, mock_read_csv):
        mock_df = pd.DataFrame({'id': [1, 2, 3], 'boneage': [10, 15, 20], 'male': [1, 0, 1]})
        mock_read_csv.return_value = mock_df
        
        loader = DataLoader('/mock/path/images', '/mock/path/labels.csv')
        
        image_ids = [1, 2]
        labels, missing_ids = loader.load_labels(image_ids)
        
        # Test if correct labels are loaded
        self.assertEqual(labels.shape, (2, 2))  # 2 images with 2 labels (boneage, gender)
        self.assertEqual(missing_ids, [])
        
    @patch('pandas.read_csv')
    def test_load_labels_missing_column(self, mock_read_csv):
        mock_df = pd.DataFrame({'id': [1, 2], 'boneage': [10, 15]})  # Missing 'male' column
        mock_read_csv.return_value = mock_df
        
        loader = DataLoader('/mock/path/images', '/mock/path/labels.csv')
        
        image_ids = [1, 2]
        
        with self.assertRaises(ValueError):
            loader.load_labels(image_ids)

if __name__ == '__main__':
    unittest.main()
