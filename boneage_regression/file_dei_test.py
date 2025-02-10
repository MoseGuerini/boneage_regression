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
        valid_string = "123"
        self.assertTrue(is_numeric(valid_string))
        
    def test_is_numeric_invalid(self):
        invalid_string = "abc"
        self.assertFalse(is_numeric(invalid_string))
    
    def test_image_path_validation_valid(self):
        """Test che il percorso dell'immagine venga accettato se valido."""
        
        # Crea una cartella temporanea che esiste
        with tempfile.TemporaryDirectory() as valid_temp_dir:
            valid_path = valid_temp_dir  # Il percorso esiste

            # Test per percorso valido
            loader = DataLoader(image_path=valid_path, labels_path=None)
            self.assertEqual(loader.image_path, pathlib.Path(valid_path))  # Verifica che il percorso sia impostato correttamente

    def test_image_path_validation_invalid(self):
        """Test che venga sollevata un'eccezione se il percorso dell'immagine non esiste."""
        
        # Percorso che non esiste
        invalid_path = r'C:\path\that\does\not\exist'  # Un percorso che sicuramente non esiste

        # Test per percorso non valido
        with self.assertRaises(FileNotFoundError):
            DataLoader(image_path=invalid_path, labels_path=None)  # Verifica che venga sollevata l'eccezione
            
    def test_init_labels_path_none(self):
        """Test che labels_path sia None se non passato."""
        image_path = r'C:\Users\nicco\Desktop\Test_folder\test_image_path\valid_path'
        loader = DataLoader(image_path=image_path, labels_path=None)
        self.assertIsNone(loader.labels_path)  # Verifica che labels_path sia None

    def test_init_num_images(self):
        """Test che num_images venga correttamente passato."""
        image_path = r'C:\Users\nicco\Desktop\Test_folder\test_image_path\valid_path'
        num_images = 100
        loader = DataLoader(image_path=image_path, labels_path=None, num_images=num_images)
        self.assertEqual(loader.num_images, num_images)  # Verifica che num_images sia uguale a 100

    def test_init_preprocessing(self):
        """Test che preprocessing venga correttamente passato."""
        image_path = r'C:\Users\nicco\Desktop\Test_folder\test_image_path\valid_path'
        loader = DataLoader(image_path=image_path, labels_path=None, preprocessing=True)
        self.assertTrue(loader.preprocessing)  # Verifica che preprocessing sia True

    def test_init_target_size(self):
        """Test che target_size venga correttamente passato."""
        image_path = r'C:\Users\nicco\Desktop\Test_folder\test_image_path\valid_path'
        target_size = (256, 256)
        loader = DataLoader(image_path=image_path, labels_path=None, target_size=target_size)
        self.assertEqual(loader.target_size, target_size)  # Verifica che target_size sia (256, 256)

    def test_init_num_workers(self):
        """Test che num_workers venga correttamente passato."""
        image_path = r'C:\Users\nicco\Desktop\Test_folder\test_image_path\valid_path'
        num_workers = 8
        loader = DataLoader(image_path=image_path, labels_path=None, num_workers=num_workers)
        self.assertEqual(loader.num_workers, num_workers)  # Verifica che num_workers sia uguale a 8
        
    def test_image_path_setter_valid(self):
        """Test setter di image_path con un percorso valido."""
        loader = DataLoader(image_path=r'C:\Users\nicco\Desktop\Test_folder\test_image_path\valid_path', labels_path=None)
        # Assicurati che il percorso venga correttamente impostato
        self.assertEqual(loader.image_path, pathlib.Path(r'C:\Users\nicco\Desktop\Test_folder\test_image_path\valid_path'))

    def test_target_size_setter_valid(self):
        """Test setter di target_size con una tupla valida."""
        loader = DataLoader(image_path=r'C:\Users\nicco\Desktop\Test_folder\test_image_path\valid_path', labels_path=None)
        loader.target_size = (256, 256)
        self.assertEqual(loader.target_size, (256, 256))  # Verifica che target_size venga assegnato correttamente

    def test_target_size_setter_invalid(self):
        """Test setter di target_size con una tupla non valida."""
        loader = DataLoader(image_path=r'C:\Users\nicco\Desktop\Test_folder\test_image_path\valid_path', labels_path=None)
        with self.assertRaises(ValueError):
            loader.target_size = (256, -256)  # Tupla con valore negativo

        with self.assertRaises(ValueError):
            loader.target_size = (256,)  # Tupla con un solo valore

        with self.assertRaises(ValueError):
            loader.target_size = (256, "abc")  # Tupla con un valore non intero

    def test_num_images_setter_valid(self):
        """Test setter di num_images con un valore valido."""
        loader = DataLoader(image_path=r'C:\Users\nicco\Desktop\Test_folder\test_image_path\valid_path', labels_path=None)
        loader.num_images = 100
        self.assertEqual(loader.num_images, 100)  # Verifica che num_images venga assegnato correttamente

    def test_num_images_setter_invalid(self):
        """Test setter di num_images con un valore non valido."""
        loader = DataLoader(image_path=r'C:\Users\nicco\Desktop\Test_folder\test_image_path\valid_path', labels_path=None)
        with self.assertRaises(ValueError):
            loader.num_images = -10  # Valore negativo

        with self.assertRaises(ValueError):
            loader.num_images = "abc"  # Tipo di dato errato

    def test_preprocessing_setter_valid(self):
        """Test setter di preprocessing con un valore valido."""
        loader = DataLoader(image_path=r'C:\Users\nicco\Desktop\Test_folder\test_image_path\valid_path', labels_path=None)
        loader.preprocessing = True
        self.assertTrue(loader.preprocessing)  # Verifica che preprocessing venga impostato correttamente

    def test_preprocessing_setter_invalid(self):
        """Test setter di preprocessing con un valore non valido."""
        loader = DataLoader(image_path=r'C:\Users\nicco\Desktop\Test_folder\test_image_path\valid_path', labels_path=None)
        with self.assertRaises(ValueError):
            loader.preprocessing = "True"  # Tipo di dato errato




            
if __name__ == '__main__':
    unittest.main()