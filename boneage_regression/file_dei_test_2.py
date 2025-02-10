import unittest
import pathlib
from unittest.mock import patch, mock_open
import numpy as np
import pandas as pd
from class_to_read_data import DataLoader  # Sostituisci con il tuo modulo

class DataLoaderTests(unittest.TestCase):

    @patch('matplotlib.pyplot.imread')
    @patch('tensorflow.image.resize')
    def test_load_images_valid(self, mock_resize, mock_imread):
        """Test load_images con immagini valide."""
        loader = DataLoader(image_path=r'C:\Users\nicco\Desktop\Test_folder\test_image_path\valid_path', labels_path=None, preprocessing=True)
        mock_resize.return_value = np.zeros((128, 128, 3))  # Immagine ridimensionata
        mock_imread.return_value = np.zeros((256, 256))  # Immagine in bianco e nero

        # Mock dei file immagine
        loader.image_path = pathlib.Path(r'C:\Users\nicco\Desktop\Test_folder\test_image_path\valid_path')
        loader.num_images = 2  # Limita a 2 immagini per il test
        loader.load_images()

        # Verifica che il numero di immagini caricate sia corretto
        self.assertEqual(len(loader.filtered_images_rgb), 2)

    def test_load_labels_valid(self):
        """Test load_labels con un CSV valido."""
        loader = DataLoader(image_path=r"C:\Users\nicco\Desktop\Test_folder\test_image_path\valid_path", labels_path=r"C:\Users\nicco\Desktop\Test_folder\test_label_path\valid_path\train.csv")

        # Mock del DataFrame
        mock_df = pd.DataFrame({
            'id': [1, 2, 3],
            'boneage': [12, 15, 10],
            'male': [1, 0, 1]
        })
        with patch('pandas.read_csv', return_value=mock_df):
            labels, missing_ids = loader.load_labels([1, 2])

        # Verifica che le etichette siano caricate correttamente
        self.assertEqual(labels.shape, (2, 2))  # Due etichette per due immagini
        self.assertEqual(missing_ids, [])

    def test_load_labels_invalid_path(self):
        """Test load_labels con un percorso non valido."""
        loader = DataLoader(image_path=r'C:\Users\nicco\Desktop\Test_folder\test_image_path\valid_path', labels_path="C:/Users/nicco/Desktop/invalid_labels.csv")

        with self.assertRaises(FileNotFoundError):
            loader.load_labels([1, 2])

    def test_load_labels_missing_columns(self):
        """Test load_labels con colonne mancanti nel CSV."""
        loader = DataLoader(image_path=r'C:\Users\nicco\Desktop\Test_folder\test_image_path\valid_path', labels_path=r"C:\Users\nicco\Desktop\Test_folder\test_label_path\train_missing_gender")

        # Mock di un CSV senza la colonna 'male'
        mock_df = pd.DataFrame({
            'id': [1, 2],
            'boneage': [12, 15]
        })
        with patch('pandas.read_csv', return_value=mock_df):
            with self.assertRaises(ValueError):
                loader.load_labels([1, 2]) 
        
if __name__ == "__main__":
    unittest.main()
