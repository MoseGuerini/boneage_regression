import unittest
import sys
import pathlib
from unittest.mock import patch
import pandas as pd

# Add the "boneage_regression" directory to sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "boneage_regression"))

from data_class import DataLoader

# Obtain script folder and parent folder
current_dir = pathlib.Path(__file__).parent
parent_dir = current_dir.parent

# Attach the correct paths
valid_image_path = parent_dir / "Test_dataset" / "Test"
valid_labels_path = parent_dir / "Test_dataset" / "test.csv"


class TestDataLoader(unittest.TestCase):

    def test_paths(self):
        """
        Test the behavior of the DataLoader when provided with valid and invalid paths.
        """
        # Test: DataLoader should accept valid paths
        loader = DataLoader(
            image_path=valid_image_path,
            labels_path=valid_labels_path
        )
        self.assertEqual(loader.image_path, valid_image_path)
        self.assertEqual(loader.labels_path, valid_labels_path)

        # Test: DataLoader should raise an error for invalid paths
        invalid_labels_path = parent_dir / "unexisting_path"
        invalid_image_path = parent_dir / "unexisting_path"

        # Check that DataLoader raises FileNotFoundError when given an invalid label path
        with self.assertRaises(FileNotFoundError):
            DataLoader(
                image_path=valid_image_path,
                labels_path=invalid_labels_path
            )

        # Check that DataLoader raises FileNotFoundError when given an invalid image path
        with self.assertRaises(FileNotFoundError):
            DataLoader(
                image_path=invalid_image_path,
                labels_path=valid_labels_path
            )

    def test_load_labels_missing_columns(self):
        """
        Test that a CSV file with a missing column raises a ValueError.
        """
        invalid_labels_path = (
            parent_dir / "Test_dataset" / "training_missing_column.csv"
        )

        # Check that DataLoader raises a ValueError when the labels CSV has missing columns
        with self.assertRaises(ValueError):
            DataLoader(
                image_path=valid_image_path,
                labels_path=invalid_labels_path
            )

    def test_target_size(self):
        """
        Test that the target_size parameter is validated properly.
        """
        valid_target_size = (256, 256)

        # Test with valid target size
        loader = DataLoader(
            image_path=valid_image_path,
            labels_path=valid_labels_path,
            target_size=valid_target_size
        )
        self.assertEqual(loader._target_size, valid_target_size)

        # Define invalid target sizes and test each case
        invalid_target_sizes = [
            256,               # Not a tuple
            'abc',             # Not a tuple
            (0.1, 0.1),        # Values are not integers
            (128, 'abc'),      # Second value is not an integer
            (128, -128),       # Negative value
            (128, 64)          # Valid tuple, but check the use case
        ]

        for invalid_target_size in invalid_target_sizes:
            with self.assertRaises(ValueError):
                DataLoader(
                    image_path=valid_image_path,
                    labels_path=valid_labels_path,
                    target_size=invalid_target_size
                )

    def test_num_images(self):
        """
        Test that the num_images parameter is validated properly.
        """
        # Test with valid num_images
        valid_num_images = 10

        loader = DataLoader(
            image_path=valid_image_path,
            labels_path=valid_labels_path,
            num_images=valid_num_images
        )
        self.assertEqual(loader.num_images, valid_num_images)

        # Test with None (default behavior)
        valid_num_images = None
        loader = DataLoader(
            image_path=valid_image_path,
            labels_path=valid_labels_path,
            num_images=valid_num_images
        )
        self.assertEqual(loader.num_images, valid_num_images)

        # List of invalid num_images values for testing
        invalid_num_images = [
            (256, 1),    # Tuple instead of a single integer
            'abc',       # String instead of an integer
            0.1,         # Float instead of an integer
            -10          # Negative integer, should raise ValueError
        ]

        # Loop through each invalid value and assert that ValueError is raised
        for invalid_value in invalid_num_images:
            with self.assertRaises(ValueError):
                DataLoader(
                    image_path=valid_image_path,
                    labels_path=valid_labels_path,
                    num_images=invalid_value
                )

    def test_preprocessing(self):
        """Test preprocessing"""
        valid_preprocessing = False

        loader_false = DataLoader(
            image_path=valid_image_path,
            labels_path=valid_labels_path,
            preprocessing=valid_preprocessing
            )
        self.assertEqual(loader_false.preprocessing, valid_preprocessing)


if __name__ == '__main__':
    unittest.main()
