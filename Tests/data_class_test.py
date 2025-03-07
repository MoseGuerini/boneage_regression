import unittest
import sys
import pathlib
from unittest.mock import patch
import pandas as pd

# Add the "boneage_regression" directory to sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "boneage_regression"))

from boneage_regression.data_class import DataLoader
from boneage_regression.data_class import is_numeric


class TestDataLoader(unittest.TestCase):

    def test_paths(self):
        """
        Test the behavior of the DataLoader when provided with valid and invalid paths.
        """

        # Define the current and parent directory paths
        current_dir = pathlib.Path(__file__).parent
        parent_dir = current_dir.parent

        # Define valid image and label paths
        valid_image_path = parent_dir / "Test_dataset" / "Test"
        valid_labels_path = parent_dir / "Test_dataset" / "test.csv"

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

        # Define the current and parent directory paths
        current_dir = pathlib.Path(__file__).parent
        parent_dir = current_dir.parent

        # Define valid image path and invalid labels path with missing column
        valid_image_path = parent_dir / "Test_dataset" / "Test"
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

        # Define the current and parent directory paths
        current_dir = pathlib.Path(__file__).parent
        parent_dir = current_dir.parent

        # Define valid image path, labels path, and valid target size
        valid_image_path = parent_dir / "Test_dataset" / "Test"
        valid_labels_path = parent_dir / "Test_dataset" / "test.csv"
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

        # Define current and parent directory paths
        current_dir = pathlib.Path(__file__).parent
        parent_dir = current_dir.parent

        # Define valid image and labels paths
        valid_image_path = parent_dir / "Test_dataset" / "Test"
        valid_labels_path = parent_dir / "Test_dataset" / "test.csv"

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


    """def test_load_labels_missing_columns(self):

        # Simulate a DataFrame with a missing 'id' column
        mock_df = pd.DataFrame({
            "id": [1, 2, 3],
            "boneage": [100, 120, 130],
            "male": [1, 0, 1]
        })

        with patch("boneage_regression.data_class.pd.read_csv", return_value=mock_df):

            # Mock pathlib.Path.iterdir() to prevent filesystem access
            with patch("pathlib.Path.iterdir", return_value=[]):

                # Use dummy paths (not actually accessed)
                dummy_image_path = "dummy/image/path"
                dummy_labels_path = "dummy/labels/path"

                # Ensure a ValueError is raised due to the missing 'id' column
                with self.assertRaises(ValueError):
                    DataLoader(
                        image_path=dummy_image_path,
                        labels_path=dummy_labels_path
                    ) """
    
    def test_preprocessing(self):

        # Obtain script folder and parent folder
        current_dir = pathlib.Path(__file__).parent
        parent_dir = current_dir.parent

        # Attach the correct paths
        valid_image_path = parent_dir / "Test_dataset" / "Test"
        valid_labels_path = parent_dir / "Test_dataset" / "test.csv"

        valid_preprocessing = False
        loader_false = DataLoader(
            image_path=valid_image_path,
            labels_path=valid_labels_path,
            preprocessing=valid_preprocessing
            )
        self.assertEqual(loader_false.preprocessing, valid_preprocessing)
        


if __name__ == '__main__':
    unittest.main()
