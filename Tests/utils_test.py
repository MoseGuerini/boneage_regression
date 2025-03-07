import unittest
import sys
import pathlib
import argparse
import tempfile
from tensorflow import keras

# Add the "boneage_regression" directory to sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "boneage_regression"))

from boneage_regression.utils import (
    is_integer,
    check_rate,
    get_last_conv_layer_name,
    str2bool,
    check_folder
)


class UtilsTest(unittest.TestCase):

    def test_is_numeric(self):
        """Test cases for the is_numeric function."""

        # Valid integer string should return True
        self.assertTrue(is_integer("123"))

        # Invalid inputs should return False
        self.assertFalse(is_integer("abc"))    # Alphabetic string
        self.assertFalse(is_integer(0.1))      # Float
        self.assertFalse(is_integer([1, 1]))   # List of integers

    def test_check_rate(self):
        """Test cases for the check_rate function."""

        # Valid float and float-like string should return the correct value
        self.assertEqual(check_rate("0.1"), 0.1)  # Float as string
        self.assertEqual(check_rate(0.1), 0.1)    # Actual float

        # Invalid inputs should raise argparse.ArgumentTypeError
        with self.assertRaises(argparse.ArgumentTypeError):
            check_rate("abc")    # Non-numeric string

        with self.assertRaises(argparse.ArgumentTypeError):
            check_rate([1, 2])   # List input

        with self.assertRaises(argparse.ArgumentTypeError):
            check_rate(8)        # Integer (out of valid float range)

    def test_str2bool(self):
        """Test cases for the str2bool function."""

        # Valid truthy values should return True
        for valid in ["true", "t", "yes", "y", "1", 1]:
            self.assertEqual(str2bool(valid), True)

        # Valid falsy values should return False
        for valid in ["false", "f", "no", "n", "0", 0]:
            self.assertEqual(str2bool(valid), False)

        # Invalid values should raise an ArgumentTypeError
        for invalid in ["maybe", "2", "yesno", "tru", "fal", "-1", "random"]:
            with self.assertRaises(argparse.ArgumentTypeError):
                str2bool(invalid)

        # Non-boolean values (including floats and lists) should raise an error
        for invalid in ["0.1", [1, 1], 0.1, 0.0]:
            with self.assertRaises(argparse.ArgumentTypeError):
                str2bool(invalid)

    def test_check_folder(self):
        """Test cases for the check_folder function."""
        # Test error raised when a file path is provided instead of a directory
        with tempfile.NamedTemporaryFile() as tmp_file:
            with self.assertRaises(argparse.ArgumentTypeError):
                check_folder(tmp_file.name)

        # Test error raised for a non-existent directory path
        with self.assertRaises(argparse.ArgumentTypeError):
            check_folder("/non/existent/path")

        # Test with an existing directory path
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)

            # Create required folders and CSV files
            (tmp_path / 'Training').mkdir()
            (tmp_path / 'Test').mkdir()
            (tmp_path / 'training.csv').touch()
            (tmp_path / 'test.csv').touch()

            # Call check_folder and validate the result
            result = check_folder(tmp_dir)
            self.assertIsInstance(result, pathlib.Path)
            self.assertEqual(result, tmp_path)

        # Check that missing folders/files raise an error
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)

            # Create only part of the required structure
            (tmp_path / 'Training').mkdir()  # Missing 'Test'
            (tmp_path / 'test.csv').touch()  # Missing 'training.csv'

            with self.assertRaises(argparse.ArgumentTypeError):
                check_folder(tmp_dir)

    def test_get_last_conv_layer_name(self):
        """Test cases for the get_last_conv_layer_name function."""

        # Test with a model having multiple Conv2D layers
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation="relu", name="conv1"),
            keras.layers.Conv2D(64, (3, 3), activation="relu", name="conv2"),
        ])
        self.assertEqual(get_last_conv_layer_name(model), "conv2")

        # Test with a model having only one Conv2D layer
        model = keras.Sequential([
            keras.layers.Conv2D(16, (3, 3), activation="relu", name="conv_single"),
            keras.layers.Dense(64, activation="relu", name="dense")
        ])
        self.assertEqual(get_last_conv_layer_name(model), "conv_single")

        # Test with a model having Conv2D layers along with other types of layers
        model = keras.Sequential([
            keras.layers.Conv2D(16, (3, 3), activation="relu", name="conv1"),
            keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),
            keras.layers.Conv2D(32, (3, 3), activation="relu", name="conv2"),
            keras.layers.Flatten(name="flatten"),
            keras.layers.Dense(128, activation="relu", name="dense1")
        ])
        self.assertEqual(get_last_conv_layer_name(model), "conv2")

        # Test with a model that contains no Conv2D layers, expecting a ValueError
        model = keras.Sequential([
            keras.layers.Dense(128, activation="relu", name="dense1"),
            keras.layers.Dense(64, activation="relu", name="dense2")
        ])
        with self.assertRaises(ValueError):
            get_last_conv_layer_name(model)


if __name__ == '__main__':
    unittest.main()
