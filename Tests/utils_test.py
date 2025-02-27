import unittest
import sys
import pathlib
import argparse
from tensorflow import keras

# Aggiungi "boneage_regression" a sys.path
sys.path.insert(
    0,
    str(pathlib.Path(__file__).resolve().parent.parent / "boneage_regression")
)

# Import delle classi e funzioni
from utils import is_numeric, check_rate, get_last_conv_layer_name, str2bool


class UtilsTest(unittest.TestCase):

    def test_is_numeric(self):
        """An integer must be accepted"""
        valid_string = "123"
        self.assertTrue(is_numeric(valid_string))
        """Everything different from an integer must not be accepted"""
        invalid_string = "abc"
        self.assertFalse(is_numeric(invalid_string))
        invalid_string = 0.1
        self.assertFalse(is_numeric(invalid_string))
        invalid_string = [1, 1]
        self.assertFalse(is_numeric(invalid_string))

    def test_check_rate(self):
        """A float and a string must be accepted"""
        valid_string = "0.1"
        self.assertEqual(check_rate(valid_string), 0.1)
        valid_string = 0.1
        self.assertEqual(check_rate(valid_string), 0.1)

        """Everything different from a valid float must not be accepted"""
        invalid_string = "abc"
        with self.assertRaises(argparse.ArgumentTypeError):
            check_rate(invalid_string)
        invalid_string = [1, 2]
        with self.assertRaises(argparse.ArgumentTypeError):
            check_rate(invalid_string)
        invalid_string = 8
        with self.assertRaises(argparse.ArgumentTypeError):
            check_rate(invalid_string)

        """Test string representation of float"""
        valid_float_string = "0.9"
        self.assertEqual(check_rate(valid_float_string), 0.9)

    def test_str2bool(self):
        """A float and a string must be accepted"""
        for valid in ["true", "t", "yes", "y", "1", 1]:
            self.assertEqual(str2bool(valid), True)
        for valid in ["false", "f", "no", "n", "0", 0]:
            self.assertEqual(str2bool(valid), False)

        for invalid in ["maybe", "2", "yesno", "tru", "fal", "-1", "random"]:
            with self.assertRaises(argparse.ArgumentTypeError):
                str2bool(invalid)
        for invalid in ["0.1", [1, 1], 0.1, 0.0]:
            with self.assertRaises(argparse.ArgumentTypeError):
                str2bool(invalid)

    def test_get_last_conv_layer_name(self):
        """Should return the last Conv2D layer name in a model with"""
        """multiple Conv2D layers"""
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation="relu", name="conv1"),
            keras.layers.Conv2D(64, (3, 3), activation="relu", name="conv2"),
            keras.layers.Dense(128, activation="relu", name="dense")
        ])
        self.assertEqual(get_last_conv_layer_name(model), "conv2")

        """Should return the Conv2D layer name when there is only one"""
        model = keras.Sequential([
            keras.layers.Conv2D(
                16,
                (3, 3),
                activation="relu",
                name="conv_single"
                ),
            keras.layers.Dense(
                64,
                activation="relu",
                name="dense"
                )
        ])
        self.assertEqual(get_last_conv_layer_name(model), "conv_single")

        """Should correctly find the last Conv2D layer in a mixed model"""
        model = keras.Sequential([
            keras.layers.Conv2D(16, (3, 3), activation="relu", name="conv1"),
            keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),
            keras.layers.Conv2D(32, (3, 3), activation="relu", name="conv2"),
            keras.layers.Flatten(name="flatten"),
            keras.layers.Dense(128, activation="relu", name="dense1")
        ])
        self.assertEqual(get_last_conv_layer_name(model), "conv2")

        """Should raise ValueError if no Conv2D layer exists"""
        model = keras.Sequential([
            keras.layers.Dense(128, activation="relu", name="dense1"),
            keras.layers.Dense(64, activation="relu", name="dense2")
        ])
        with self.assertRaises(ValueError):
            get_last_conv_layer_name(model)


if __name__ == '__main__':
    unittest.main()
