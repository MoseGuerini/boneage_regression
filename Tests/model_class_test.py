import sys
import pathlib
import unittest
import numpy as np

# Add the "boneage_regression" directory to sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "boneage_regression"))

from model_class import CnnModel


class MockData:
    """Class to mock training and testing data for CnnModel."""
    def __init__(self):
        self.X = np.random.rand(10, 64, 64, 3)
        self.X_gender = np.random.rand(10, 1)
        self.y = np.random.rand(10)


class TestCnnModel(unittest.TestCase):

    def setUp(self):
        """Initialize mock data and CnnModel."""
        data_train = MockData()
        data_test = MockData()
        self.model = CnnModel(data_train, data_test)

    def test_initialize_model(self):
        """Test if CnnModel is initialized with mock data correctly."""

        # Check that training data has the correct shape
        self.assertEqual(self.model._X_train.shape, (10, 64, 64, 3))
        self.assertEqual(self.model._y_train.shape, (10,))
        self.assertEqual(self.model._X_gender_train.shape, (10, 1))

        # Check that test data has the correct shape
        self.assertEqual(self.model._X_test.shape, (10, 64, 64, 3))
        self.assertEqual(self.model._y_test.shape, (10,))
        self.assertEqual(self.model._X_gender_test.shape, (10, 1))


if __name__ == '__main__':
    unittest.main()
