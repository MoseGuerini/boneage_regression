import unittest
from boneage_regression.hyperparameters import hyperp_space_size
from boneage_regression.utils import hyperp_dict

class TestHyperpSpaceSize(unittest.TestCase):
    def test_hyperp_space_size(self):
        _ = hyperp_dict ([1, 2], [32, 64], [0.2, 0.5], [128, 256], [2, 3])

        expected_size = 32
        self.assertEqual(hyperp_space_size(), expected_size)

        _ = hyperp_dict (1, 64, 0.5, 256, 3)

        expected_size = 1
        self.assertEqual(hyperp_space_size(), expected_size)

if __name__ == '__main__':
    unittest.main()
