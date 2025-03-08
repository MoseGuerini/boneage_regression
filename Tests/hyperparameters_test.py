import unittest
import sys
import pathlib

# Add the "boneage_regression" directory to sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "boneage_regression"))

from hyperparameters import hyperp_space_size
from utils import hyperp_dict


class TestHyperparameters(unittest.TestCase):
    def test_hyperp_space_size(self):

        hyp = hyperp_dict([1, 2], [32, 64], [0.2, 0.5], [2, 3])
        self.assertEqual(hyperp_space_size(hyp), 16)

        hyp = hyperp_dict([1], [64], [0.5], [3])
        self.assertEqual(hyperp_space_size(hyp), 1)


if __name__ == '__main__':
    unittest.main()
