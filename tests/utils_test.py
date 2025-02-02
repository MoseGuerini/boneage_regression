import unittest
import sys
from pathlib import Path

# Risale di un livello nella struttura del progetto e accede alla cartella "utils"
utils_path = Path(__file__).resolve().parent.parent / "boneage_regression"

# Aggiunge la cartella "utils" al percorso di ricerca dei moduli
sys.path.insert(0, str(utils_path))

import utils

class UtilsTest(unittest.TestCase):

    def test_load_images(self):
        with self.assertRaises(FileNotFoundError):
            utils.load_images('This_directory_does_not_exist')