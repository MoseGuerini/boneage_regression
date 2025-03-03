import unittest
import sys
import pathlib

# Aggiungi "boneage_regression" a sys.path
sys.path.insert(
    0,
    str(pathlib.Path(__file__).resolve().parent.parent / "boneage_regression")
)

# Import delle classi e funzioni
from data_class import DataLoader
from data_class import is_numeric


class TestDataLoader(unittest.TestCase):

    def test_paths(self):
        """Valid paths must let the function proceed"""

        # Obtain script folder and parent folder
        current_dir = pathlib.Path(__file__).parent
        parent_dir = current_dir.parent

        # Attach the correct paths
        valid_image_path = parent_dir / "Test_dataset" / "Test"
        valid_labels_path = parent_dir / "Test_dataset" / "test.csv"

        # Test: DataLoader must accept valid paths
        loader = DataLoader(
            image_path=valid_image_path,
            labels_path=valid_labels_path
            )
        self.assertEqual(loader.image_path, valid_image_path)
        self.assertEqual(loader.labels_path, valid_labels_path)

        """If one or more paths are invalid, the function must not preceed"""
        invalid_labels_path = parent_dir / "unexisting_path"
        invalid_image_path = parent_dir / "unexisting_path"

        with self.assertRaises(FileNotFoundError):
            DataLoader(
                image_path=valid_image_path,
                labels_path=invalid_labels_path
                )

        with self.assertRaises(FileNotFoundError):
            DataLoader(
                image_path=invalid_image_path,
                labels_path=valid_labels_path
                )

    def test_load_labels_missing_columns(self):
        """If a column is missing the csv file must not be accepted"""

        # Obtain script folder and parent folder
        current_dir = pathlib.Path(__file__).parent
        parent_dir = current_dir.parent

        # Attach the correct paths
        valid_image_path = parent_dir / "Test_dataset" / "Test"
        invalid_labels_path = (
            parent_dir / "Test_dataset" / "training_missing_column.csv"
        )

        with self.assertRaises(ValueError):
            DataLoader(
                image_path=valid_image_path,
                labels_path=invalid_labels_path
                )

    def test_target_size(self):
        """Only a couple of integer must be accepted"""

        # Obtain script folder and parent folder
        current_dir = pathlib.Path(__file__).parent
        parent_dir = current_dir.parent

        # Attach the correct paths
        valid_image_path = parent_dir / "Test_dataset" / "Test"
        valid_labels_path = parent_dir / "Test_dataset" / "test.csv"

        valid_target_size = (256, 256)
        loader = DataLoader(
            image_path=valid_image_path,
            labels_path=valid_labels_path,
            target_size=valid_target_size
            )
        self.assertEqual(loader.target_size, valid_target_size)
        invalid_target_size = 256
        with self.assertRaises(ValueError):
            DataLoader(
                image_path=valid_image_path,
                labels_path=valid_labels_path,
                target_size=invalid_target_size
                )
        invalid_target_size = 'abc'
        with self.assertRaises(ValueError):
            DataLoader(
                image_path=valid_image_path,
                labels_path=valid_labels_path,
                target_size=invalid_target_size
                )
        invalid_target_size = (0.1, 0.1)
        with self.assertRaises(ValueError):
            DataLoader(
                image_path=valid_image_path,
                labels_path=valid_labels_path,
                target_size=invalid_target_size
                )
        invalid_target_size = (128, 'abc')
        with self.assertRaises(ValueError):
            DataLoader(
                image_path=valid_image_path,
                labels_path=valid_labels_path,
                target_size=invalid_target_size
                )
        invalid_target_size = (128, -128)
        with self.assertRaises(ValueError):
            DataLoader(
                image_path=valid_image_path,
                labels_path=valid_labels_path,
                target_size=invalid_target_size
                )
        invalid_target_size = (128, 64)
        with self.assertRaises(ValueError):
            DataLoader(
                image_path=valid_image_path,
                labels_path=valid_labels_path,
                target_size=invalid_target_size
                )

    def test_num_images(self):
        """The function must only accept an integer as num_images"""

        # Obtain script folder and parent folder
        current_dir = pathlib.Path(__file__).parent
        parent_dir = current_dir.parent

        # Attach the correct paths
        valid_image_path = parent_dir / "Test_dataset" / "Test"
        valid_labels_path = parent_dir / "Test_dataset" / "test.csv"

        valid_num_imges = 10
        loader = DataLoader(
            image_path=valid_image_path,
            labels_path=valid_labels_path,
            num_images=valid_num_imges
            )
        self.assertEqual(loader.num_images, valid_num_imges)
        valid_num_imges = None
        loader = DataLoader(
            image_path=valid_image_path,
            labels_path=valid_labels_path,
            num_images=valid_num_imges
            )
        self.assertEqual(loader.num_images, valid_num_imges)
        invalid_num_images = (256, 1)
        with self.assertRaises(ValueError):
            DataLoader(
                image_path=valid_image_path,
                labels_path=valid_labels_path,
                num_images=invalid_num_images
                )
        invalid_num_images = 'abc'
        with self.assertRaises(ValueError):
            DataLoader(
                image_path=valid_image_path,
                labels_path=valid_labels_path,
                num_images=invalid_num_images
                )
        invalid_num_images = 0.1
        with self.assertRaises(ValueError):
            DataLoader(
                image_path=valid_image_path,
                labels_path=valid_labels_path,
                num_images=invalid_num_images
                )
        invalid_num_images = -10
        with self.assertRaises(ValueError):
            DataLoader(
                image_path=valid_image_path,
                labels_path=valid_labels_path,
                num_images=invalid_num_images
                )

"""     def test_preprocessing(self):
        """The function must only accept an integer as preprocessing"""

        # Obtain script folder and parent folder
        current_dir = pathlib.Path(__file__).parent
        parent_dir = current_dir.parent

        # Attach the correct paths
        valid_image_path = parent_dir / "Test_dataset" / "Test"
        valid_labels_path = parent_dir / "Test_dataset" / "test.csv"

        valid_preprocessing = True
        valid_num_workers = 8
        valid_target_size = (64, 64)
        loader_true = DataLoader(
            image_path=valid_image_path,
            labels_path=valid_labels_path,
            preprocessing=valid_preprocessing,
            num_workers=valid_num_workers,
            target_size=valid_target_size
            )
        self.assertEqual(loader_true.preprocessing, valid_preprocessing)
        valid_preprocessing = False
        loader_false = DataLoader(
            image_path=valid_image_path,
            labels_path=valid_labels_path,
            preprocessing=valid_preprocessing,
            num_workers=valid_num_workers,
            target_size=valid_target_size
            )
        self.assertEqual(loader_false.preprocessing, valid_preprocessing)
        # Now be sure preprocessing does not lose images
        self.assertEqual(len(loader_true.X), len(loader_false.X))
        invalid_preprocessing = (256, 1)
        with self.assertRaises(ValueError):
            DataLoader(
                image_path=valid_image_path,
                labels_path=valid_labels_path,
                preprocessing=invalid_preprocessing,
                num_workers=valid_num_workers,
                target_size=valid_target_size
                )
        invalid_preprocessing = 'abc'
        with self.assertRaises(ValueError):
            DataLoader(
                image_path=valid_image_path,
                labels_path=valid_labels_path,
                preprocessing=invalid_preprocessing,
                num_workers=valid_num_workers,
                target_size=valid_target_size
                )
        invalid_preprocessing = 0.1
        with self.assertRaises(ValueError):
            DataLoader(
                image_path=valid_image_path,
                labels_path=valid_labels_path,
                preprocessing=invalid_preprocessing,
                num_workers=valid_num_workers,
                target_size=valid_target_size
                )
        invalid_preprocessing = -10
        with self.assertRaises(ValueError):
            DataLoader(
                image_path=valid_image_path,
                labels_path=valid_labels_path,
                preprocessing=invalid_preprocessing,
                num_workers=valid_num_workers,
                target_size=valid_target_size
                )
        invalid_preprocessing = 1
        with self.assertRaises(ValueError):
            DataLoader(
                image_path=valid_image_path,
                labels_path=valid_labels_path,
                preprocessing=invalid_preprocessing,
                num_workers=valid_num_workers,
                target_size=valid_target_size
                ) """


if __name__ == '__main__':
    unittest.main()
