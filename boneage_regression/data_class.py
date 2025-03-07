import pathlib

import numpy as np
from loguru import logger
import pandas as pd

from utils import is_numeric, sorting_and_preprocessing

try:
    import matlab.engine
except ImportError:
    logger.info("matlab.engine package not found.")


class DataLoader:
    """
    Class for loading and preprocessing the BoneAge dataset.
    """

    def __init__(
        self, image_path, labels_path, target_size=(256, 256),
        num_images=None, preprocessing=False, num_workers=12
    ):
        """
        Initialize the DataLoader for the BoneAge dataset.

        :param image_path: Path to the folder containing the images.
        :type image_path: str
        :param labels_path: Path to the file containing labels.
        :type labels_path: str
        :param target_size: Target size for image
        resizing. Deafult: (256, 256).
        :type target_size: tuple(int, int)
        :param num_images: Number of images to load. Default: None
        (loads all images).
        :type num_images: int
        :param preprocessing: Boolean indicating whether preprocessing
        is required. Deafult: False.
        :type preprocessing: bool
        :param num_workers: Number of workers for parallel preprocessing
        in MATLAB. Default: 12.
        :type num_workers: int
        """
        self._image_path = pathlib.Path(image_path)
        self._labels_path = pathlib.Path(labels_path)
        self._target_size = target_size
        self._num_images = num_images
        self._preprocessing = preprocessing
        self._num_workers = num_workers
        self.X, self.ids, self.X_gender, self.y = self.load_images()

    @property
    def image_path(self):
        return self._image_path

    @image_path.setter
    def image_path(self, value):
        path = pathlib.Path(value)
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(
                f"Invalid path: {value}. The directory does not exist."
            )
        self._image_path = path

    @property
    def labels_path(self):
        return self._labels_path

    @labels_path.setter
    def labels_path(self, value):
        path = pathlib.Path(value)
        if not path.exists():
            raise FileNotFoundError(
                f"Invalid path: {value}. The file does not exist."
            )
        self._labels_path = path

    @property
    def target_size(self):
        return self._target_size

    @target_size.setter
    def target_size(self, value):
        if hasattr(self, "_target_size"):
            raise AttributeError(
                "target_size cannot be modified after assignment."
            )
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and all(isinstance(i, int) and i > 0 for i in value)
            and value[0] == value[1]
        ):
            self._target_size = value
        else:
            raise ValueError(
                "Invalid target_size: {value}. Must be a tuple of two "
                "identical positive integers."
            )

    @property
    def num_images(self):
        return self._num_images

    @num_images.setter
    def num_images(self, value):
        if hasattr(self, "_num_images"):
            raise AttributeError(
                "num_images cannot be modified after assignment."
            )
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError(
                f"Invalid num_images: {value}."
                f" Must be a positive integer or None."
            )
        self._num_images = value

    @property
    def preprocessing(self):
        return self._preprocessing

    @preprocessing.setter
    def preprocessing(self, value):
        if not isinstance(value, bool):
            raise ValueError(f"Invalid preprocessing: {value}."
                             f" preprocessing must be a boolean"
                             f" (True or False).")
        self._preprocessing = value

    def preprocess_images(self):
        """
        Preprocess images using MATLAB.

        This function processes images in the `self._image_path` folder using
        parallel computation. A MATLAB process is started, and the MATLAB
        function reads images from `self._image_path`, preprocesses them,
        and saves them in subfolders inside 'Processed_images' folder.
        The `self._image_path` attribute is then updated accordingly.

        Notes:

            - Preprocessing includes pixel intensity normalization, resizing,
            and padding.

            - If your version of MATLAB does not support the specified number
            of workers, the default value from your MATLAB environment will
            be used.
        """
        
        # Start a matlab process to augment contrast and center the images
        file_path = pathlib.Path(__file__).resolve()
        logger.info("Performing MATLAB preprocessing...")

        eng = matlab.engine.start_matlab()

        eng.addpath(str(file_path.parent / 'Matlab_function'))
        eng.preprocessing(
            str(self._image_path),
            str(file_path.parent.parent / 'Processed_images' / self._image_path.name ),
            self._num_workers, self._target_size[1], nargout=0
        )

        self._image_path = str(file_path.parent.parent / 'processed_images'/ self._image_path.name)
        eng.quit()
        logger.info(f"{self._image_path.name} processed images saved in processed_images/{self._image_path.name}")


    def load_images(self):
        """
        Load images from `self._image_path` and apply preprocessing if
        specified.

        This function reads images from `self._image_path`,
        converts them to RGB, resizes them to the target dimensions,
        and filters them according to the available labels.

        If `self._preprocessing` is True, the MATLAB preprocessing method
        is called.

        :return: tuple containing:

            - `filtered_images_rgb` (np.ndarray): NumPy array of the
            preprocessed images.

            - `filtered_ids` (np.ndarray): NumPy array of valid image IDs
            (those with a corresponding label).

            - `gender` (np.ndarray): Corresponding gender for the
            images (or `None` if not available).

            - `boneage` (np.ndarray): Corresponding age for the
            images (or `None` if not available).
        """

        path = self._image_path

        if self._preprocessing:
            self.preprocess_images()

        # Ordering file names
        image_files = [f for f in path.iterdir() if
                       f.is_file() and is_numeric(f.stem)]
        image_files = sorted(image_files, key=lambda x: int(x.stem))

        if self._num_images:
            image_files = image_files[:self._num_images]

        images, ids = sorting_and_preprocessing(image_files,
                                                    self._target_size)

        logger.info(f"{len(images)} images loaded.")

        # Loading labels
        labels, missing_ids = self.load_labels(ids)

        # Discarding images whose ID is present in missing_ids
        # (they would have no labels)
        filtered_images = [img for img, img_id in zip(images, ids) if
                               img_id not in missing_ids]
        filtered_ids = [img_id for img_id in ids if img_id not in missing_ids]

        logger.info(f"{len(filtered_images)} images are ready to be used.")

        boneage, gender = zip(*labels)  # Spiltting boneage and gender

        return (
            np.array(filtered_images, dtype=np.float32)/255,
            np.array(filtered_ids, dtype=np.int32),
            np.array(gender, dtype=np.int32).reshape(-1, 1),
            np.array(boneage, dtype=np.int32)
        )

    def load_labels(self, image_ids):
        """
        Loads labels from a CSV file and returns only those corresponding to
        the provided image IDs.

        This function filters the labels to ensure that each image has a
        corresponding label and vice versa. Missing image IDs or labels without
        corresponding images are logged as warnings.

        :param image_ids: IDs of the loaded images.
        :type image_ids: list

        :return: A tuple containing:

            - `label_pairs`: NumPy array of pairs (boneage, gender) for each
            valid image.

            - `valid_ids`: list of image IDs that have a corresponding label
            in the CSV.

        :raises FileNotFoundError: If the labels file cannot be found at the
        specified path.

        :raises ValueError: If required columns ('id', 'boneage', 'male') are
        missing from the CSV.
        """

        df = pd.read_csv(self.labels_path, nrows=self.num_images)
        df.columns = df.columns.str.lower()

        # Searching for missing informations
        required_columns = ['id', 'boneage', 'male']
        missing_columns = [col for col in required_columns if col not in
                           df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Searching for images with no corresponding labels
        missing_ids = [img_id for img_id in image_ids if
                       img_id not in df['id'].to_numpy()]

        # Searching for labels with no corresponding images
        missing_images = [label_id for label_id in df['id'].to_numpy() if
                          label_id not in image_ids]

        if missing_ids:
            logger.warning(f"Warning: The following image IDs are missing in"
                           f" the label file:"
                           f"{', '.join(map(str, missing_ids))}")

        if missing_images:
            logger.warning(f"Warning: The following labels do not correspond"
                           f" to any image:"
                           f"{', '.join(map(str, missing_images))}")

        label_df = df[df['id'].isin(image_ids)]

        boneage = label_df['boneage'].to_numpy()
        gender = label_df['male'].astype(int).to_numpy()

        # Creating array of couples (boneage, gender)
        label_pairs = np.array(list(zip(boneage, gender)))

        return label_pairs, missing_ids