import pathlib

import numpy as np
from loguru import logger
import pandas as pd

from utils import is_integer, convert_and_resize

try:
    import matlab.engine
except ImportError:
    logger.info("matlab.engine package not found.")


class DataLoader:
    """
    A class for loading and preprocessing the BoneAge dataset.

    This class is responsible for reading images from a specified directory,
    loading corresponding labels from a CSV file, resizing images to a target
    size, handling missing labels, and optionally performing preprocessing
    using MATLAB. The images and labels are stored as NumPy arrays, ready
    for use in machine learning models.

    Attributes:
        _image_path: Path to the directory containing the images.

        _labels_path: Path to the CSV file containing image labels.

        _target_size: Target size for image resizing (must be square).

        _num_images: Number of images to load (None to load all).

        _preprocessing: Boolean flag indicating whether MATLAB preprocessing
        is applied.

        _num_workers: Number of workers for parallel preprocessing in MATLAB.

        X: NumPy array containing the preprocessed images.

        ids: NumPy array containing the corresponding image IDs.

        X_gender: NumPy array containing gender labels.

        y: NumPy array containing bone age labels.

    Methods:
        __init__(image_path, labels_path, target_size=(256, 256),
                 num_images=None, preprocessing=False, num_workers=12):
            Initializes the DataLoader instance and loads the images
            and labels.

        preprocess_images():
            Applies MATLAB-based preprocessing to the images, including
            intensity normalization, resizing, and padding.

        load_images():
            Loads images from the dataset directory, resizes them,
            filters missing labels, and applies preprocessing if enabled.

        load_labels(image_ids):
            Loads labels from the CSV file, ensuring that each image
            has a corresponding label and vice versa.
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
        self.image_path = pathlib.Path(image_path)
        self.labels_path = pathlib.Path(labels_path)
        self.target_size = target_size
        self.num_images = num_images
        self.preprocessing = preprocessing
        self.num_workers = num_workers
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
    def num_workers(self):
        return self._num_workers
    
    @num_workers.setter
    def num_workers(self, value):
        if hasattr(self, "_num_workers"):
            raise AttributeError(
                "num_workers cannot be modified after assignment."
            )
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError(
                f"Invalid num_workers: {value}."
                f" Must be a positive integer or None."
            )
        self._num_workers = value
    
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
        dir_path = pathlib.Path().resolve()

        logger.info("Performing MATLAB preprocessing...")
        eng = matlab.engine.start_matlab()

        eng.addpath(str(dir_path / 'Matlab_function'))

        eng.preprocessing(
            str(self._image_path),
            str(dir_path.parent / 'Preprocessed_images' / self._image_path.name),
            self._num_workers,
            self._target_size[1],
            nargout=0
        )

        self._image_path = (
            dir_path.parent / 'Preprocessed_images' / self._image_path.name
        )
        eng.quit()

        logger.info(
            f"{self._image_path.name} processed images saved in "
            f"Preprocessed_images/{self._image_path.name}"
        )

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
            images.

            - `boneage` (np.ndarray): Corresponding age for the
            images.
        """

        path = self._image_path

        if self._preprocessing:
            self.preprocess_images()

        # Ordering file names numerically
        image_files = [f for f in path.iterdir() if
                       f.is_file() and is_integer(f.stem)]
        image_files = sorted(image_files, key=lambda x: int(x.stem))

        if self._num_images:
            image_files = image_files[:self._num_images]

        images, ids = convert_and_resize(image_files,
                                         self._target_size)

        logger.info(f"{len(images)} images loaded.")

        # Loading labels and identifying missing ones
        boneage, gender, missing_ids = self.load_labels(ids)

        # Filtering out images with missing labels
        filtered_images = [img for img, img_id in zip(images, ids) if
                           img_id not in missing_ids]
        filtered_ids = [img_id for img_id in ids if img_id not in missing_ids]

        logger.info(f"{len(filtered_images)} images are ready to be used.")

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

            - `gender` (np.ndarray): Corresponding gender for the
            images.

            - `boneage` (np.ndarray): Corresponding age for the
            images.

            - `valid_ids`: list of image IDs that have a corresponding label
            in the CSV.

        :raises ValueError: If required columns ('id', 'boneage', 'male') are
        missing from the CSV.
        """

        df = pd.read_csv(self._labels_path, nrows=self._num_images)
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
            logger.warning("Warning: The following image IDs are missing in"
                           " the label file:"
                           f"{', '.join(map(str, missing_ids))}")

        if missing_images:
            logger.warning("Warning: The following labels do not correspond"
                           " to any image:"
                           f"{', '.join(map(str, missing_images))}")

        label_df = df[df['id'].isin(image_ids)]

        boneage = label_df['boneage'].to_numpy()
        gender = label_df['male'].astype(int).to_numpy()

        return boneage, gender, missing_ids
