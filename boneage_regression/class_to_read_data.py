import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from loguru import logger
import matlab.engine
import pandas as pd

def is_numeric(s):
    """Check if a given string represents a valid integer.

    :param s: The string to verify.
    :return: True if the string is an integer, False otherwise.
    """
    try:
        int(s)
        return True
    except ValueError:
        logger.warning(f"Value '{s}' is not valid. The image file name must be an integer.")
        return False

class DataLoader:
    """
    Class for loading and preprocessing the BoneAge dataset.
    """
    def __init__(self, image_path, labels_path, target_size=(128, 128), num_images=None, preprocessing=False, num_workers=12):
        """
        Initialize the DataLoader for the BoneAge dataset.

        :param image_path: Path to the folder containing the images.
        :param labels_path: Path to the file containing labels.
        :param target_size: Tuple specifying the target size for image resizing (default: (128, 128)).
        :param num_images: Number of images to load (default: None, loads all images).
        :param preprocessing: Boolean indicating whether preprocessing is required (default: False).
        :param num_workers: Number of workers for parallel preprocessing in MATLAB (default: 12).
        """
        self.image_path = pathlib.Path(image_path)
        self.labels_path = pathlib.Path(labels_path) if labels_path else None
        self.target_size = target_size
        self.num_images = num_images
        self.preprocessing = preprocessing
        self.num_workers = num_workers

    def preprocess_images(self):
        """
        Preprocess images using MATLAB.

        This function processes images in the `self.image_path` folder using parallel computation. 
        A MATLAB process is started, and the MATLAB function reads images from `self.image_path`, 
        preprocesses them, and saves them in a folder named `self.image_path + '_processed'`. 
        The `self.image_path` attribute is then updated accordingly.

        Notes:
            - Preprocessing includes pixel intensity normalization, resizing, and padding.
            - If your version of MATLAB does not support the specified number of workers, 
            the default value from your MATLAB environment will be used.

        :raises FileNotFoundError: If the specified image path does not exist.
        """

        logger.info("Performing MATLAB preprocessing...")

        eng = matlab.engine.start_matlab()
        eng.addpath(r'C:\Users\nicco\boneage_regression\boneage_regression_coding')
        eng.preprocessing(str(self.image_path), str(r'C:\Users\nicco\Desktop\output_images'), self.num_workers, self.target_size(1), nargout = 0) 
        #Number of workers for parallel preprocessing and dimension of images can also be set. Defualt values are 12 and 128.
        self.image_path = pathlib.Path(str(r'C:\Users\nicco\Desktop\output_images'))
        self.preprocessing = False  # Flag deactivation after preprocessing.
        eng.quit()

    def load_images(self):
        """
        Load images from `self.image_path` and apply preprocessing if specified.

        This function reads images from `self.image_path`, converts them to RGB, 
        resizes them to the target dimensions, and filters them according to the available labels.

        If `self.preprocessing` is True, the MATLAB preprocessing function is executed.

        :return: tuple containing:
            - `filtered_images_rgb` (np.ndarray): NumPy array of the preprocessed images.
            - `filtered_ids` (np.ndarray): NumPy array of valid image IDs (those with a corresponding label).
            - `labels` (np.ndarray or None): Corresponding labels for the images (or `None` if not available).

        :raises FileNotFoundError: If the image directory does not exist.
        """

        path = self.image_path

        if not path.is_dir():
            raise FileNotFoundError(f"No such file or directory: {path}")
        
        if self.preprocessing:
            self.preprocess_images()

        # Ordering file names
        image_files = [f for f in path.iterdir() if f.is_file() and is_numeric(f.stem)]
        image_files = sorted(image_files, key=lambda x: int(x.stem))

        if self.num_images:
            image_files = image_files[:self.num_images]

        images_rgb = []
        ids = []

        for img_path in image_files:
            img = plt.imread(img_path)
            img_id = int(img_path.stem)

            # Switch to RGB if needed (RGB are better from CNN point of view)
            if len(img.shape) == 2:  # BW images
                img = np.stack([img] * 3, axis=-1)

            # Resizing
            img_resized = tf.image.resize(img, self.target_size).numpy()
            
            images_rgb.append(img_resized)
            ids.append(img_id)

        logger.info(f"{len(images_rgb)} images loaded.")
        
        # Loading labels
        labels = None
        if self.labels_path:
            labels, missing_ids = self.load_labels(ids)
            
        # Discarding images whose ID is present in missing_ids (they would have no labels)
        filtered_images_rgb = [img for img, img_id in zip(images_rgb, ids) if img_id not in missing_ids]
        filtered_ids = [img_id for img_id in ids if img_id not in missing_ids]

        logger.info(f"{len(filtered_images_rgb)} images are ready to be used.")

        return np.array(filtered_images_rgb, dtype=np.uint8), np.array(filtered_ids, dtype=np.int32), labels
    
    
    def load_labels(self, image_ids):
        """
        Loads labels from a CSV file and returns only those corresponding to the provided image IDs.

        This function filters the labels to ensure that each image has a corresponding label
        and vice versa. Missing image IDs or labels without corresponding images are logged as warnings.

        :param image_ids: A list of IDs of the loaded images.
        :return: A tuple containing:
            - `label_pairs`: A NumPy array of pairs (boneage, gender) for each valid image.
            - `valid_ids`: A list of image IDs that have a corresponding label in the CSV.
        :raises FileNotFoundError: If the labels file cannot be found at the specified path.
        :raises ValueError: If required columns ('id', 'boneage', 'male') are missing from the CSV.
        """
        if not self.labels_path.is_file():
            raise FileNotFoundError(f"No such file or directory: {self.labels_path}")
        
        df = pd.read_csv(self.labels_path)
        df.columns = df.columns.str.lower()

        # Searching for images with no corresponding labels
        missing_ids = [img_id for img_id in image_ids if img_id not in df['id'].to_numpy()]

        # Searching for labels with no corresponding images
        missing_images = [label_id for label_id in df['id'].to_numpy() if label_id not in image_ids]

        if missing_ids:
            logger.warning(f"Warning: The following image IDs are missing in the label file: {missing_ids}")

        if missing_images:
            logger.warning(f"Warning: The following labels do not correspond to any image: {missing_images}")
        
        # Searching for missing informations
        required_columns = ['id', 'boneage', 'male']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        label_df = df[df['id'].isin(image_ids)]
        
        valid_ids = label_df['id'].to_numpy()

        boneage = label_df['boneage'].to_numpy()
        gender = label_df['male'].astype(int).to_numpy()
        
        # Creating array of couples (boneage, gender)
        label_pairs = np.array(list(zip(boneage, gender)))

        return label_pairs, missing_ids

    
loader = DataLoader(r"C:\Users\nicco\Desktop\Preprocessed_dataset_prova\Preprocessed_foto", r"C:\Users\nicco\Desktop\Preprocessed_dataset_prova\train.csv", preprocessing=False, num_images=20)
ids, images, labels = loader.load_images()