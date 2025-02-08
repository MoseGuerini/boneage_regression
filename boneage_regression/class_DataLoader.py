import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matlab.engine
from loguru import logger
import matplotlib as plt

class DataLoader:
    def __init__(self, image_path, labels_path, preprocessing=False, target_size=(128, 128), num_images=None):
        """
        Classe per il caricamento e preprocessing del dataset di Bone Age.

        :param image_path: Percorso della cartella contenente le immagini.
        :param labels_path: Percorso del file CSV con le etichette.
        :param img_size: Dimensione target delle immagini (default: 128x128).
        :param num_images: Numero di immagini da caricare (default: None, carica tutte).
        """
        self.image_path = pathlib.Path(image_path)
        self.labels_path = pathlib.Path(labels_path)
        self.target_size = target_size
        self.num_images = num_images
        self.preprocessing = preprocessing
        

    def load_images(self):
        """Carica le immagini dalla cartella, ridimensionandole se necessario."""
        path = pathlib.Path(self.image_path)

        if not path.is_dir():
            raise FileNotFoundError(f'No such file or directory: {path}')
        
        if self.preprocessing is True:
            self.preprocess_images(self.image_path)
            self.preprocessing = False
            path = pathlib.Path(self.image_path)
            
               # Elenco delle immagini nella cartella, ordinate per nome
        
        # Ordina le immagini per nome
        image_files = sorted(self.image_path.iterdir(), key=lambda x: int(x.stem))

        # Se necessario, carica solo un subset
        if self.num_images:
            image_files = image_files[:self.num_images]
            
        images = []
        images_rgb = []
        ids = []
        
        for img in image_files:
            if len(img.shape) == 2:  # Se l'immagine è in scala di grigi (1 canale)
                img_rgb = np.stack([img] * 3, axis=-1)  # Crea 3 canali duplicati
            else:
                img_rgb = img  # Se già ha 3 canali (RGB), la lascio così com'è

        logger.info(f'Read images from the dataset')
    
        # Carica le immagini
        images = [plt.imread(name) for name in self.num_images]
    
        # Estrai gli ID dalle immagini
        id = [name.stem for name in self.num_images]

        for img in images:
            if len(img.shape) == 2:  # Se l'immagine è in scala di grigi (1 canale)
                img_rgb = np.stack([img] * 3, axis=-1)  # Crea 3 canali duplicati
            else:
                img_rgb = img  # Se già ha 3 canali (RGB), la lascio così com'è
            image_rgb_resized = tf.image.resize(img_rgb, self.target_size).numpy()
            images_rgb.append(image_rgb_resized)

        return np.array(images_rgb, dtype=np.uint8), np.array(ids, dtype=np.int32)

    def load_labels(self):
        """Carica le etichette dal file CSV."""
        if not self.labels_path.is_file():
            raise FileNotFoundError(f"File CSV non trovato: {self.labels_path}")

        df = pd.read_csv(self.labels_path)

        # Assicura che ci siano le colonne giuste
        required_columns = ["id", "boneage", "male"]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Mancano le colonne: {missing_cols}")

        return df["id"].to_numpy(), df["boneage"].to_numpy(), df["male"].astype(int).to_numpy()

    def return_dataset(self):
        """Restituisce feature (immagini) e labels (età ossea e genere)."""
        images, image_ids = self.load_images()
        label_ids, boneage, gender = self.load_labels()

        # Filtra le etichette in base alle immagini disponibili
        valid_indices = [np.where(label_ids == img_id)[0][0] for img_id in image_ids if img_id in label_ids]
        
        boneage_filtered = boneage[valid_indices]
        gender_filtered = gender[valid_indices]

        return images, boneage_filtered, gender_filtered

    def preprocess_images(self):
        """Preprocessing: ridimensiona e normalizza immagini."""
        eng = matlab.engine.start_matlab()
        eng.preprocessing(self.image_path, r"Preprocessed_images")
        eng.quit()
        
        self.image_path = r"Preprocessed_images"



# --- ESEMPIO DI UTILIZZO ---
dataset = BoneAgeDataset(image_path="data/images", labels_path="data/boneage.csv", img_size=(128, 128), num_images=200)
images, boneage, gender = dataset.return_dataset()
images = dataset.preprocess_images(images)

print("Shape delle immagini preprocessate:", images.shape)
print("Boneage shape:", boneage.shape)
print("Gender shape:", gender.shape)
