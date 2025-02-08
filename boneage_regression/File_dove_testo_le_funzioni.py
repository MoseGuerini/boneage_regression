from utils import load_images
import numpy as np
import tensorflow as tf
from PIL import Image
from cnn import cnn

image, ids = load_images(r'C:\Users\nicco\Desktop\foto')
target_size = (128, 128)  # Imposta una dimensione fissa

# Assicurati che le immagini siano in scala di grigi e poi espandi a 3 canali
image_rgb = []
for img in image:
    if len(img.shape) == 2:  # Se l'immagine è in scala di grigi (1 canale)
        img_rgb = np.stack([img] * 3, axis=-1)  # Crea 3 canali duplicati
    else:
        img_rgb = img  # Se già ha 3 canali (RGB), la lascio così com'è
    image_rgb.append(img_rgb)

# Converti in un array NumPy
image_rgb = np.array(image_rgb, dtype=np.float32)

# Ridimensiona tutte le immagini
images_resized = tf.image.resize(image_rgb, target_size).numpy()
print(images_resized.shape)  # Dovrebbe essere (num_immagini, 128, 128, 3)

utils_path = Path(__file__).resolve().parent.parent / "boneage_regression"
sys.path.insert(0, str(utils_path))

