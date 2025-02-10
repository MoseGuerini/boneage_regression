import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from scipy.ndimage import zoom

# Carica il modello pre-addestrato
res_model = ResNet50()
res_model.summary()

# Caricamento immagine senza OpenCV
img_path = r'C:\Users\nicco\Desktop\cat_image.png'
img = image.load_img(img_path, target_size=(224, 224))  # Ridimensiona l'immagine
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
img_array = preprocess_input(img_array)

# Modello per estrarre feature maps
conv_output = res_model.get_layer("conv5_block3_out").output
pred_output = res_model.get_layer("predictions").output
model = Model(res_model.input, outputs=[conv_output, pred_output])

# Ottieni predizione e feature maps
conv, pred = model.predict(img_array)

# Decodifica predizione
decoded_preds = decode_predictions(pred)
print(decoded_preds)

# Visualizzazione feature maps
scale = 224 / conv.shape[1]
plt.figure(figsize=(16, 16))
for i in range(36):
    plt.subplot(6, 6, i + 1)
    plt.imshow(img)
    plt.imshow(zoom(conv[0, :, :, i], zoom=(scale, scale)), cmap='jet', alpha=0.3)
plt.show()

# Calcolo Heatmap
target = np.argmax(pred, axis=1).squeeze()
w, b = res_model.get_layer("predictions").get_weights()  # Ottieni pesi del layer di output
weights = w[:, target]  # Estrai i pesi della classe predetta
heatmap = np.dot(conv.squeeze(), weights)  # Prodotto scalare per ottenere la heatmap

# Visualizza heatmap
plt.figure(figsize=(12, 12))
plt.imshow(img)
plt.imshow(zoom(heatmap, zoom=(scale, scale)), cmap='jet', alpha=0.5)
plt.show()
