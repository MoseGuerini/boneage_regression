from utils import return_dataset, load_images, load_labels

image_path = r"C:\Users\nicco\Desktop\Test_dataset\Training"  # Inserisci il percorso alla cartella delle immagini
labels_path = r"C:\Users\nicco\Desktop\Test_dataset\training.csv.csv"  # Inserisci il percorso al file delle etichette

num_images = 30; #specifica quante immagini correi che fossero caricate

# Esegui la funzione
try:
    # Carica una parte delle immagini e i nomi
    features, names = load_images(image_path)
    feature, labels_filtered, gender_filtered = return_dataset(image_path, labels_path, num_images)
    #print(f"Features: {features}")
    print(f"Names: {names}")
    print(f"Ages: {labels_filtered}")
    print(f"Gender: {gender_filtered}")
except ValueError as e:
    print(f"Error: {e}")
