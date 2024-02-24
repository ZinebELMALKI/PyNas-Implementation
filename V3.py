import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, Input
from keras.models import Model
from keras.optimizers import Adam
import cv2
import os
import time
import matplotlib.pyplot as plt


start_time = time.time()  # Enregistrez le temps de début

# Chemin vers le dossier contenant les images floues
blurry_image_folder = r"C:\Users\USER\Desktop\pynas\blurred_images"

# Chemin vers le dossier contenant les images nettes
clear_image_folder = r"C:\Users\USER\Desktop\pynas\clear_images"

# Liste pour stocker les images floues et nettes
blurry_images = []
clear_images = []

# Parcourir les fichiers dans les dossiers respectifs
for filename in os.listdir(blurry_image_folder):
    if filename.endswith(('.jpg', '.png')):
        image_path = os.path.join(blurry_image_folder, filename)
        image = cv2.imread(image_path)
        # Convertir de BGR à RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        blurry_images.append(image)

for filename in os.listdir(clear_image_folder):
    if filename.endswith(('.jpg', '.png')):
        image_path = os.path.join(clear_image_folder, filename)
        image = cv2.imread(image_path)
        # Convertir de BGR à RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        clear_images.append(image)

# Convertir les listes en tableaux numpy
x_train = np.array(blurry_images)
y_train = np.array(clear_images)

# Définir le modèle de défloutage (exemple simple)

input_shape = x_train[0].shape  # Utilisez la forme d'une image d'entraînement
inputs = Input(shape=input_shape)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
outputs = Conv2D(3, (3, 3), activation='relu', padding='same')(
    x)  # Activation 'relu' pour la couleur


model = Model(inputs, outputs)
model.compile(optimizer=Adam(lr=0.001), loss='mse')

# Entraîner le modèle avec les données préparées
# Exemple avec 100 époques et une taille de lot de 32
history = model.fit(x_train, y_train, epochs=100, batch_size=32)


# Créez un dossier pour enregistrer les images défloutées s'il n'existe pas
output_folder = r"C:\Users\USER\Desktop\pynas\results"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Chemin vers le dossier contenant les images floues à déflouter
input_folder = r"C:\Users\USER\Desktop\pynas\imgs"


# Parcourez les images floues du dossier et défloutez-les
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        # Appliquez le modèle pour déflouter l'image
        deblurred_image = model.predict(np.expand_dims(image, axis=0))[0]

        # Enregistrez l'image défloutée
        deblurred_image = (deblurred_image * 255).astype(np.uint8)
        output_path = os.path.join(output_folder, f"deblurred_{filename}")
        cv2.imwrite(output_path, cv2.cvtColor(
            deblurred_image, cv2.COLOR_RGB2BGR))

print(f"Images defloutees enregistrees dans le dossier : {output_folder}")


end_time = time.time()  # Enregistrez le temps de fin

# Calculez la durée d'exécution en secondes
execution_time = end_time - start_time

print(f"Temps d'exécution : {execution_time:.2f} secondes")


# Tracer la fonction de perte pendant l'entraînement
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Perte d\'entraînement')
plt.xlabel('Époques')
plt.ylabel('Perte')
plt.title('Évolution de la perte pendant l\'entraînement')
plt.legend()
plt.grid(True)
plt.show()
