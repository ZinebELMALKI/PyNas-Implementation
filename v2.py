# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt  # Pour l'affichage de l'image

# # Définir le dossier d'entrée et de sortie
# input_folder = r"C:\Users\USER\Desktop\pynas\imgs"
# output_folder = r"C:\Users\USER\Desktop\pynas\results"

# # Créer le dossier de sortie s'il n'existe pas déjà
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # Parcourir toutes les images dans le dossier d'entrée
# for image_filename in os.listdir(input_folder):
#     if image_filename.endswith(('.jpg', '.png')):
#         # Charger l'image floue
#         image_path = os.path.join(input_folder, image_filename)
#         image = cv2.imread(image_path)
#         # Convertir de BGR (par défaut d'OpenCV) à RGB
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Appliquer une méthode de défloutage (par exemple, défloutage gaussien)
#         deblurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX=2)

#         # Afficher l'image floue et l'image défloutée côte à côte
#         plt.figure(figsize=(10, 5))
#         plt.subplot(1, 2, 1)
#         plt.imshow(image)
#         plt.title('Image Floue')
#         plt.axis('off')

#         plt.subplot(1, 2, 2)
#         plt.imshow(deblurred_image)
#         plt.title('Image Défloutée')
#         plt.axis('off')

#         plt.show()

#         # Sauvegarder l'image défloutée dans le dossier de sortie
#         output_path = os.path.join(output_folder, image_filename)
#         cv2.imwrite(output_path, cv2.cvtColor(
#             deblurred_image, cv2.COLOR_RGB2BGR))

import os
import numpy as np
from PIL import Image
from keras.layers import Dense, Input, Conv2D, Flatten, Reshape, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import matplotlib.pyplot as plt

# Chemin vers le dossier contenant les images floues
blurry_image_folder = r"C:\Users\USER\Desktop\pynas\imgs"

# Chemin vers le dossier où vous souhaitez enregistrer les images nettes
output_folder = r"C:\Users\USER\Desktop\pynas\results"

# Assurez-vous que le dossier de sortie existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Charger les images floues depuis le dossier
blurry_images = []
for filename in os.listdir(blurry_image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image = Image.open(os.path.join(blurry_image_folder, filename))
        # Normaliser les valeurs des pixels entre 0 et 1
        image_array = np.array(image) / 255.0
        blurry_images.append(image_array)

# Paramètres réseau
input_shape = (128, 128, 3)
kernel_size = 3
latent_dim = 256

# Définir l'encodeur
inputs = Input(shape=input_shape)
x = inputs
for filters in [64, 128, 256]:
    x = Conv2D(filters=filters, kernel_size=kernel_size,
               activation='relu', strides=2, padding='same')(x)
shape_before_flattening = K.int_shape(x)
x = Flatten()(x)
encoded = Dense(latent_dim, activation='relu')(
    x)  # Ajout de la fonction d'activation ReLU

# Définir le décodeur
x = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(encoded)
x = Reshape(shape_before_flattening[1:])(x)
for filters in [256, 128, 64]:
    x = Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                        activation='relu', strides=2, padding='same')(x)
decoded = Conv2DTranspose(
    filters=3, kernel_size=kernel_size, activation='sigmoid', padding='same')(x)

# Créer le modèle
autoencoder = Model(inputs, decoded)
# Diminution du taux d'apprentissage
autoencoder.compile(optimizer=Adam(lr=0.0001), loss='mse')

# Ajout de rappels pour sauvegarder le modèle
model_checkpoint = ModelCheckpoint(
    'autoencoder_best_model.h5', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)

# Sélectionner un sous-ensemble d'images pour l'entraînement
num_samples = 10  # Choisissez le nombre d'échantillons souhaité
x_train = np.array(blurry_images[:num_samples])

# Entraîner le modèle

# Entraîner le modèle sans division de validation
history = autoencoder.fit(x_train, x_train, epochs=100, batch_size=32,
                          callbacks=[model_checkpoint, reduce_lr])


# Visualisation des courbes d'apprentissage (utilisez la perte d'entraînement)
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Charger le meilleur modèle enregistré
autoencoder.load_weights('autoencoder_best_model.h5')

# Générer des images nettes
decoded_images = autoencoder.predict(x_train)

# Enregistrer les images nettes dans le dossier de sortie
for i, decoded_image in enumerate(decoded_images):
    decoded_image *= 255.0  # Reconvertir les valeurs entre 0 et 255
    decoded_image = decoded_image.astype(np.uint8)
    output_path = os.path.join(output_folder, f"decoded_image_{i}.png")
    Image.fromarray(decoded_image).save(output_path)

print("Images nettes enregistrées dans le dossier :", output_folder)
