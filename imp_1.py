
# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, UpSampling2D, Concatenate
# from tensorflow.keras.models import Model

# # Chargement et prétraitement des données
# image_folder_path = r"C:\Users\USER\Desktop\pynas\imgs"

# images = []
# for filename in os.listdir(image_folder_path):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#         image_path = os.path.join(image_folder_path, filename)
#         image = cv2.imread(image_path)
#         image = cv2.resize(image, (128, 128))
#         images.append(image)

# images = np.array(images) / 255
# print("Nombre d'images à traiter :", len(images))

# # Définition des paramètres
# input_shape = (128, 128, 3)
# kernel_size = 3
# layer_filters = [64, 128, 256]
# num_scales = 3
# batch_size = 32
# latent_dim = 256

# # Architecture du modèle Autoencoder pyramidale
# input_layer = Input(shape=input_shape, name='input_image')

# # Encodeur pyramidale
# encoded_list = []
# x = input_layer
# for scale in range(num_scales):
#     for filters in layer_filters:
#         x = Conv2D(filters=filters, kernel_size=kernel_size,
#                    strides=2, activation='relu', padding='same')(x)
#     encoded_list.append(x)

# # Décodeur pyramidale
# decoded_list = []
# for scale in range(num_scales - 1, -1, -1):
#     x = encoded_list[scale]
#     for filters in layer_filters[::-1]:
#         x = Conv2DTranspose(filters=filters, kernel_size=kernel_size,
#                             strides=2, activation='relu', padding='same')(x)
#         x = UpSampling2D()(x)  # Mise à l'échelle pour ajuster les dimensions
#     decoded_list.append(x)

# # Réajuster les dimensions des images décodées à la taille des images d'entrée (128x128)
# decoded_list = [tf.image.resize(decoded, input_shape[:2])
#                 for decoded in decoded_list]

# concatenated_decoded = Concatenate(axis=-1)(decoded_list)

# final_output = Conv2DTranspose(filters=3, kernel_size=kernel_size,
#                                activation='sigmoid', padding='same')(concatenated_decoded)

# pyramid_autoencoder = Model(
#     inputs=input_layer, outputs=final_output, name='pyramid_autoencoder')
# pyramid_autoencoder.compile(loss='mse', optimizer='adam')

# # Entraînement du modèle
# history = pyramid_autoencoder.fit(images, images, epochs=100, batch_size=32)

# # Prédiction et enregistrement des images décodées individuellement
# output_folder = r"C:\Users\USER\Desktop\pynas\results"
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# for i, image in enumerate(images):
#     decoded_image = pyramid_autoencoder.predict(
#         np.expand_dims(image, axis=0))[0]
#     output_path = os.path.join(output_folder, f"decoded_image_{i}.png")
#     decoded_image = (decoded_image * 255).astype(np.uint8)
#     cv2.imwrite(output_path, decoded_image)

# print("Images décodées enregistrées dans le dossier :", output_folder)

import os
import numpy as np
from PIL import Image
from keras.layers import Dense, Input, Conv2D, Flatten, Reshape, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K


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
encoded = Dense(latent_dim)(x)

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
autoencoder.compile(optimizer=Adam(lr=0.001), loss='mse')


# Entraîner le modèle
x_train = np.array(blurry_images)
autoencoder.fit(x_train, x_train, epochs=100,
                batch_size=32)


# Générer des images nettes
decoded_images = autoencoder.predict(x_train)

# Enregistrer les images nettes dans le dossier de sortie
for i, decoded_image in enumerate(decoded_images):
    decoded_image *= 255.0  # Reconvertir les valeurs entre 0 et 255
    decoded_image = decoded_image.astype(np.uint8)
    output_path = os.path.join(output_folder, f"decoded_image_{i}.png")
    Image.fromarray(decoded_image).save(output_path)

print("Images nettes enregistrees dans le dossier :", output_folder)
