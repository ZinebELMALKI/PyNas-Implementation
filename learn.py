from PIL import ImageFilter
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Charger l'image
img = Image.open(r"C:\Users\USER\Desktop\pynas\imgs\arbresblurred.png")

# Afficher l'image chargée
# img.show()

# Récupérer et afficher la taille de l'image (en pixels)
w, h = img.size
print("Largeur : {} px, hauteur : {} px".format(w, h))

# Afficher son mode de quantification
print("Format des pixels : {}".format(img.mode))

# Récupérer et afficher la valeur du pixel à une position précise
px_value = img.getpixel((20, 100))
print("Valeur du pixel situé en (20,100) : {}".format(px_value))


# Récupérer les valeurs de tous les pixels sous forme d'une matrice
mat = np.array(img)
mat

# Afficher la taille de la matrice de pixels
print("Taille de la matrice de pixels : {}".format(mat.shape))


# Charger l'image comme matrice de pixels
img = np.array(Image.open(
    r"C:\Users\USER\Desktop\pynas\imgs\arbresblurred.png"))

# Générer et afficher l'histogramme
# Pour le normaliser : argument density=True dans plt.hist
# Pour avoir l'histogramme cumulé : argument cumulative=True
n, bins, patches = plt.hist(img.flatten(), bins=range(256))
# plt.show()


# Charger l'image sous forme d'une matrice de pixels
img = np.array(Image.open(
    r"C:\Users\USER\Desktop\pynas\imgs\arbresblurred.png"))
# Générer le bruit gaussien de moyenne nulle et d'écart-type 7 (variance 49)
noise = np.random.normal(0, 7, img.shape)

# Assurez-vous que les valeurs de l'image et du bruit sont des entiers entre 0 et 255
img = np.clip(img, 0, 255).astype(np.uint8)
noise = np.clip(noise, 0, 255).astype(np.uint8)

# Ajouter le bruit à l'image
noisy_img = img + noise

# Assurez-vous que les valeurs de l'image bruitée restent dans la plage 0-255
noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

# Créer l'image bruitée et l'afficher
val = Image.fromarray(noisy_img)

# val.show()

# Convertir le tableau NumPy en objet Image
noisy_img_pil = Image.fromarray(noisy_img)

# Appliquer le lissage par moyennage (fenêtre de taille 9) et afficher le résultat
noisy_img_blurred = noisy_img_pil.filter(ImageFilter.BoxBlur(1))
noisy_img_blurred.show()
