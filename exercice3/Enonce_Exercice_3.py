from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow, get_cmap
import matplotlib.pyplot as plt

# Open the image from the working directory
image = Image.open('gr_cathedrale.png')

# Convert the image into a np.array
X = np.asarray(image)

# Print the information of the data
print("Format : ", X.shape)
print("Nombre de nuances de gris : ", X.max())

# Affiche l'image seule
def affiche_image(image):
    imshow(X,cmap=get_cmap('gray'))

# Definition d'une fonction qui affiche 2 imges cote a cote
def affiche_deux_images(img1, img2):
  _, axes = plt.subplots(ncols=2)
  axes[0].imshow(img1, cmap=plt.get_cmap('gray'))
  axes[1].imshow(img2, cmap=plt.get_cmap('gray'))

# Definition d'une fonction qui affiche 3 images cote a cote
def affiche_trois_images(img1, img2, img3):
  _, axes = plt.subplots(ncols=3)
  axes[0].imshow(img1, cmap=plt.get_cmap('gray'))
  axes[1].imshow(img2, cmap=plt.get_cmap('gray'))
  axes[2].imshow(img3, cmap=plt.get_cmap('gray'))


affiche_image(X)

#%% Exercice 1 : Pooling : Max, Moyen et Median

def pooling_max(X,ratio_x,ratio_y):
  
  return

#%% Exercice 2 : Convolution
# Definitions des donnees
X_1 = [80,0,0,0,0,0,80]
X_2 = [60,20,10,0,10,20,60]
X_3 = [10,20,30,40,60,70,80]
# Definition des filtres
F_1 = [1,2,1]
F_1_norm = [0.25,0.5,0.25]
F_2 = [-1,2,-1]
F_3 = [0,1,2]
F_3_inv = [2,1,0]

# Ces lignes permettent de tester les fonctions de convolutions et cross_correlation
# Les decommenter une fois que vos fonctions sont implementees
# # Convolution avec F_1
# print("Convolution avec F_1 = [1,2,1] et F_1_norm = [0.25,0.5,0.25] :")
# print("Convolution X_1*F_1 : ", convolution1D(X_1, F_1)) #[80, 0, 0, 0, 80]
# print("Convolution X_1*F_1_norm : ", convolution1D(X_1, F_1_norm)) # [20.0, 0.0, 0.0, 0.0, 20.0]
# print("Convolution X_2*F_1 : ", convolution1D(X_2, F_1)) #[110, 40, 20, 40, 110]
# print("Convolution X_2*F_1_norm : ", convolution1D(X_2, F_1_norm)) #[27.5, 10.0, 5.0, 10.0, 27.5]
# print("Convolution X_3*F_1 : ", convolution1D(X_3, F_1)) #[80, 120, 170, 230, 280]
# print("Convolution X_3*F_1_norm : ", convolution1D(X_3, F_1_norm),'\n') #[20.0, 30.0, 42.5, 57.5, 70.0]

# # Convolution avec F_2
# print("Convolution avec F_2 = [-1,2,-1]") #[-1,2,-1]
# print("Convolution X_1*F_2 : ", convolution1D(X_1, F_2)) #[-80, 0, 0, 0, -80]
# print("Convolution X_2*F_2 : ", convolution1D(X_2, F_2)) #[-30, 0, -20, 0, -30]
# print("Convolution X_3*F_2 : ", convolution1D(X_3, F_2),'\n') #[0, 0, -10, 10, 0] 

# # Convolution avec F_3
# print("Convolution avec F_3 = [0,1,2]")
# print("Convolution X_1*F_3 : ", convolution1D(X_1, F_3)) #[160, 0, 0, 0, 0]
# print("Convolution X_2*F_3 : ", convolution1D(X_2, F_3)) #[140, 50, 20, 10, 40]
# print("Convolution X_3*F_3 : ", convolution1D(X_3, F_3)) #[40, 70, 100, 140, 190]
# print("Convolution X_3*F_3_inv : ", convolution1D(X_3, F_3_inv),'\n') #[80, 110, 160, 200, 230]

# # Comparaison entre convolution et cross_correlation
# print("Comparaison entre convolution et convolution sur filtres inverses")
# print("Convolution X_2*F_3 : ", convolution1D(X_2, F_3)) #[140, 50, 20, 10, 40]
# print("Convolution X_2*F_3_inv : ", cross_correlation1D(X_2, F_3_inv)) #[140, 50, 20, 10, 40]

#%% Exercice 2 : Padding


F_1 = [1,2,1]
X_1 = [80,0,0,0,0,0,80]
# print("Convolution X_1*F_1 : ", convolution1D_padding(X_1, F_1)) #[160, 80, 0, 0, 0, 80, 160]



#%% Exercice 2 : Stride

F_1 = [1,2,1]
X_1 = [80,0,0,10,0,0,1,0,0,10,0,0,80]
# print("Convolution X_1*F_1 : ", convolution1D_stride(X_1, F_1,2)) #[80, 20, 1, 1, 20]


#%% Exercice 2 : Convolution 2D


#%% Filtres à tester sur l'image X_pool qui est obtenue par pooling sur l'image
### X originale

s = 5
filtre_1 = np.ones((s,s))/100

filtre_2 = np.array([[0.0625, 0.125, 0.0625],
                     [0.125, 0.25, 0.125],
                     [0.0625, 0.125, 0.0625]])

filtre_3 = np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]])

filtre_4 = np.array([[2, 0, -2],
                     [4, 0, -4],
                     [2, 0, -2]])

filtre_5 = np.array([[0, 0, 0],
                    [-1, 1, 0],
                    [0, 0, 0]])

# Faire varier la valeur centrale entre 0 et -200
filtre_5 = np.array([[0, 1, 0],
                     [1, -200, 1],
                     [0, 1, 0]])

filtre_6 = np.array([[1, 1, 1],
                     [1, -200, 1],
                     [1, 1, 1]])


# Faire varier la valeur centrale entre 0 et 200
filtre_7 = np.array([[0, -1, 0],
                     [-1, 10, -1],
                     [0, -1, 0]])

filtre_8 = np.array([[-1, -1, -1],
                     [-1, 10, -1],
                     [-1, -1, -1]])


Filtre_9 = np.array([[0, 0, -1, 0, 0],
                     [0, 0, -1, 0, 0],
                     [-1, -1, 10, -1, -1],
                     [0, 0, -1, 0, 0],
                     [0, 0, -1, 0, 0]])


