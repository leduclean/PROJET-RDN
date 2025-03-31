from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow, get_cmap
import matplotlib.pyplot as plt

# Ouvrir l'image depuis le répertoire de travail
image = Image.open("gr_cathedrale.png")

# Convertir l'image en un np.array
X = np.asarray(image)

# Affiche les informations sur les données
# print("Format : ", X.shape)
# print("Nombre de nuances de gris : ", X.max())


# Affiche l'image seule
def affiche_image(img: np.array) -> None:
    """
    Affiche une image en niveaux de gris.

    Paramètres:
        img (np.array): Image à afficher.
    """
    imshow(img, cmap=get_cmap("gray"))


# Définition d'une fonction qui affiche 2 images côte à côte
def affiche_deux_images(img1: np.array, img2: np.array) -> None:
    """
    Affiche deux images côte à côte.

    Paramètres:
        img1 (np.array): Première image.
        img2 (np.array): Deuxième image.
    """
    _, axes = plt.subplots(ncols=2)
    axes[0].imshow(img1, cmap=plt.get_cmap("gray"))
    axes[1].imshow(img2, cmap=plt.get_cmap("gray"))
    plt.show()


# Définition d'une fonction qui affiche 3 images côte à côte
def affiche_trois_images(img1: np.array, img2: np.array, img3: np.array) -> None:
    """
    Affiche trois images côte à côte.

    Paramètres:
        img1 (np.array): Première image.
        img2 (np.array): Deuxième image.
        img3 (np.array): Troisième image.
    """
    _, axes = plt.subplots(ncols=3)
    axes[0].imshow(img1, cmap=plt.get_cmap("gray"))
    axes[1].imshow(img2, cmap=plt.get_cmap("gray"))
    axes[2].imshow(img3, cmap=plt.get_cmap("gray"))
    plt.show()


# Affichage de l'image originale
affiche_image(X)


# %% Exercice 1 : Pooling : Max, Moyen et Médian

def get_block_values(X: np.array, ratio_x: int, ratio_y: int) -> np.array:
    """
    Divise la matrice initiale X en blocs de taille (ratio_x, ratio_y).

    Paramètres:
        X (np.array): Matrice à diviser de forme (Dx, Dy).
        ratio_x (int): Ratio en lignes pour la division en blocs.
        ratio_y (int): Ratio en colonnes pour la division en blocs.

    Retourne:
        np.array: Tableau contenant les blocs de X.
    """
    # Conversion en type float pour utiliser np.nan
    X = X.astype(float)

    l, c = X.shape  # Dimensions originales
    # Calculer le nombre de lignes/colonnes à compléter
    reste_l = l % ratio_x
    reste_c = c % ratio_y
    new_l = l if reste_l == 0 else l + (ratio_x - reste_l)
    new_c = c if reste_c == 0 else c + (ratio_y - reste_c)
    # Créer une matrice remplie de np.nan (valeur facilement ignorée par la suite)
    X_pad = np.full((new_l, new_c), np.nan, dtype=float)
    X_pad[:l, :c] = X
    # Diviser en blocs de taille (ratio_x, ratio_y)
    blocs = X_pad.reshape(new_l // ratio_x, ratio_x, new_c // ratio_y, ratio_y).swapaxes(1, 2)
    return blocs


def pooling_max(X: np.array, ratio_x: int, ratio_y: int) -> np.array:
    """
    Applique un pooling en prenant la valeur maximale de chaque bloc.

    Paramètres:
        X (np.array): Matrice initiale à réduire de forme (Dx, Dy).
        ratio_x (int): Ratio en x pour le pooling.
        ratio_y (int): Ratio en y pour le pooling.

    Retourne:
        np.array: Matrice réduite de forme (Dx/ratio_x, Dy/ratio_y).
    """
    blocs = get_block_values(X, ratio_x, ratio_y)
    # Calcul du maximum en ignorant les np.nan
    Y = np.nanmax(blocs, axis=(2, 3))
    return Y


def pooling_mean(X: np.array, ratio_x: int, ratio_y: int) -> np.array:
    """
    Applique un pooling en prenant la valeur moyenne de chaque bloc.

    Paramètres:
        X (np.array): Matrice initiale à réduire de forme (Dx, Dy).
        ratio_x (int): Ratio en x pour le pooling.
        ratio_y (int): Ratio en y pour le pooling.

    Retourne:
        np.array: Matrice réduite de forme (Dx/ratio_x, Dy/ratio_y).
    """
    blocs = get_block_values(X, ratio_x, ratio_y)
    # Calcul de la moyenne en ignorant les np.nan
    Y = np.nanmean(blocs, axis=(2, 3))
    return Y


def pooling_median(X: np.array, ratio_x: int, ratio_y: int) -> np.array:
    """
    Applique un pooling en prenant la valeur médiane de chaque bloc.
    (Si le nombre de valeurs est pair, retourne la valeur supérieure.)

    Paramètres:
        X (np.array): Matrice initiale à réduire de forme (Dx, Dy).
        ratio_x (int): Ratio en x pour le pooling.
        ratio_y (int): Ratio en y pour le pooling.

    Retourne:
        np.array: Matrice réduite de forme (Dx/ratio_x, Dy/ratio_y).
    """
    blocs = get_block_values(X, ratio_x, ratio_y)
    # Calcul de la médiane en ignorant les np.nan et en prenant la valeur supérieure
    Y = np.nanquantile(blocs, 0.5, axis=(2, 3), method="higher")
    return Y


# Calcul des différentes versions de pooling sur l'image
X_max = pooling_max(X, 120, 107)
X_mean = pooling_mean(X, 120, 107)
X_median = pooling_median(X, 120, 107)

# Pour afficher les trois images côte à côte, décommentez la ligne suivante :
# affiche_trois_images(X_max, X_mean, X_median)
# plt.show()


# %% Exercice 2 : Convolution

def convolution1D(X: list, F: list) -> list:
    """
    Effectue une convolution 1D sur les données X à l'aide du filtre F.

    Paramètres:
        X (list): Données à convoluer (taille : N).
        F (list): Filtre à appliquer (taille : H).

    Retourne:
        list: Liste résultante de la convolution (taille : N - H + 1).
    """
    H = len(F)
    N = len(X)
    Z = []
    for i in range(0, N - H + 1):
        zi = 0
        for h in range(0, H):
            zi += X[i + H - (h + 1)] * F[h]
        Z.append(zi)
    return Z


def cross_correlation1D(X: list, F: list) -> list:
    """
    Effectue une corrélation croisée 1D sur les données X à l'aide du filtre F (non inversé).

    Paramètres:
        X (list): Données à traiter (taille : N).
        F (list): Filtre à appliquer (taille : H).

    Retourne:
        list: Liste résultante de la corrélation (taille : N - H + 1).
    """
    H, N = len(F), len(X)
    Z = []
    for i in range(0, N - H + 1):
        zi = 0
        for h in range(0, H):
            zi += X[i + h] * F[h]
        Z.append(zi)
    return Z


# Définitions des données
X_1 = [80, 0, 0, 0, 0, 0, 80]
X_2 = [60, 20, 10, 0, 10, 20, 60]
X_3 = [10, 20, 30, 40, 60, 70, 80]
# Définitions des filtres
F_1 = [1, 2, 1]
F_1_norm = [0.25, 0.5, 0.25]
F_2 = [-1, 2, -1]
F_3 = [0, 1, 2]
F_3_inv = [2, 1, 0]

# Les lignes suivantes permettent de tester les fonctions de convolution et de corrélation croisée.
# Décommentez-les une fois que vos fonctions sont implémentées.
# Exemple de convolution avec F_1 :
# print("Convolution avec F_1 = [1, 2, 1] et F_1_norm = [0.25, 0.5, 0.25] :")
# print("Convolution X_1 * F_1 : ", convolution1D(X_1, F_1))         # [80, 0, 0, 0, 80]
# print("Convolution X_1 * F_1_norm : ", convolution1D(X_1, F_1_norm))   # [20.0, 0.0, 0.0, 0.0, 20.0]
# print("Convolution X_2 * F_1 : ", convolution1D(X_2, F_1))         # [110, 40, 20, 40, 110]
# print("Convolution X_2 * F_1_norm : ", convolution1D(X_2, F_1_norm))   # [27.5, 10.0, 5.0, 10.0, 27.5]
# print("Convolution X_3 * F_1 : ", convolution1D(X_3, F_1))         # [80, 120, 170, 230, 280]
# print("Convolution X_3 * F_1_norm : ", convolution1D(X_3, F_1_norm), '\n')  # [20.0, 30.0, 42.5, 57.5, 70.0]

# Exemple de convolution avec F_2 :
# print("Convolution avec F_2 = [-1, 2, -1]")
# print("Convolution X_1 * F_2 : ", convolution1D(X_1, F_2))         # [-80, 0, 0, 0, -80]
# print("Convolution X_2 * F_2 : ", convolution1D(X_2, F_2))         # [-30, 0, -20, 0, -30]
# print("Convolution X_3 * F_2 : ", convolution1D(X_3, F_2), '\n')    # [0, 0, -10, 10, 0]

# Exemple de convolution avec F_3 :
# print("Convolution avec F_3 = [0, 1, 2]")
# print("Convolution X_1 * F_3 : ", convolution1D(X_1, F_3))         # [160, 0, 0, 0, 0]
# print("Convolution X_2 * F_3 : ", convolution1D(X_2, F_3))         # [140, 50, 20, 10, 40]
# print("Convolution X_3 * F_3 : ", convolution1D(X_3, F_3))         # [40, 70, 100, 140, 190]
# print("Convolution X_3 * F_3_inv : ", convolution1D(X_3, F_3_inv), '\n')  # [80, 110, 160, 200, 230]

# Comparaison entre convolution et corrélation croisée avec filtres inversés :
# print("Comparaison entre convolution et corrélation croisée sur filtres inversés")
# print("Convolution X_2 * F_3 : ", convolution1D(X_2, F_3))          # [140, 50, 20, 10, 40]
# print("Corrélation croisée X_2 * F_3_inv : ", cross_correlation1D(X_2, F_3_inv))  # [140, 50, 20, 10, 40]


# Interprétation des convolutions :
# * F_1 : lissage du signal, version normalisée (on conserve l'amplitude globale).
# * F_2 : passe-haut, qui met en avant les transitions (d'où une valeur négative en début et une valeur positive en fin lorsque le signal augmente).
# * F_3 : accentue les valeurs à gauche en leur attribuant un poids plus important.
# * F_3_inv : accentue les valeurs à droite en leur attribuant un poids plus important.
# Relation entre convolution et corrélation croisée : l'une applique le filtre dans un sens, l'autre dans le sens inverse.


# %% Exercice 2 : Padding
def convolution1D_padding(X: list, F: list) -> list:
    """
    Effectue une convolution 1D avec padding afin que la sortie ait le même nombre d'éléments que X.

    Paramètres:
        X (list): Données à convoluer (taille : N).
        F (list): Filtre à appliquer (taille : H).

    Retourne:
        list: Résultat de la convolution (taille : N).
    """
    N, H = len(X), len(F)
    nb_zeros = (H - 1) // 2  # Nombre de zéros correspondant à la partie arrondie de H/2
    X_pad = [0] * nb_zeros + X + [0] * nb_zeros
    return convolution1D(X_pad, F)


F_1 = [1, 2, 1]
X_1 = [80, 0, 0, 0, 0, 0, 80]
print("Convolution X_1 * F_1 (avec padding) : ", convolution1D_padding(X_1, F_1))
# Résultat attendu : [160, 80, 0, 0, 0, 80, 160]


# %% Exercice 2 : Stride
def convolution1D_stride(X: list, F: list, k: int) -> list:
    """
    Effectue une convolution 1D avec un pas (stride) k.

    Paramètres:
        X (list): Données à convoluer (taille : N).
        F (list): Filtre à appliquer (taille : H).
        k (int): Facteur de stride (décalage entre chaque application du filtre).

    Retourne:
        list: Résultat de la convolution avec stride (taille approximative : (N - H) // k + 1).
    """
    N, H = len(X), len(F)
    Z = []
    for i in range((N - H) // k + 1):
        zi = 0
        for h in range(H):
            zi += X[i * k + (H - 1) - h] * F[h]
        Z.append(zi)
    return Z


F_1 = [1, 2, 1]
X_1 = [80, 0, 0, 10, 0, 0, 1, 0, 0, 10, 0, 0, 80]
print("Convolution X_1 * F_1 (avec stride de 2) : ", convolution1D_stride(X_1, F_1, 2))
# Résultat attendu : [80, 20, 1, 1, 20]


# %% Exercice 2 : Convolution 2D

def cross_correlation2D(X: np.array, F: np.array) -> list:
    """
    Effectue une corrélation croisée 2D entre une image et un filtre.

    Paramètres:
        X (np.array): Image d'entrée (matrice 2D).
        F (np.array): Filtre à appliquer (matrice 2D).

    Retourne:
        list: Résultat de la corrélation croisée sous forme de liste (matrice 2D).
    """
    Dx, Dy = X.shape
    Hx, Hy = F.shape
    
    resultat = []
    for i in range(Dx - Hx + 1):
        ligne = []
        for j in range(Dy - Hy + 1):
            somme = 0
            for ip in range(Hx):
                for jp in range(Hy):
                    somme += X[i + ip][j + jp] * F[ip][jp]
            ligne.append(somme)
        resultat.append(ligne)
    
    return resultat


def applique_filtre(X_original: np.array, F: np.array) -> None:
    """
    Applique un filtre 2D à une image via corrélation croisée et affiche l'image originale à côté de l'image filtrée.

    Paramètres:
        X_original (np.array): Image originale.
        F (np.array): Filtre à appliquer.
    """
    X_filtre = cross_correlation2D(X_original, F)
    affiche_deux_images(X_original, np.array(X_filtre))


# %% Filtres à tester sur l'image X_pool obtenue par pooling sur l'image
# Image originale
X = np.asarray(image)
# On peut choisir l'une des méthodes de pooling :
# X_pool = pooling_max(X, 120, 107)
X_pool = pooling_mean(X, 120, 107)
# X_pool = pooling_median(X, 120, 107)


# %% Filtre 1
# Ce filtre remplace chaque pixel par la moyenne de son voisinage sur une plage 5x5.
# Le résultat est un flou (même si la somme des termes du filtre ne vaut pas 1, matplotlib normalise les valeurs).
s = 5
filtre_1 = np.ones((s, s)) / 100
print(filtre_1)

print("Filtre 1")
applique_filtre(X_pool, filtre_1)
plt.show()


# %% Filtre 2
# Ce filtre remplace chaque pixel par la moyenne de son voisinage, avec un poids plus élevé au centre,
# imitant une fonction gaussienne pour préserver les contours.
filtre_2 = np.array([[0.0625, 0.125, 0.0625],
                     [0.125,  0.25,  0.125],
                     [0.0625, 0.125, 0.0625]])
print("Filtre 2")
applique_filtre(X_pool, filtre_2)
plt.show()


# %% Filtre 3
# Ce filtre présente des valeurs négatives et positives, accentuant les variations verticales en assombrissant l'image.
filtre_3 = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]])
print("Filtre 3")
applique_filtre(X_pool, filtre_3)
plt.show()


# %% Filtre 4
# Ce filtre accentue les variations horizontales en modifiant les colonnes plutôt que les lignes.
filtre_4 = np.array([[ 2,  0, -2],
                     [ 4,  0, -4],
                     [ 2,  0, -2]])
print("Filtre 4")
applique_filtre(X_pool, filtre_4)
plt.show()


# %% Filtre 5
# Ce filtre, similaire au filtre 3, est plus brutal car il ne prend en compte que 2 pixels sur la même ligne horizontale.
filtre_5 = np.array([[0, 0, 0],
                     [-1, 1, 0],
                     [0, 0, 0]])
print("Filtre 5 : Dérivée horizontale simple")
applique_filtre(X_pool, filtre_5)
plt.show()


# %% Filtre 5.bis
# Le centre est négatif entouré de valeurs positives, accentuant les contours pour mieux séparer la cathédrale de l'arrière-plan.
filtre_5bis = np.array([[0, 1, 0],
                        [1, -200, 1],
                        [0, 1, 0]])
print("Filtre 5.bis : Renforcement des contours (Laplacien modifié)")
applique_filtre(X_pool, filtre_5bis)
plt.show()


# %% Filtre 6
# Ce filtre prend en compte également les diagonales, accentuant les contours mais avec un léger flou supplémentaire.
filtre_6 = np.array([[ 1,  1,  1],
                     [ 1, -200, 1],
                     [ 1,  1,  1]])
print("Filtre 6")
applique_filtre(X_pool, filtre_6)
plt.show()


# %% Filtre 7
# Ce filtre accentue les centres, faisant ressortir davantage les détails de l'image.
filtre_7 = np.array([[ 0, -1,  0],
                     [-1, 10, -1],
                     [ 0, -1,  0]])
print("Filtre 7")
applique_filtre(X_pool, filtre_7)
plt.show()


# %% Filtre 8
# Même principe d'accentuation en prenant en compte les diagonales, pour accentuer encore plus les pixels.
filtre_8 = np.array([[-1, -1, -1],
                     [-1, 10, -1],
                     [-1, -1, -1]])
print("Filtre 8")
applique_filtre(X_pool, filtre_8)
plt.show()


# %% Filtre 9
# Ce filtre accentue en forme de croix, prenant en compte plus de valeurs que le filtre 7 pour détecter les contours sur une plus grande zone.
filtre_9 = np.array([[ 0,  0, -1,  0,  0],
                     [ 0,  0, -1,  0,  0],
                     [-1, -1, 10, -1, -1],
                     [ 0,  0, -1,  0,  0],
                     [ 0,  0, -1,  0,  0]])
print("Filtre 9 : Détection de contours en croix")
applique_filtre(X_pool, filtre_9)
plt.show()


# %% Question Finale

"""
Réponse (à revoir) :

Dans un réseau de neurones classique, chaque couche est généralement définie par une transformation affine suivie d'une activation :

    Z(i+1) = σ(W(i) * Z(i) + b(i))

Ici, la matrice de poids W(i) et le biais b(i) représentent des paramètres à apprendre. Cependant, dans l'architecture définie par

    Z(i+1) = σ(Z(i) * F(i))

les poids sont directement intégrés sous forme d'un filtre F(i), souvent de taille réduite, comme dans les réseaux convolutionnels.
Cela permet de réduire considérablement le nombre total de paramètres du réseau, diminuant ainsi la complexité du modèle et les besoins en mémoire.
"""
