from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow, get_cmap
import matplotlib.pyplot as plt

# Open the image from the working directory
image = Image.open("gr_cathedrale.png")

# Convert the image into a np.array
X = np.asarray(image)

# Print the information of the data
# print("Format : ", X.shape)
# print("Nombre de nuances de gris : ", X.max())


# Affiche l'image seule
def affiche_image(image):
    imshow(X, cmap=get_cmap("gray"))


# Definition d'une fonction qui affiche 2 imges cote a cote
def affiche_deux_images(img1, img2):
    _, axes = plt.subplots(ncols=2)
    axes[0].imshow(img1, cmap=plt.get_cmap("gray"))
    axes[1].imshow(img2, cmap=plt.get_cmap("gray"))


# Definition d'une fonction qui affiche 3 images cote a cote
def affiche_trois_images(img1, img2, img3):
    _, axes = plt.subplots(ncols=3)
    axes[0].imshow(img1, cmap=plt.get_cmap("gray"))
    axes[1].imshow(img2, cmap=plt.get_cmap("gray"))
    axes[2].imshow(img3, cmap=plt.get_cmap("gray"))


affiche_image(X)

# %% Exercice 1 : Pooling : Max, Moyen et Median


def get_block_values(X: np.array, ratio_x: int, ratio_y: int) -> np.array:
    """Return a np.array containing block divisions (shape(ratio_x, ratio_y)) of the
    initial matrix X

    Args:
        X (np.array): matrix to divide in blocks shape(Dx, Dy)
        ratio_x (int): row ratio of block division
        ratio_y (int): column ratio of block division

    Returns:
        np.array: the block divisions of X
    """
    # Ensure float type to use np.nan
    X = X.astype(float)

    l, c = X.shape  # Original dimensions
    # Compute the number of rows/columns to complete
    remainder_l = l % ratio_x
    remainder_c = c % ratio_y
    new_l = l if remainder_l == 0 else l + (ratio_x - remainder_l)
    new_c = c if remainder_c == 0 else c + (ratio_y - remainder_c)
    # Create matrix filled with np.nan (value that can be easily ignored later)
    X_padded = np.full((new_l, new_c), np.nan, dtype=float)
    X_padded[:l, :c] = X
    # Divide into blocks of size (ratio_x, ratio_y)
    blocks = X_padded.reshape(
        new_l // ratio_x, ratio_x, new_c // ratio_y, ratio_y
    ).swapaxes(1, 2)
    return blocks


# For the following functions,
# axis = (2,3) since the array has the form (n_blocks_rows, n_blocks_columns, ratio_x, ratio_y)


def pooling_max(X: np.array, ratio_x: int, ratio_y: int) -> np.array:
    """Pooling choosing the maximum value of the block area

    Args:
        X (np.array): the initial matrix to pool shape(Dx, Dy)
        ratio_x (int): the x ratio of pooling
        ratio_y (int): the y ratio of pooling

    Returns:
        np.array: a pooled matrix that has (Dx/ratio_x, Dy/ratio_y)
    """
    blocks = get_block_values(X, ratio_x, ratio_y)
    # Compute max while ignoring np.nan
    Y = np.nanmax(blocks, axis=(2, 3))
    return Y


def pooling_mean(X: np.array, ratio_x: int, ratio_y: int) -> np.array:
    """Pooling choosing the average value of the block area

    Args:
        X (np.array): the initial matrix to pool shape(Dx, Dy)
        ratio_x (int): the x ratio of pooling
        ratio_y (int): the y ratio of pooling

    Returns:
        np.array: a pooled matrix that has (Dx/ratio_x, Dy/ratio_y)
    """
    blocks = get_block_values(X, ratio_x, ratio_y)
    # Compute mean while ignoring np.nan
    Y = np.nanmean(blocks, axis=(2, 3))
    return Y


def pooling_median(X: np.array, ratio_x: int, ratio_y: int) -> np.array:
    """Pooling choosing the median value of the block area (if even values: returns the superior value)

    Args:
        X (np.array): the initial matrix to pool shape(Dx, Dy)
        ratio_x (int): the x ratio of pooling
        ratio_y (int): the y ratio of pooling

    Returns:
        np.array: a pooled matrix that has (Dx/ratio_x, Dy/ratio_y)
    """
    blocks = get_block_values(X, ratio_x, ratio_y)
    # Compute median while ignoring np.nan and taking the superior value
    Y = np.nanquantile(blocks, 0.5, axis=(2, 3), method="higher")
    return Y


X_max = pooling_max(X, 120, 107)
X_mean = pooling_mean(X, 120, 107)
X_median = pooling_median(X, 120, 107)

# affiche_trois_images(X_max, X_mean, X_median)
# plt.show()


# %% Exercice 2 : Convolution


def convolution1D(X: list, F: list) -> list:
    """For a data X, make a 1 dimension convolution by the filter F

    Args:
        X (list): data to convolut (size: N)
        F (list): Filter (size: H)

    Returns:
        list: convolution output list (size: N - H )
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
    """For a data X, make a 1 dimension convolution by the reversed filter F

    Args:
        X (list): data to convolut (size: N)
        F (list): Filter (size: H)

    Returns:
        list: convolution output list (size: N - H + 1)
    """
    H, N = len(F), len(X)
    Z = []
    for i in range(0, N - H + 1):
        zi = 0
        for h in range(0, H):
            zi += X[i + h] * F[h]
        Z.append(zi)
    return Z


# Definitions des donnees
X_1 = [80, 0, 0, 0, 0, 0, 80]
X_2 = [60, 20, 10, 0, 10, 20, 60]
X_3 = [10, 20, 30, 40, 60, 70, 80]
# Definition des filtres
F_1 = [1, 2, 1]
F_1_norm = [0.25, 0.5, 0.25]
F_2 = [-1, 2, -1]
F_3 = [0, 1, 2]
F_3_inv = [2, 1, 0]
# Ces lignes permettent de tester les fonctions de convolutions et cross_correlation
# Les decommenter une fois que vos fonctions sont implementees
# Convolution avec F_1
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
# print("Comparaison entre convolution et cross_correlation sur filtres inverses")
# print("Convolution X_2*F_3 : ", convolution1D(X_2, F_3)) #[140, 50, 20, 10, 40]
# print("cross correlation X_2*F_3_inv : ", cross_correlation1D(X_2, F_3_inv)) #[140, 50, 20, 10, 40]


# *Interprétation des convolutions:
# *F_1: lissage du signal, version normalisé: on conserve l'amplitude globale.
# *F_2: passe haut: met en avant les transitions -> c'est pour ca qu'on a une valeur - au début pour F3 et a la fin une valeur positive lorsque le signal entrant devient haut
# *F_3: accentue les valeurs à gauche en leur donnant un poids plus grand
# *F_3_inv: accentue les valeurs à droite en leur donnat un poids plus grand


# * Relation entre convolution et cross_correlation: l'un applique le filtre dans un sens et l'autre dans l'autre sens
# %% Exercice 2 : Padding
def convolution1D_padding(X: list, F: list) -> list:
    """For a data X and a filter F, make a convolution that has the same number of element as X

    Args:
        X (list): data to convolut (size: N)
        F (list): filter (size: H)

    Returns:
        list: convolution output (size: N)
    """
    N, H = len(X), len(F)
    num_zeros = (H - 1) // 2  # the number of zeros correspond to the round part of H/2
    X_padded = [0] * num_zeros + X + [0] * num_zeros
    return convolution1D(X_padded, F)


F_1 = [1, 2, 1]
X_1 = [80, 0, 0, 0, 0, 0, 80]
print(
    "Convolution X_1*F_1 : ", convolution1D_padding(X_1, F_1)
)  # [160, 80, 0, 0, 0, 80, 160]


# %% Exercice 2 : Stride
def convolution1D_stride(X: list, F: list, k: int) -> list:
    """For a data X and a filter F, make a k-Stride convolution -> a convolution with a k jump each iterations

    Args:
        X (list): data to convolut (size: N)
        F (list): filter to apply (size: H)
        k (int): Stride factor -> filter jump on iteration

    Returns:
        list: stride-convolution output (size: E((N - H)/k))
    """
    N, H = len(X), len(F)
    Z = []
    for i in range(
        (N - H) // k + 1
    ):  # It's ot specified if the len as to be correct for all X, F and k be using //, we assure that it steel correct
        zi = 0
        for h in range(H):
            zi += X[i * k + (H - 1) - h] * F[h]
        Z.append(zi)
    return Z


F_1 = [1, 2, 1]
X_1 = [80, 0, 0, 10, 0, 0, 1, 0, 0, 10, 0, 0, 80]
print("Convolution X_1*F_1 : ", convolution1D_stride(X_1, F_1, 2))  # [80, 20, 1, 1, 20]


# %% Exercice 2 : Convolution 2D

def cross_correlation2D(X,F):
    Dx, Dy = X.shape
    Hx, Hy, = F.shape
    
    liste_return = []
    for i in range(Dx-Hx+1):
        ligne = []
        for j in range(Dy-Hy+1):
            
            somme = 0
            
            for ip in range(Hx):
                for jp in range(Hy):
                    
                    somme +=X[i+ip][j+jp]*F[ip][jp]
                    
            ligne.append(somme)
            
        liste_return.append(ligne)
    
    return liste_return

def applique_filtre(X_old,F):
    X_new = cross_correlation2D(X_old,F)
    affiche_deux_images(X_old, X_new)
    return
    

# %% Filtres à tester sur l'image X_pool qui est obtenue par pooling sur l'image
### X originale

X = np.asarray(image)
#X_pool = pooling_max(X, 120, 107)
X_pool = pooling_mean(X, 120, 107)
#X_pool = pooling_median(X,120,107)



#%% Filtre 1
# Ce filtre remplace chaque pixel par la moyenne de son voisinage sur une plage 5 par 5
# Ducoup ca devient flou
# meme si la somme des termes de filtre_1 ne vaut pas 1, il n'y a pourtant pas d'assombrissement car matplotlib normalise les valeurs de l'image
s = 5
filtre_1 = np.ones((s, s)) / 100
print(filtre_1)

print("Filtre 1")
applique_filtre(X_pool, filtre_1)
plt.show()

#%% Filtre 2
# Comme le filtre d'avant, il remplace chaque pixel par la moyenne de son voisinage mais ici, le centre a une plus grande valeur comme
# la fonction gaussienne en probabilité, ducoup le flou preserve mieux les contours
filtre_2 = np.array(
    [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]]
)

print("Filtre 2")
applique_filtre(X_pool, filtre_2)
plt.show()

#%% Filtre 3
# Ici il y a des valeurs négatives et positives. Ce qui va mettre en avant les variations verticales de l'image en les assombrissant
# Dans l'exemple de l'image les plus grandes variations verticales sont celles des toits, qui sont bien noir dans la representation
filtre_3 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

print("Filtre 3")
applique_filtre(X_pool, filtre_3)
plt.show()

#%% Filtre 4
# Ici il y a des valeurs négatives et positives mais dans les colonnes a la place des lignes
# Ce qui va mettre en avant les variations horizontales de l'image en les assombrissant
# Dans l'exemple de l'image les plus grandes variations horizontales sont celles des murs, qui sont bien noir dans la representation
# Ici les murs assombris sont ceux de droites, et les murs clairs sont ceux de droites car les valeurs positives dans la colonne de gauche, 
# Ducoup quand on passe le filtre de gauche a droite, de haut en bas, on commencera toujours a gauche et la cathedrale, etant plus sombre
# si on est a sa gauche a cause des valeurs negatives, ce filtre rendre les points sur les murs blanc ( donc une valeur plus petite)
# Et a droite c'est l'inverse ducoup ils deviennent noirs

filtre_4 = np.array([[2, 0, -2], [4, 0, -4], [2, 0, -2]])

print("Filtre 4")
applique_filtre(X_pool, filtre_4)
plt.show()

#%% Filtre 5 
# Fonctionne environ comme le filtre 3 sauf que la le filtre est beaucoup plus brutal qu'avant car il prend seulement 2 pixels en compte sur la meme ligne
# horizontale compare à 3 lignes horizontales précédemment
filtre_5 = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])

print("Filtre 5 : Dérivée horizontale simple")
applique_filtre(X_pool, filtre_5)
plt.show()

#%% Filtre 5.bis
# la valeur du centre est négative et entouré de valeurs positives
# Cela va donc accentuer les contours ce qui dans le cas de la cathedrale permet de la separer de l'arriere plan ( le ciel)
filtre_5bis = np.array([[0, 1, 0], [1, -200, 1], [0, 1, 0]])

print("Filtre 5.bis : Renforcement des contours (Laplacien modifié)")
applique_filtre(X_pool, filtre_5bis)
plt.show()

#%% Filtre 6
# la valeur du centre est négative et entouré de valeurs positives, mais compare a avant on prend les diagonales en comptes
# Cela va donc accentuer les contours ce qui dans le cas de la cathedrale permet de la separer de l'arriere plan ( le ciel)
# Mais va rendre le tout un peu plus flou que précédemment
filtre_6 = np.array([[1, 1, 1], [1, -200, 1], [1, 1, 1]])

print("Filtre 6")
applique_filtre(X_pool, filtre_6)
plt.show()

#%% Filtre 7
# Ici à la place d'accentuer les contours on va accentuer les centres, ainsi Les détails de l'image ressortent beaucoup plus que 
# Précédemment

filtre_7 = np.array([[0, -1, 0], [-1, 10, -1], [0, -1, 0]])

print("Filtre 7")
applique_filtre(X_pool, filtre_7)
plt.show()

#%% Filtre 8
# Même principe d'accentuation, sauf que la on prend les diagonales en compte ce qui va encore plus accentuer les pixels
filtre_8 = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])

print("Filtre 8")
applique_filtre(X_pool, filtre_8)
plt.show()

#%% Filtre 9 =
# Fait une accentuation en forme de croix et va prendre plus de valeurs que le filtre 7, il va cherche plus loing
filtre_9 = np.array(
    [
        [0, 0, -1, 0, 0],
        [0, 0, -1, 0, 0],
        [-1, -1, 10, -1, -1],
        [0, 0, -1, 0, 0],
        [0, 0, -1, 0, 0],
    ]
)

print("Filtre 9 : Détection de contours en croix")
applique_filtre(X_pool, filtre_9)
plt.show()

#%%Question Finale

"""
Reponse ChatGPT faut revoir
Dans un réseau de neurones classique, chaque couche est généralement définie par une transformation affine suivie d'une activation :

Z(i+1) = \sigma(W(i)Z(i) + b(i))

Ici, la matrice de poids W(i) et le biais b(i) représentent des paramètres à apprendre. Cependant, dans l'architecture donnée par l'équation
Z(i+1) = \sigma(Z(i) * F(i)), les poids sont directement intégrés sous forme d'un filtre F(i) qui est souvent de taille réduite, notamment dans 
le cadre des réseaux convolutionnels. Cela permet une forte diminution du nombre total de paramètres du réseau, ce qui réduit la complexité du 
modèle et les besoins en mémoire.

"""