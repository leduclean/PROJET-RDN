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


# %% Filtres à tester sur l'image X_pool qui est obtenue par pooling sur l'image
### X originale

s = 5
filtre_1 = np.ones((s, s)) / 100

filtre_2 = np.array(
    [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]]
)

filtre_3 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

filtre_4 = np.array([[2, 0, -2], [4, 0, -4], [2, 0, -2]])

filtre_5 = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])

# Faire varier la valeur centrale entre 0 et -200
filtre_5 = np.array([[0, 1, 0], [1, -200, 1], [0, 1, 0]])

filtre_6 = np.array([[1, 1, 1], [1, -200, 1], [1, 1, 1]])


# Faire varier la valeur centrale entre 0 et 200
filtre_7 = np.array([[0, -1, 0], [-1, 10, -1], [0, -1, 0]])

filtre_8 = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])


Filtre_9 = np.array(
    [
        [0, 0, -1, 0, 0],
        [0, 0, -1, 0, 0],
        [-1, -1, 10, -1, -1],
        [0, 0, -1, 0, 0],
        [0, 0, -1, 0, 0],
    ]
)
