# %% Introduction
# Import
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ajout au path du dossier actuel pour pouvoir utiliser Visualiseur (structure en package)
current_dir = os.path.dirname(os.path.abspath("exercice2/"))  # Dossier du fichier
project_root = os.path.dirname(current_dir)  # Dossier du projet (remonte d'un niveau)

if project_root not in sys.path:
    sys.path.append(project_root)

from visualisation.visualizer import Visualiseur

# Création d'une instance de la classe Visualizer pour les tracés
viz = Visualiseur()

# Question 1
# Si on cherche à minimiser J, alors on cherche à maximiser les probabilités y_{nk} ∈ [0,1] car ln(1) = 0 alors que si y_{nk} proche de 0,
# ln(y_{nk}) devient très petit.
# Ce qui veut dire qu'on cherche la plus grande probabilité de y_{nk}, donc la plus grande probabilité de réaliser une bonne classification.


# %% Import des données
def readdataset2d(fname):
    with open(fname, "r") as file:
        X, T = [], []
        for l in file:
            x = l.strip().split()
            X.append((float(x[0]), float(x[1])))
            T.append(int(x[2]))
        T = np.reshape(np.array(T), (-1, 1))
    return np.array(X), T


# %% Mise en place des fonctions pour la regression à plusieurs classes
def convertit(T: np.array, K: int) -> np.array:
    """
    Convertit un vecteur d'étiquettes en une matrice d'encodage one-hot.

    Args:
        T (np.array): Tableau des étiquettes de taille (N, 1) où T[i] = j avec j l'étiquette de classe.
        K (int): Nombre de classes.

    Returns:
        np.array: Tableau de taille (N, K) où new_T[i, j] = 1 si l'exemple i appartient à la classe j, sinon 0.
    """
    new_T = np.zeros((len(T), K))
    for i, classification in enumerate(T):
        new_T[i, classification] = 1
    return new_T


def softmax(A: np.array) -> np.array:
    """
    Applique la fonction softmax à chaque ligne d'un tableau pour obtenir des probabilités.

    Args:
        A (np.array): Tableau de taille (n, p) sur lequel appliquer la fonction softmax.

    Returns:
        np.array: Tableau de taille (n, p) où chaque ligne correspond à la distribution de probabilités softmax
                  de la ligne correspondante de A.
    """
    B = []
    for ligne in A:
        B.append(1 / np.sum(np.exp(ligne)) * np.exp(ligne))
    return np.array(B)


def predit_proba(X: np.array, W: np.array, b: np.array) -> np.array:
    """
    Calcule les probabilités de classes pour chaque exemple en utilisant la fonction softmax.

    Args:
        X (np.array): Tableau des données de taille (N, D), où N est le nombre d'exemples et D le nombre de caractéristiques.
        W (np.array): Matrice des paramètres de régression de taille (D, K), où K est le nombre de classes.
        b (np.array): Vecteur biais de taille (1, K).

    Returns:
        np.array: Tableau de taille (N, K) contenant les probabilités prédites pour chaque classe.
    """
    return softmax(X.dot(W) + b)


def predit_classe(Y: np.array, K: int) -> np.array:
    """
    Convertit les probabilités de classes en une prédiction one-hot en sélectionnant la classe ayant la probabilité maximale.

    Args:
        Y (np.array): Tableau des probabilités de taille (N, K).
        K (int): Nombre de classes.

    Returns:
        np.array: Matrice one-hot de taille (N, K) où chaque ligne a 1 à l'indice de la classe prédite.
    """
    # Obtention de l'indice de la classe avec la probabilité maximale pour chaque exemple
    prediction_index = np.argmax(Y, axis=1)
    C = convertit(prediction_index, K)
    return C


def initialise(D: int, K: int) -> tuple[np.array, np.array, np.array, np.array]:
    """
    Initialise les paramètres de la régression logistique.

    Args:
        D (int): Nombre de dimensions (caractéristiques).
        K (int): Nombre de classes.

    Returns:
        tuple:
            - np.array: Matrice W initialisée aléatoirement de taille (D, K).
            - np.array: Vecteur b initialisé aléatoirement de taille (1, K).
            - np.array: Tableau Y initial contenant les probabilités prédites.
            - np.array: Tableau C initial contenant les prédictions one-hot.
    """
    W = np.random.uniform(-2, 2, size=(D, K))
    b = np.random.randn(1, K)
    Y = predit_proba(X_train, W, b)
    C = predit_classe(Y, K)
    return W, b, Y, C


def cross_entropy(Y: np.array, T: np.array) -> float:
    """
    Calcule l'erreur de cross-entropie entre les prédictions et les étiquettes réelles.

    Args:
        Y (np.array): Tableau des probabilités prédites de taille (N, K).
        T (np.array): Tableau des étiquettes réelles one-hot de taille (N, K).

    Returns:
        float: Valeur de l'erreur de cross-entropie.
    """
    N, K = Y.shape
    J = 0
    for n in range(N):
        for k in range(K):
            if T[n, k] == 1:
                if np.log(Y[n, k]) == 0.0:
                    continue
                else:
                    J -= T[n, k] * np.log(Y[n, k])
    return J


def updateWb(
    W: np.array, b: np.array, X: np.array, Y: np.array, T: np.array, lr: float
) -> None:
    """
    Met à jour les poids et biais de la régression logistique en utilisant la descente de gradient.

    Args:
        W (np.array): Matrice des poids à mettre à jour.
        b (np.array): Vecteur des biais à mettre à jour.
        X (np.array): Données d'entrée de taille (N, D).
        Y (np.array): Prédictions actuelles de taille (N, K).
        T (np.array): Étiquettes réelles one-hot de taille (N, K).
        lr (float): Taux d'apprentissage.
    """
    W -= lr * (X.transpose()).dot((Y - T))
    for j in range(0, K):
        grad = 0
        for n in range(N):
            grad += np.sum(Y - T, axis=1)[n]
        b[0, j] -= lr * grad


def taux_precision(C: np.array, T: np.array) -> float:
    """
    Calcule le taux de précision en comparant les prédictions avec les étiquettes réelles.

    Args:
        C (np.array): Matrice des prédictions one-hot de taille (N, K).
        T (np.array): Matrice des étiquettes réelles one-hot de taille (N, K).

    Returns:
        float: Pourcentage de prédictions correctes.
    """
    N = T.shape[0]
    well_classified = 0
    for i in range(N):
        if np.argmax(C[i]) == np.argmax(T[i]):
            well_classified += 1
    return well_classified * 100 / N


# * Logistic Regression Function
def regression_logistique(
    W, b, X, Y, T, lr=0.01, nb_iter=1000, int_affiche=100, quiet=False
):
    """
    Entraîne un modèle de régression logistique à l'aide de la descente de gradient et affiche périodiquement
    l'erreur de cross-entropie et la précision.

    Args:
        W (np.array): Matrice des poids initiale.
        b (np.array): Vecteur des biais initial.
        X (np.array): Données d'entraînement de taille (N, D).
        Y (np.array): Prédictions initiales (probabilités) de taille (N, K).
        T (np.array): Étiquettes réelles one-hot de taille (N, K).
        lr (float, optionnel): Taux d'apprentissage (défaut: 0.1).
        nb_iter (int, optionnel): Nombre d'itérations (défaut: 1000).
        int_affiche (int, optionnel): Intervalle d'itérations pour afficher l'erreur et la précision (défaut: 100).
        quiet (bool, optionnel): Si True, n'affiche pas les informations intermédiaires (défaut: False).

    Returns:
        tuple(list, list):
         - Historique de l'erreur de cross-entropie sous forme de liste de tuples (itération, erreur).
         - Historique de la précision sous forme de liste de tuples (itération, erreur).

    """
    suite_erreur = [(0, cross_entropy(Y, T))]
    suite_precision = [(0, taux_precision(predit_classe(Y, K), T))]
    for i in range(1, nb_iter + 1):
        updateWb(W, b, X, Y, T, lr)
        Y[:] = predit_proba(X, W, b)
        if i % int_affiche == 0:
            erreur_iter = cross_entropy(Y, T)
            precision_iter = taux_precision(predit_classe(Y, K), T)
            if not quiet:
                print("Erreur d'entropy à l'itération ", i, " : ", erreur_iter)
                print("Précision à l'itération ", i, " : ", precision_iter)
            suite_erreur.append((i, erreur_iter))
            suite_precision.append((i, precision_iter))
    return suite_erreur, suite_precision


# %% Importer le jeu de données pour la régression logistique - 4 classes
# * Décommentez les lignes ci-dessous pour utiliser le jeu de données à 4 classes pour la régression logistique

# # Dimensions et recupération des données
# X_train, T_train = readdataset2d("probleme_4_classes")
# N, D = X_train.shape
# K = 4

# # On conserve T pour l'affichage avec les frontières
# T_conserve = T_train.copy()

# # Initialisation des parametres pour la regresssion logistique
# W, b, Y_train, C_train_init = initialise(D, K)
# T_train = convertit(T_train, K)
# W_init = W.copy()
# b_init = b.copy()

# # Regression
# suite_erreur, suite_precision = regression_logistique(W, b, X_train, Y_train, T_train, lr = 0.01)

# # Affichages
# viz.tracer_progression_entrainement(suite_erreur, suite_precision)
# viz.tracer_frontieres_decision(W, b, X_train, T_conserve)

# %% Importer le jeu de données pour la régression logistique - 5 classes
# * Décommentez les lignes ci-dessous pour utiliser le jeu de données à 5 classes pour la régression logistique


# Dimensions et recupération des données

# X_train, T_train = readdataset2d("probleme_5_classes")
# N, D = X_train.shape
# K = 5

# # On conserve T pour l'affichage avec les frontières
# T_conserve = T_train.copy()

# # Initialisation des parametres pour la regresssion logistique
# W, b, Y_train, C_train_init = initialise(D, K)
# T_train = convertit(T_train, K)
# W_init = W.copy()
# b_init = b.copy()

# # Regression
# suite_erreur, suite_precision = regression_logistique(W, b, X_train, Y_train, T_train, lr = 0.01)
# print(suite_precision)
# # Affichages
# viz.tracer_progression_entrainement(suite_erreur, suite_precision)
# viz.tracer_frontieres_decision(W, b, X_train, T_conserve)

# %% Importer le jeu de données pour la régression logistique - Problème à 5 classes plus difficile
# * Décommentez les lignes ci-dessous pour utiliser le jeu de données à 5 classes plus difficile pour la régression logistique


# # # Dimensions et recupération des données

# X_train, T_train = readdataset2d("probleme_5_plus_difficile")
# N, D = X_train.shape
# K = 5

# # On conserve T pour l'affichage avec les frontières
# T_conserve = T_train.copy()

# # Initialisation des parametres pour la regresssion logistique
# W, b, Y_train, C_train_init = initialise(D, K)
# T_train = convertit(T_train, K)
# W_init = W.copy()
# b_init = b.copy()

# # Regression
# suite_erreur, suite_precision = regression_logistique(W, b, X_train, Y_train, T_train, lr = 0.01)

# # Affichages
# viz.tracer_progression_entrainement(suite_erreur, suite_precision)
# viz.tracer_frontieres_decision(W, b, X_train, T_conserve)


# %% RDN pour des donnnées plus difficiles

# Ici, nous utilisons un réseau de neurones dense (à couches cachées multiples)
# pour le problème à 5 classes plus difficile.
X_train, T_train = readdataset2d("probleme_5_plus_difficile")
N, D = X_train.shape
K = 5
T_conserve = T_train.copy()


def sigma(x: float) -> float:
    """
    Calcule la fonction d'activation sigmoïde.

    Args:
        x (float): La valeur d'entrée.

    Returns:
        float: La sortie de la fonction sigmoïde, calculée par 1/(1 + exp(-x)).
    """
    return 1 / (1 + np.exp(-x))


def predit_proba_dense(parameters: list) -> list:
    """
    Effectue une prédiction à l'aide d'un réseau de neurones dense avec plusieurs couches cachées.

    Le réseau applique une fonction sigmoïde pour les couches cachées et une fonction softmax pour la couche de sortie.

    Args:
        parameters (list): Liste de tuples (W, b) représentant les paramètres de chaque couche.

    Returns:
        list: Liste des activations pour chaque couche, la dernière correspondant aux probabilités de sortie.
    """
    W, b = parameters[0]
    datas = [sigma(X_train.dot(W) + b)]  # Sigmoïde pour la première couche cachée
    for i in range(1, len(parameters)):
        W, b = parameters[i]
        Wprev = datas[-1]
        if i == len(parameters) - 1:  # Dernière couche (sortie)
            datas.append(softmax(Wprev.dot(W) + b))
        else:
            datas.append(sigma(Wprev.dot(W) + b))
    return datas


def create_parameters(dimensions: list) -> list:
    """
    Crée et initialise les paramètres (poids et biais) pour un réseau de neurones dense.

    Chaque poids est initialisé à partir d'une distribution uniforme et chaque biais à partir d'une distribution normale.

    Args:
        dimensions (list): Liste des dimensions de chaque couche. Par exemple, [2, 6, 2, 5] définit un réseau avec
                           une couche d'entrée de 2 neurones, deux couches cachées de 6 et 2 neurones, et une couche de sortie de 5 neurones.

    Returns:
        list: Liste de tuples (W, b) pour chaque couche du réseau.
    """
    parameters = []
    for indice in range(len(dimensions) - 1):
        W = np.random.uniform(-2, 2, size=(dimensions[indice], dimensions[indice + 1]))
        b = np.random.randn(1, dimensions[indice + 1])
        parameters.append((W, b))
    return parameters


def initialise_dense(dimensions: list, K) -> tuple[list, np.array, np.array]:
    """
    Initialise les paramètres d'un réseau de neurones dense et calcule une première prédiction.

    Args:
        dimensions (list): Liste des dimensions pour chaque couche du réseau.
        K (int): Nombre de classes (utilisé pour déterminer les prédictions one-hot).

    Returns:
        tuple:
            - list: Liste des paramètres (W, b) pour chaque couche.
            - np.array: Tableau des activations (datas) de chaque couche.
            - np.array: Prédiction initiale sous forme one-hot.
    """
    parameters = create_parameters(dimensions)
    datas = predit_proba_dense(parameters)
    C = predit_classe(datas[-1], K)
    return parameters, datas, C


def updateWb(X: np.array, parameters: list, datas: list, T: np.array, lr: float):
    """
    Met à jour les poids et biais d'un réseau de neurones dense par rétropropagation.

    La mise à jour se fait en commençant par la couche de sortie (softmax) puis en remontant à travers les couches cachées (sigmoïde).

    Args:
        X (np.array): Données d'entrée de taille (N, D).
        parameters (list): Liste des paramètres (W, b) pour chaque couche.
        datas (list): Liste des activations obtenues pour chaque couche.
        T (np.array): Étiquettes réelles one-hot de taille (N, K).
        lr (float): Taux d'apprentissage.
    """
    Y = datas[-1]  # Sortie après softmax
    delta = Y - T  # Gradient initial (dérivée de softmax)

    # Mise à jour de la dernière couche
    W_last, b_last = parameters[-1]
    Z_prev = datas[-2] if len(datas) >= 2 else X  # Activation de la couche précédente
    parameters[-1] = (
        W_last - lr * np.dot(Z_prev.T, delta),  # Mise à jour de W
        b_last - lr * np.sum(delta, axis=0, keepdims=True),  # Mise à jour de b
    )

    # Rétropropagation pour les couches cachées utilisant la dérivée de la sigmoïde
    for i in range(len(parameters) - 2, -1, -1):
        W_curr, b_curr = parameters[i]
        Z_prev = X if i == 0 else datas[i - 1]
        Z_curr = datas[i]

        delta = np.dot(delta, parameters[i + 1][0].T) * (Z_curr * (1 - Z_curr))

        parameters[i] = (
            W_curr - lr * np.dot(Z_prev.T, delta),
            b_curr - lr * np.sum(delta, axis=0, keepdims=True),
        )


def reseau_dense(
    X: np.array,
    parameters: list,
    datas: list,
    T: np.array,
    K,
    lr=0.001,
    nb_iter=1000,
    int_affiche=100,
    quiet=False,
):
    """
    Entraîne un réseau de neurones dense en utilisant la rétropropagation et la descente de gradient.

    À chaque itération, la fonction met à jour les paramètres du réseau et enregistre l'erreur de cross-entropie
    ainsi que le taux de précision.

    Args:
        X (np.array): Données d'entrée de taille (N, D).
        parameters (list): Liste des paramètres (W, b) pour chaque couche du réseau.
        datas (list): Liste des activations initiales pour chaque couche.
        T (np.array): Étiquettes réelles one-hot de taille (N, K).
        K (int): Nombre de classes.
        lr (float, optionnel): Taux d'apprentissage (défaut: 0.001).
        nb_iter (int, optionnel): Nombre d'itérations d'entraînement (défaut: 1000).
        int_affiche (int, optionnel): Intervalle d'itérations pour afficher les statistiques (défaut: 100).
        quiet (bool, optionnel): Si True, n'affiche pas les informations intermédiaires (défaut: False).

    Returns:
        tuple:
            - list: Historique de l'erreur de cross-entropie sous forme de tuples (itération, erreur).
            - list: Historique du taux de précision sous forme de tuples (itération, précision).
    """
    Y = datas[-1]
    suite_erreur = [(0, cross_entropy(Y, T))]
    suite_precision = [(0, taux_precision(predit_classe(Y, K), T))]
    for i in range(1, nb_iter + 1):
        updateWb(X, parameters, datas, T, lr)
        datas[:] = predit_proba_dense(parameters)[:]
        Y = datas[-1]
        if i % int_affiche == 0:
            erreur_iter = cross_entropy(Y, T)
            precision_iter = taux_precision(predit_classe(Y, K), T)
            if not quiet:
                print("Cross-entropy error at iteration ", i, " : ", erreur_iter)
                print("Accuracy at iteration ", i, " : ", precision_iter)
            suite_erreur.append((i, erreur_iter))
            suite_precision.append((i, precision_iter))
    return suite_erreur, suite_precision


# Paramètres pour le réseau de neurones
dimensions = [2, 6, 2, 5]
T_train = convertit(T_train, K)
parameters, datas, C_train_init = initialise_dense(dimensions, K)

# Entraînement du réseau de neurones et suivi de l'erreur et de la précision
suite_erreur, suite_precision = reseau_dense(X_train, parameters, datas, T_train, K)

# Prédictions finales après entraînement
Y_fin = datas[-1]
C_train_final = predit_classe(Y_fin, K)
W_fin, b_fin = parameters[-1]

# Décommenter pour visualiser les résultats:

viz.tracer_progression_entrainement(suite_erreur, suite_precision)

# ! Attention, décommenter la section ci dessous uniquement si dimensions a comme avant dernier parametre 2
# ! en effet, on a besoin d'avoir des parametres de dimension 2 si on veut tracer les courbes de separation
# viz.tracer_frontieres_decision(
#     W_fin, b_fin, X_train, T_conserve, architecure=dimensions
# )


# %%
