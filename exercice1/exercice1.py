#%% Introduction : Lecture des jeux de données fournis 
import numpy as np
import matplotlib.pyplot as plt 
import sys
import os
# ? est ce que le prof veut un fichier jupyther executable a la fin ?  
current_dir = os.path.dirname(os.path.abspath("exercice1/"))  # Dossier du fichier
project_root = os.path.dirname(current_dir)  # Dossier du projet (remonte d'un niveau)

if project_root not in sys.path:
    sys.path.append(project_root)

from visualisation.visualizer import Visualiseur

def readdataset2d(nom_fichier: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Lit un fichier de données en 2D et retourne les caractéristiques et les étiquettes associées.

    Paramètres:
        nom_fichier (str): Chemin vers le fichier contenant les données. Chaque ligne doit contenir trois valeurs
                           séparées par des espaces (deux pour les caractéristiques et une pour l'étiquette).

    Retourne:
        tuple[np.ndarray, np.ndarray]: Un tuple contenant:
            - np.ndarray: Tableau des caractéristiques de forme (N, 2).
            - np.ndarray: Tableau des étiquettes de forme (N, 1).
    """
    with open(nom_fichier, "r") as fichier:
        X, T = [], []
        for ligne in fichier:
            valeurs = ligne.strip().split()
            X.append((float(valeurs[0]), float(valeurs[1])))
            T.append(int(valeurs[2]))
        T = np.reshape(np.array(T), (-1, 1))
    return np.array(X), T

#%% Import du jeu de données d'entraînement
X_train, T_train = readdataset2d("nuage_exercice_1")
N, D = X_train.shape
# plt.scatter(X_train[:,0], X_train[:,1], c=T_train, s=10)

#%% Import du jeu de données de test
X_test, T_test = readdataset2d("nuage_test_exercice_1")
N, D = X_test.shape
# plt.scatter(X_test[:,0], X_test[:,1], c=T_test, s=10)

#%%
# 1. Implémentation du réseau

def sigma(x: float) -> float:
    """
    Calcule la fonction sigmoïde pour une valeur d'entrée.

    Paramètres:
        x (float): Valeur d'entrée.

    Retourne:
        float: Résultat de la fonction sigmoïde, défini par 1 / (1 + exp(-x)).
    """
    return 1 / (1 + np.exp(-x))


def predit_classe(Y: np.ndarray) -> np.ndarray:
    """
    Prédit la classe en arrondissant la sortie du modèle.

    Paramètres:
        Y (np.ndarray): Sortie du réseau de neurones.

    Retourne:
        np.ndarray: Classes prédites (valeurs arrondies).
    """
    return np.round(Y)


def taux_precision(C: np.ndarray, T: np.ndarray) -> float:
    """
    Calcule le taux de précision des prédictions en pourcentage.

    Paramètres:
        C (np.ndarray): Classes prédites.
        T (np.ndarray): Classes réelles.

    Retourne:
        float: Taux de précision, calculé en pourcentage.
    """
    N = len(T)
    return np.sum(np.equal(T, C)) * 100 / N


def cross_entropy(Y: np.ndarray, T: np.ndarray) -> float:
    """
    Calcule l'erreur d'entropie croisée entre la sortie du modèle et les cibles réelles.

    Paramètres:
        Y (np.ndarray): Sortie du modèle (prédictions).
        T (np.ndarray): Cibles réelles.

    Retourne:
        float: Erreur d'entropie croisée.
    """
    N = len(Y)
    J = 0
    for i in range(N):
        if T[i] == 1:
            if np.log(Y[i]) == 0.0:
                continue
            else:
                J -= np.log(Y[i])
        else:
            if np.log(1 - Y[i]) == 0.0:
                continue
            else:
                J -= np.log(1 - Y[i])
    return J


def create_parameters(dimensions: list[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Initialise les paramètres du réseau de neurones (poids et biais) pour chaque couche.

    Paramètres:
        dimensions (list[int]): Liste des dimensions de chaque couche (par exemple, [2, 15, 3, 1]).

    Retourne:
        list[tuple[np.ndarray, np.ndarray]]: Liste de tuples (W, b) pour chaque couche.
    """
    parameters = []
    for indice in range(len(dimensions) - 1):
        W = np.random.uniform(-2, 2, size=(dimensions[indice], dimensions[indice + 1]))
        b = np.random.randn(1, dimensions[indice + 1])
        parameters.append((W, b))
    return parameters


def get_datas_from_parameters(parameters: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
    """
    Calcule les activations de chaque couche du réseau en fonction des paramètres.

    Paramètres:
        parameters (list[tuple[np.ndarray, np.ndarray]]): Liste des poids et biais de chaque couche.

    Retourne:
        list[np.ndarray]: Liste des activations pour chaque couche.
    """
    W, b = parameters[0]
    datas = [sigma(np.dot(X_train, W) + b)]
    for i in range(1, len(parameters)):
        W, b = parameters[i]
        activation_precedente = datas[-1]
        datas.append(sigma(np.dot(activation_precedente, W) + b))
    return datas


def initialise(dimensions: list[int]) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[np.ndarray], np.ndarray]:
    """
    Initialise les paramètres du réseau, calcule les activations et prédit les classes initiales.

    Paramètres:
        dimensions (list[int]): Liste des dimensions des couches du réseau.

    Retourne:
        tuple:
            - list[tuple[np.ndarray, np.ndarray]]: Paramètres (poids et biais) du réseau.
            - list[np.ndarray]: Activations de chaque couche.
            - np.ndarray: Classes prédites initialement.
    """
    parameters = create_parameters(dimensions)
    datas = get_datas_from_parameters(parameters)
    C = predit_classe(datas[-1])
    return parameters, datas, C


def updateW(X: np.ndarray, parameters: list[tuple[np.ndarray, np.ndarray]], datas: list[np.ndarray], T: np.ndarray, lr: float) -> None:
    """
    Met à jour les poids et biais du réseau de neurones par rétropropagation.

    Paramètres:
        X (np.ndarray): Données d'entrée.
        parameters (list[tuple[np.ndarray, np.ndarray]]): Paramètres actuels (poids et biais) du réseau.
        datas (list[np.ndarray]): Activations intermédiaires de chaque couche.
        T (np.ndarray): Cibles réelles.
        lr (float): Taux d'apprentissage.
    """
    Y = datas[-1]
    delta = Y - T
    W_last, b_last = parameters[-1]
    activation_precedente = datas[-2] if len(datas) >= 2 else X
    parameters[-1] = (
        W_last - lr * np.dot(activation_precedente.T, delta),
        b_last - lr * np.sum(delta, axis=0, keepdims=True),
    )

    # Rétropropagation pour les couches cachées
    for i in range(len(parameters) - 2, -1, -1):
        W_courant, b_courant = parameters[i]
        activation_entree = X if i == 0 else datas[i - 1]
        activation_courante = datas[i]
        delta = np.dot(delta, parameters[i + 1][0].T) * (activation_courante * (1 - activation_courante))
        parameters[i] = (
            W_courant - lr * np.dot(activation_entree.T, delta),
            b_courant - lr * np.sum(delta, axis=0, keepdims=True),
        )


def reseau(
    X: np.ndarray,
    parameters: list[tuple[np.ndarray, np.ndarray]],
    datas: list[np.ndarray],
    T: np.ndarray,
    lr: float = 0.0004,
    nb_iter: int = 10000,
    int_affiche: int = 100,
    quiet: bool = False,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """
    Entraîne le réseau de neurones et renvoie les courbes d'erreur et de précision.

    Paramètres:
        X (np.ndarray): Données d'entrée.
        parameters (list[tuple[np.ndarray, np.ndarray]]): Paramètres initiaux du réseau.
        datas (list[np.ndarray]): Activations intermédiaires du réseau.
        T (np.ndarray): Cibles réelles.
        lr (float, optionnel): Taux d'apprentissage (défaut: 0.001).
        nb_iter (int, optionnel): Nombre d'itérations (défaut: 10000).
        int_affiche (int, optionnel): Intervalle d'affichage (défaut: 100).
        quiet (bool, optionnel): Mode silencieux (défaut: False).

    Retourne:
        tuple:
            - list[tuple[int, float]]: Liste des tuples (itération, erreur) pour la courbe d'erreur.
            - list[tuple[int, float]]: Liste des tuples (itération, précision) pour la courbe de précision.
    """
    Y = datas[-1]
    suite_erreur = [(0, cross_entropy(Y, T))]
    suite_precision = [(0, taux_precision(predit_classe(Y), T))]
    for i in range(1, nb_iter + 1):
        updateW(X, parameters, datas, T, lr)
        datas[:] = get_datas_from_parameters(parameters)[:]
        Y = datas[-1]
        if i % int_affiche == 0:
            erreur_iter = cross_entropy(Y, T)
            precision_iter = taux_precision(predit_classe(Y), T)
            if not quiet:
                print("Erreur cross-entropy à l'itération", i, ":", erreur_iter)
                print("Précision à l'itération", i, ":", precision_iter)
            suite_erreur.append((i, erreur_iter))
            suite_precision.append((i, precision_iter))
    return suite_erreur, suite_precision

def courbes_lr(
    dimensions: list[int],
    alpha_min: float, alpha_max: float, nbr_elements: int
) -> tuple[list[list[tuple[int, float]]], np.ndarray]:
    """
    Calcule un tableau de modulation du taux d'apprentissage en entraînant le réseau pour différentes valeurs de lr.

    Paramètres:
        alpha_min (float): Valeur minimale du taux d'apprentissage.
        alpha_max (float): Valeur maximale du taux d'apprentissage.
        nbr_elements (int): Nombre de valeurs à tester.
        a_afficher (str, optionnel): Type de courbe à afficher ("error" ou "precision", défaut: "error").

    Retourne:
        tuple:
            - list[list[tuple[int, float]]]: Liste des courbes (erreur ou précision) pour chaque valeur de lr.
            - np.ndarray: Tableau des valeurs de taux d'apprentissage testées.
    """
    parameters, datas, _ = initialise(dimensions)
    lr_values = np.linspace(alpha_min, alpha_max, nbr_elements)
    liste_courbes = [
            reseau(X_train, parameters, datas, T_train, lr, quiet=True) for lr in lr_values
        ]
    return liste_courbes, lr_values

def single_courbe_depuis_dimensions(dimension: list) -> list[tuple[list[tuple[int, float]], list[tuple[int, float]]]]:
    """
    Calcule les courbes d'erreur et de précision pour une seule architecture de reseau

    Paramètres:
        list_of_dimensions (list[list[int]]): Liste des architectures (dimensions) à tester.

    Retourne:
        list[tuple[list[tuple[int, float]], list[tuple[int, float]]]]: Liste des courbes d'erreur et de précision pour chaque architecture.
    """
    parameters, datas, _ = initialise(dimension)
    suite_erreur, suite_precision = reseau(X_train, parameters,datas, T_train, quiet=True)
    return suite_erreur, suite_precision

def multiple_courbes_depuis_dimensions(
    list_of_dimensions: list[list[int]], single = False,
) -> list[list[tuple[list[tuple[int, float]], list[tuple[int, float]]]]]:
    """
    Calcule les courbes d'erreur et de précision pour plusieurs architectures de réseau.

    Paramètres:
        list_of_dimensions (list[list[int]]): Liste des architectures (dimensions) à tester.

    Retourne:
        list[list[tuple[list[tuple[int, float]], list[tuple[int, float]]]]]: Liste des listes des courbes d'erreur et de précision pour chaque architecture.
    """
    curves_array = []
    for dimension in list_of_dimensions:
        parameters_tmp, datas_tmp, _ = initialise(dimension)
        suite_erreur_tmp, suite_precision_tmp = reseau(X_train, parameters_tmp, datas_tmp, T_train, quiet=True)
        curves_array.append((suite_erreur_tmp, suite_precision_tmp))
    return curves_array


# %% 
# Affichage

# Création d'un visualiseur et affichage des courbes
viz = Visualiseur()

# # * Affichage de l'erreur et du taux de precision pour un reseau quelquonque 
# suite_erreur, suite_precision = single_courbe_depuis_dimensions(dimension = [2, 7, 7, 7, 1])
# viz.tracer_progression_entrainement(suite_erreur, suite_precision)

# * Impact du taux d'apprentissage :
# courbes_lr, lr_modulation = courbes_lr([2,3,3,3,1], 0.0004, 0.001, 6)
# viz.tracer_multiple(courbes_lr, a_afficher ="precision", modulation_taux = lr_modulation)



# # * Tracé pour toutes les dimensions

# DIMENSIONS = [
#     [2, 3, 3, 1],
#     [2, 7, 7, 7, 1],
#     [2, 15, 15, 1],
#     [2, 3, 15, 15, 1],
#     [2, 15, 15, 3, 1],
#     [2, 40, 1],
#     [2, 20, 20, 1],
#     [2, 5, 4, 4, 4, 4, 1]
# ]

# # error 
# viz.tracer_multiple(multiple_courbes_depuis_dimensions(DIMENSIONS), DIMENSIONS,"error")
# precision
# viz.tracer_multiple(multiple_courbes_depuis_dimensions(DIMENSIONS), DIMENSIONS,"precision")


# # * Impact du nombre de couches intermédiaires:

# dimensions_nbr_couches_diff = [
#     [2, 5, 1],
#     [2, 5, 4, 1],
#     [2, 5, 4, 4, 1],
#     [2, 5, 4, 4, 4, 1],
# ]

# viz.tracer_multiple(multiple_courbes_depuis_dimensions(dimensions_nbr_couches_diff), dimensions_nbr_couches_diff, "precision")

# *Impact de la répartitions des dimensions
dimensions_reparties = [
    [2,3,8,1],
    [2,8,3,1]
]

viz.tracer_multiple(multiple_courbes_depuis_dimensions(dimensions_reparties), dimensions_reparties, "precision")


# %%
