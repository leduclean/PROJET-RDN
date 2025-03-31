import numpy as np
import matplotlib.pyplot as plt 
from visualisation.visualizer import Visualizer
#%% Introduction : Lecture des jeux de données fournis 
def readdataset2d(fname):
    with open(fname, "r") as file:
        X, T = [], []
        for l in file:
            x = l.strip().split()
            X.append((float(x[0]), float(x[1])))
            T.append(int(x[2]))
        T = np.reshape(np.array(T), (-1,1)) 
    return np.array(X), T

#%% Import du jeu de données d'entrainement
X_train, T_train = readdataset2d("exercice1/nuage_exercice_1")
N, D = X_train.shape
# plt.scatter(X_train[:,0], X_train[:,1], c=T_train, s = 10)

#%% Import du jeu de données de test
X_test, T_test = readdataset2d("exercice1/nuage_test_exercice_1")
N, D = X_test.shape
# plt.scatter(X_test[:,0], X_test[:,1], c=T_test, s = 10)

# %%
# 1. Implémentation du réseau
import numpy as np
import matplotlib.pyplot as plt

# %%
# 1. Implémentation du réseau

def sigma(x: float) -> float:
    """
    Fonction sigmoïde.

    Parameters
    ----------
    x : float
        Valeur d'entrée.

    Returns
    -------
    float
        Valeur transformée par la fonction sigmoïde.
    """
    return 1 / (1 + np.exp(-x))


def predit_classe(Y: np.ndarray) -> np.ndarray:
    """
    Prédit la classe en arrondissant la sortie du modèle.

    Parameters
    ----------
    Y : np.ndarray
        Sortie du réseau de neurones.

    Returns
    -------
    np.ndarray
        Classes prédites.
    """
    return np.round(Y)


def taux_precision(C: np.ndarray, T: np.ndarray) -> float:
    """
    Calcule le taux de précision des prédictions.

    Parameters
    ----------
    C : np.ndarray
        Classes prédites.
    T : np.ndarray
        Classes réelles.

    Returns
    -------
    float
        Taux de précision en pourcentage.
    """
    N = len(T)
    return np.sum(np.equal(T, C)) * 100 / N


def cross_entropy(Y: np.ndarray, T: np.ndarray) -> float:
    """
    Calcule l'erreur d'entropie croisée.

    Parameters
    ----------
    Y : np.ndarray
        Sortie du modèle.
    T : np.ndarray
        Cibles réelles.

    Returns
    -------
    float
        Erreur d'entropie croisée.
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
    Initialise les paramètres du réseau de neurones.

    Parameters
    ----------
    dimensions : list[int]
        Liste des dimensions des couches du réseau.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        Liste des poids et biais pour chaque couche.
    """
    parameters = []
    for indice in range(len(dimensions) - 1):
        W = np.random.uniform(-2, 2, size=(dimensions[indice], dimensions[indice + 1]))
        b = np.random.randn(1, dimensions[indice + 1])
        parameters.append((W, b))
    return parameters


def get_datas_from_parameters(parameters: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
    """
    Calcule les activations des couches du réseau.

    Parameters
    ----------
    parameters : list[tuple[np.ndarray, np.ndarray]]
        Liste des poids et biais.

    Returns
    -------
    list[np.ndarray]
        Liste des activations de chaque couche.
    """
    W, b = parameters[0]
    datas = [sigma(np.dot(X_train, W) + b)]
    for i in range(1, len(parameters)):
        W, b = parameters[i]
        Wprev = datas[-1]
        datas.append(sigma(np.dot(Wprev, W) + b))
    return datas


def initialise(dimensions: list[int]) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[np.ndarray], np.ndarray]:
    """
    Initialise les paramètres et les activations du réseau.

    Parameters
    ----------
    dimensions : list[int]
        Liste des dimensions des couches du réseau.

    Returns
    -------
    tuple[list[tuple[np.ndarray, np.ndarray]], list[np.ndarray], np.ndarray]
        Les paramètres, les activations et les classes initiales.
    """
    parameters = create_parameters(dimensions)
    datas = get_datas_from_parameters(parameters)
    C = predit_classe(datas[-1])
    return parameters, datas, C


def updateW(X: np.ndarray, parameters: list[tuple[np.ndarray, np.ndarray]], datas: list[np.ndarray], T: np.ndarray, lr: float) -> None:
    """
    Met à jour les poids et biais du réseau par rétropropagation.

    Parameters
    ----------
    X : np.ndarray
        Données d'entrée.
    parameters : list[tuple[np.ndarray, np.ndarray]]
        Poids et biais du réseau.
    datas : list[np.ndarray]
        Activations intermédiaires.
    T : np.ndarray
        Cibles réelles.
    lr : float
        Taux d'apprentissage.
    """
    Y = datas[-1]
    delta = Y - T
    W_last, b_last = parameters[-1]
    Z_prev = datas[-2] if len(datas) >= 2 else X  # activation de la couche précédente
    parameters[-1] = (
        W_last - lr * np.dot(Z_prev.T, delta),
        b_last - lr * np.sum(delta, axis=0, keepdims=True),
    )

    # Rétropropagation pour les couches cachées
    for i in range(len(parameters) - 2, -1, -1):
        W_curr, b_curr = parameters[i]
        Z_prev = X if i == 0 else datas[i - 1]
        Z_curr = datas[i]
        delta = np.dot(delta, parameters[i + 1][0].T) * (Z_curr * (1 - Z_curr))
        parameters[i] = (
            W_curr - lr * np.dot(Z_prev.T, delta),
            b_curr - lr * np.sum(delta, axis=0, keepdims=True),
        )


def reseau(
    X: np.ndarray,
    parameters: list[tuple[np.ndarray, np.ndarray]],
    datas: list[np.ndarray],
    T: np.ndarray,
    lr: float = 0.001,
    nb_iter: int = 10000,
    int_affiche: int = 100,
    quiet: bool = False,
) -> tuple[list[float], list[float]]:
    """
    Entraîne le réseau de neurones et renvoie les np.array pour les courbes d'erreur et de précision.

    Parameters
    ----------
    X : np.ndarray
        Données d'entrée.
    parameters : list[tuple[np.ndarray, np.ndarray]]
        Poids et biais du réseau.
    datas : list[np.ndarray]
        Activations intermédiaires.
    T : np.ndarray
        Cibles réelles.
    lr : float, optional
        Taux d'apprentissage (default: 0.001).
    nb_iter : int, optional
        Nombre d'itérations (default: 10000).
    int_affiche : int, optional
        Intervalle d'affichage (default: 100).
    quiet : bool, optional
        Mode silencieux (default: False).

    Returns
    -------
    tuple[list[float], list[float]]
        Liste des erreurs et précisions au cours des itérations.
    """
    Y = datas[-1]
    suite_erreur = [(0, cross_entropy(Y, T))]
    suite_precision = [(0, taux_precision(predit_classe(Y), T))]
    for i in range(1, nb_iter  + 1):
        updateW(X, parameters, datas, T, lr)
        datas[:] = get_datas_from_parameters(parameters)[:]
        Y = datas[-1]
        if i % int_affiche == 0:
            erreur_iter = cross_entropy(Y, T)
            precision_iter = taux_precision(predit_classe(Y), T)
            if not quiet:
                print("Erreur cross_entropy a l'iteration ", i , " : ", erreur_iter)
                print("precision cross_entropy a l'iteration ", i ," : " , precision_iter)

            suite_erreur.append((i, erreur_iter))
            suite_precision.append((i, precision_iter))
    return suite_erreur, suite_precision

dimensions = [2, 15, 15, 3, 1]
parameters, datas, C_train_init = initialise(dimensions)
suite_erreur, suite_precision = reseau(X_train, parameters, datas, T_train)
C_train_final = predit_classe(datas[-1])

DIMENSIONS = [
    [2, 3, 3, 1],
    [2, 7, 7, 7, 1],
    [2, 15, 15, 1],
    [2, 3, 15, 15, 1],
    [2, 15, 15, 3, 1],
    [2, 40, 1],
    [2, 20, 20, 1],
    [2, 5, 4, 4, 4 ,4, 1]
]

def get_lr_modulation_array(
    alpha_min: int, alpha_max: int, nbr_of_element: int, to_display: str = "error"
) -> np.ndarray:
    """
    Calcule un tableau de valeurs de modulation du taux d'apprentissage.

    Parameters
    ----------
    alpha_min : int
        Taux d'apprentissage minimal.
    alpha_max : int
        Taux d'apprentissage maximal.
    nbr_of_element : int
        Nombre d'éléments.
    to_display : str, optional
        Type de courbe à afficher ("error" ou "precision", default: "error").

    Returns
    -------
    np.ndarray
        Liste des courbes d'erreur ou de précision en fonction des valeurs du taux d'apprentissage.
    """
    lr_values = np.linspace(alpha_min, alpha_max, nbr_of_element)
    if to_display == "error":
        liste_xn = [
            (reseau(X_train, parameters, datas, T_train, lr, quiet=True)[0])
            for lr in lr_values
        ]
    if to_display == "precision":
        liste_xn = [
            (reseau(X_train, parameters, datas, T_train, lr, quiet=True)[1])
            for lr in lr_values
        ]
    return liste_xn, lr_values


def get_curve_array_from_dimension(list_of_dimensions: list[list[int]], verif: bool = True) -> list[tuple[list[float], list[float]]]:
    """
    Calcule les courbes d'erreur et de précision pour plusieurs architectures.

    Parameters
    ----------
    list_of_dimensions : list[list[int]]
        Liste des dimensions des différentes architectures de réseau.
    verif : bool, optional
        Si True, test avec plusieurs architectures, sinon avec une seule (default: True).

    Returns
    -------
    list[tuple[list[float], list[float]]]
        Liste des courbes d'erreur et de précision pour chaque architecture.
    """
    curves_array = []
    if verif == False:
        dimension = list_of_dimensions
        parameters, datas, _ = initialise(dimension)
        suite_erreur, suite_precision = reseau(
            X_train, parameters, datas, T_train, quiet=True
        )
        curves_array.append((suite_erreur, suite_precision))
        return curves_array

    for dimension in list_of_dimensions:
        parameters, datas, _ = initialise(dimension)
        suite_erreur, suite_precision = reseau(
            X_train, parameters, datas, T_train, quiet=True
        )
        curves_array.append((suite_erreur, suite_precision))
    return curves_array




viz = Visualizer()
# viz.plot_error(suite_erreur, f"architecture {dimensions}")
# viz.plot_precision(suite_precision, f"Architecture {dimensions}")
# viz.plot_training_progress(suite_erreur, suite_precision)
# * Learning rate impact:
suite_erreur_lr, lr_modulation = get_lr_modulation_array(0.0005, 0.001, 3, "error")
viz.plot_multiple(suite_erreur_lr,"error", lr_modulation)