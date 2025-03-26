import numpy as np
import matplotlib.pyplot as plt 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from display import affichage_fonction_erreur, affichage_fonction_precision

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
X_train, T_train = readdataset2d("nuage_exercice_1")
N, D = X_train.shape
# plt.scatter(X_train[:,0], X_train[:,1], c=T_train, s = 10)

#%% Import du jeu de données de test
X_test, T_test = readdataset2d("nuage_test_exercice_1")
N, D = X_test.shape
# plt.scatter(X_test[:,0], X_test[:,1], c=T_test, s = 10)

# %%
# 1. Implémentation du réseau
def sigma(x):
    return 1/(1+np.exp(-x))

def predit_classe(Y):
    return np.round(Y)

def taux_precision(C, T):
    N = len(T)
    return np.sum(np.equal(T, C))*100/N

def cross_entropy(Y,T):
    N = len(Y)
    J = 0
    for i in range(N):
        if T[i] == 1:
            if np.log(Y[i]) == 0.0:
                continue
            else :
                J -= np.log(Y[i])
        else :
            if np.log(1-Y[i]) == 0.0:
                continue
            else :
                J -= np.log(1-Y[i])
    return J


def create_parameters(dimensions: list) -> list:
    parameters = []
    for indice in range(len(dimensions) - 1):
        W = np.random.uniform(-2, 2, size = (dimensions[indice], dimensions[indice + 1]))
        # b de dimension (1, dimensions[indice+1])
        b = np.random.randn(1, dimensions[indice+1])
        parameters.append((W, b))
    return parameters

def get_datas_from_parameters(parameters: list) -> np.array:
    W, b = parameters[0]
    datas = [sigma(np.dot(X_train, W) + b)]
    for i in range(1, len(parameters)):
        W, b  = parameters[i]
        Wprev = datas[-1]
        datas.append(sigma(np.dot(Wprev, W) + b))
    return datas

def initialise(dimensions: list) -> tuple [list, np.array, np.array]:
    parameters = create_parameters(dimensions)
    datas = get_datas_from_parameters(parameters)
    C = predit_classe(datas[-1])
    return parameters, datas, C

def updateW(X: np.array, parameters: list, datas: list, T: np.array, lr: float):
    # Y est la sortie du réseau
    Y = datas[-1]
    # Calcul du delta pour la couche de sortie
    delta = Y - T 
    # Pour la dernière couche (couche L)
    W_last, b_last = parameters[-1]
    Z_prev = datas[-2] if len(datas) >= 2 else X  # activation de la couche précédente
    parameters[-1] = (W_last - lr * np.dot(Z_prev.T, delta),
                      b_last - lr * np.sum(delta, axis=0, keepdims=True))
    
    # Rétropropagation pour les couches cachées
    # Parcours de L-1 à 0
    for i in range(len(parameters) - 2, -1, -1):
        W_curr, b_curr = parameters[i]
        Z_prev = X if i == 0 else datas[i-1]
        Z_curr = datas[i]
        delta = np.dot(delta, parameters[i+1][0].T) * (Z_curr * (1 - Z_curr))
        # Mise à jour de la couche i
        parameters[i] = (W_curr - lr * np.dot(Z_prev.T, delta),
                         b_curr - lr * np.sum(delta, axis=0, keepdims=True))



def reseau(X: np.array, parameters: list, datas: list, T: np.array, lr = 0.001, nb_iter = 10000, int_affiche = 100, quiet = False):
    Y = datas[-1]
    suite_erreur = [cross_entropy(Y,T)]
    suite_precision = [taux_precision(predit_classe(Y), T)]
    for i in range(nb_iter):
        updateW(X, parameters, datas, T, lr)
        datas[:] = get_datas_from_parameters(parameters)[:]
        Y = datas[-1]
        if i % int_affiche == 0:
            erreur_iter = cross_entropy(Y, T)
            precision_iter = taux_precision(predit_classe(Y), T)
            if not quiet:
                print("Erreur cross_entropy a l'iteration ", i+1 ," : " , erreur_iter)
            suite_erreur.append(erreur_iter)
            suite_precision.append(precision_iter)
    return suite_erreur, suite_precision




dimensions = [2, 15, 15, 3, 1]
parameters, datas, C_train_init = initialise(dimensions)
suite_erreur, suite_precision = reseau(X_train, parameters, datas, T_train)
C_train_final = predit_classe(datas[-1])


def get_lr_modulation_array(alpha_min: int, alpha_max: int, nbr_of_element: int, to_display = "error") -> np.array:
    lr_values = np.linspace(alpha_min, alpha_max, nbr_of_element)
    if to_display == "error":
        liste_xn = [(reseau(X_train, parameters, datas, T_train, lr,  quiet = True)[0], lr) for lr in lr_values]
    if to_display == "precision":
        liste_xn = [(reseau(X_train, parameters, datas, T_train, lr,  quiet = True)[1], lr) for lr in lr_values]
    return liste_xn

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

def get_curve_array_from_dimension(list_of_dimensions: list):
    curves_array = []
    for dimension in list_of_dimensions :
        parameters, datas, _ = initialise(dimension)
        suite_erreur, suite_precision = reseau(X_train, parameters, datas, T_train, quiet = True)
        curves_array.append((suite_erreur, suite_precision))
    return curves_array
        

def affiche_multiple(liste_xn: list, to_display = "error", learning_rate_modulation = False , cmap='viridis'):
    n_points = len(liste_xn[0][0]) 
    X = np.arange(0, n_points)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Nombre de courbes à tracer
    n_curves = len(liste_xn)
    # Récupération d'une colormap avec n_curves couleurs distinctes
    colors = plt.cm.get_cmap(cmap, n_curves)
    # Tracer chaque courbe avec une couleur issue de la colormap
    if learning_rate_modulation:
        for i, xn in enumerate(liste_xn):
            curve_to_display, lr = xn
            ax.plot(X, curve_to_display, color=colors(i), label=f'Learning rate: {lr}')
    else:
        for i, xn in enumerate(liste_xn):
            ax.plot(X, xn[0] if to_display == "error" else xn[1], color=colors(i), label=f'Curve: {i + 1}')            
    if to_display == "error":
        ax.set_xlabel("Itérations")
        ax.set_ylabel("Entropy error")
        ax.set_title("Error comparison for different learning rate values")
        ax.legend()
    if to_display == "precision":
        ax.set_title("taux de precision")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("taux de precision")
        ax.legend()
    plt.show()


affiche_multiple(get_curve_array_from_dimension(DIMENSIONS))

# affichage_fonction_erreur(suite_erreur, label = f'Architecture {dimensions}')
# affichage_fonction_precision(suite_precision, label= f'Architecture {dimensions}')
# affiche_multiple(get_lr_modulation_array(0.0005, 0.001, 3, "error"), "error", True)