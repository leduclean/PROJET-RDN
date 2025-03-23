import numpy as np
import matplotlib.pyplot as plt 

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
plt.scatter(X_train[:,0], X_train[:,1], c=T_train, s = 10)

#%% Import du jeu de données de test
X_test, T_test = readdataset2d("nuage_test_exercice_1")
N, D = X_test.shape
plt.scatter(X_test[:,0], X_test[:,1], c=T_test, s = 10)

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


def get_parameter_dimensions(dimensions: list, parameter_indice: int) -> tuple:
    return (dimensions[parameter_indice], dimensions[parameter_indice + 1])

def create_parameters(dimensions: list) -> list:
    parameters = []
    for indice in range(len(dimensions) - 1):
        W = np.random.uniform(-2, 2, size=get_parameter_dimensions(dimensions, indice))
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



def reseau(X: np.array, parameters: list, datas: list, T: np.array, lr = 0.001, nb_iter = 10000, int_affiche = 10):
    Y = datas[-1]
    suite_erreur = [cross_entropy(Y,T)]
    for i in range(nb_iter):
        updateW(X, parameters, datas, T, lr)
        datas[:] = get_datas_from_parameters(parameters)[:]
        Y = datas[-1]
        if i % int_affiche == 0:
            erreur_iter = cross_entropy(Y, T)
            print("Erreur cross_entropy a l'iteration ", i+1 ," : " , erreur_iter)
            suite_erreur.append(erreur_iter)
    return suite_erreur

dimensions1 = [2,5,4,4,4,4,1]
parameters, datas, C_train = initialise(dimensions1)
reseau(X_train, parameters, datas, T_train)