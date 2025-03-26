import numpy as np
import matplotlib.pyplot as plt 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from display import *
from exercice1 import *

#%% Question 1
"""
Si on cherche à minimiser J, alors on cherche à maximiser les probabilités y_{nk}\in [0,1] car ln(1) = 0 alors que si y_{nk} proche de 0
ln(y_{nk}) devient très petit.
Ce qui veut dire qu'on cherche la plus grande probabibilité de y_{nk}, donc la plus grande probabilité à réaliser une bonne classification
"""
#%% Introduction : Lecture des jeux de données fournis 
def readdataset2d(fname):
    with open(fname, "r") as file:
        X, T = [], []
        for l in file:
            x = l.strip().split()
            # print("x: ",x)
            X.append((float(x[0]), float(x[1])))
            T.append(int(x[2]))
        T = np.reshape(np.array(T), (-1,1)) 
    return np.array(X), T

#%% Fonction convertit de la question 2
def convertit(T: np.array, K: int) -> np.array:
    new_T = np.zeros((len(T), K))  
    for i, classification in enumerate(T):
        new_T[i, classification - 1] = 1 
    return new_T

#print(convertit(np.array([1,2]), 2))

def softmax(A: np.array) -> np.array:
    B = []
    for ligne in A:
        B.append(1/np.sum(np.exp(ligne))*np.exp(ligne))
    return np.array(B)

def predit_proba(X: np.array, W: np.array, b: np.array):
    return softmax( X.dot(W) + b)

# idée -> parcourir la ligne associé à Xi dans Y et voir l'indice de la plus grande valeur et ensuite convertir comme avec T_train
def predit_classe(Y: np.array, K: int) -> np.array:
    prediction_index = np.argmax(Y, axis = 1) + 1
    return convertit(prediction_index, K)


def initialise(D: int, K: int) -> tuple[np.array, np.array, np.array, np.array]:    
    W = np.random.uniform(-2, 2, size = (D, K))
    b = np.random.randn(1, K)
    Y = predit_proba(X_train, W, b)
    C = predit_classe(Y, K)
    return W, b, Y, C

def cross_entropy(Y: np.array, T: np.array):
    N, K = Y.shape
    J = 0
    for n in range(N):
        for k in range(K):
            if T[n, k] == 1:
                if np.log(Y[n, k]) == 0.0:
                    continue
                else :
                    J -= T[n, k]*np.log(Y[n, k])
    return J

def updateWb(W: np.array, b: np.array,X: np.array, Y: np.array, T: np.array,lr: float):
    W -= lr*(X.transpose()).dot((Y-T))
    for j in range(0, K):
        grad = 0
        for n in range(N):
            grad += np.sum(Y - T, axis = 1)[n]
        b[0, j] -= lr*grad

def taux_precision(C: np.array, T: np.array):
    N = T.shape[0]
    well_classified = 0
    for i in range(N):
        if C[i].all == T[i].all:
            well_classified += 1
    return well_classified*100/N

def regression_logistique(W, b, X, Y, T, lr=0.1, nb_iter=10000, int_affiche=100, quiet = False ):
    suite_erreur = [cross_entropy(Y,T)]
    suite_precision = [taux_precision(predit_classe(Y, K), T)]
    for i in range(nb_iter):
        updateWb(W,b,X,Y,T,lr)
        Y[:] = predit_proba(X, W, b)
        if i % int_affiche == 0:
            erreur_iter = cross_entropy(Y, T)
            precision_iter = taux_precision(predit_classe(Y, K), T)
            if not quiet:
                print("Erreur cross_entropy a l'iteration ", i+1 ," : " , erreur_iter)
            suite_erreur.append(erreur_iter)
            suite_precision.append(precision_iter)
    return suite_erreur

def taux_precision(C: np.array, T: np.array):
    true_labels = np.argmax(T, axis=1)
    predicted_labels = np.argmax(C, axis=1)
    
    # Calcul du taux de précision
    accuracy = np.mean(true_labels == predicted_labels)
    
    return accuracy




# #%% Import du jeu de données : probleme à 4 classes
# X_train, T_train = readdataset2d("probleme_4_classes")
# N, D = X_train.shape
# K = 4
# # Pour la visualisation, on garde T_train sous sa forme originelle
# plt.scatter(X_train[:,0], X_train[:,1], c=T_train, s = 30)
# # plt.show()
# W, b, Y_train, C_train_init = initialise(D, K)
# T_train = convertit(T_train, K)
# W_init = W.copy()
# b_init = b.copy()
# affichage_fonction_erreur(regression_logistique(W, b, X_train, Y_train, T_train, lr=0.01))


# #%% Import du jeu de données : probleme à 5 classes

X_train, T_train = readdataset2d("probleme_5_classes")
N, D = X_train.shape
K = 5 #5 classes
T_conserve = T_train.copy()
# Pour la visualisation, on garde T_train sous sa forme originelle
# plt.scatter(X_train[:,0], X_train[:,1], c=T_train, s = 30)
W, b, Y_train, C_train_init = initialise(D, K)
T_train = convertit(T_train, K)
W_init = W.copy()
b_init = b.copy()
affichage_fonction_erreur(regression_logistique(W, b, X_train, Y_train, T_train, 0.05, 1000, 100))
display_datas_and_separation(W, b, X_train, T_conserve)
C_final = predit_classe(Y_train, K)
print(taux_precision(C_final, T_train))

#%% Import du jeu de données : probleme à 6 classes (mais c'est devenu 5 d'apres le prof)
# X_train, T_train = readdataset2d("probleme_6_classes")
# N, D = X_train.shape
# K = 5 # Le prof a dis que c'était 5 classes au final

# # Pour la visualisation, on garde T_train sous sa forme originelle
# plt.scatter(X_train[:,0], X_train[:,1], c=T_train, s = 30)
# # plt.show()
# W, b, Y_train, C_train_init = initialise(D, K)
# T_train = convertit(T_train, K)
# W_init = W.copy()
# b_init = b.copy()
# affichage_fonction_erreur(regression_logistique(W, b, X_train, Y_train, T_train, lr=0.01))

#%% A FAIRE :: OPTIMISER LE RDN pour le probleme à 6 classes
# Comme on voit bien que juste la ligne c'est pas assez pour bien separer les differentes classes on va 
#faire comme avant avec les reseaux denses à p couches intermediaires
# Ducoup on va faire comme l'exercice 1.
