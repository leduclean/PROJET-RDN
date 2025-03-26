import numpy as np
import matplotlib.pyplot as plt 

def affichage_fonction_erreur(suite_erreur: list, label= 'Erreur de cross entropy', couleur = 'blue'):
    n = len(suite_erreur)
    X = np.arange(0, n)
    fig,ax = plt.subplots()
    ax.plot(X, suite_erreur , color = couleur, label = label)
    plt.title("Cross entropy Error")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

def affichage_fonction_precision(suite_precision: list, label, couleur = "blue"):
    n = len(suite_precision)
    X = np.arange(0, n)
    fig,ax = plt.subplots()
    ax.plot(X, suite_precision , color = couleur, label = label)
    plt.title("taux de precision")
    plt.xlabel("Iterations")
    plt.ylabel("taux")
    plt.legend()
    plt.show()