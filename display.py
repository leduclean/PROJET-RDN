import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm  # Pour les couleurs



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

def display_datas_and_separation(W: np.array, b: np.array, X: np.array, T: np.array ):
    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    K = W.shape[1]
    
    # Définir une colormap avec K couleurs
    cmap = cm.get_cmap("viridis", K)  
    colors = [cmap(k) for k in range(K)]  # Liste des couleurs pour chaque classe

    # Tracer une frontière pour chaque classe (one-vs-all)
    for k in range(K):
        # Récupérer les poids pour la classe k
        w1, w2 = W[:, k]
        b_k = b[0, k]
        y_vals = -(w1 * x_vals + b_k) / w2
        plt.plot(x_vals, y_vals, color=colors[k], label=f"Frontière classe {k}")


    # Tracer les points avec les couleurs correspondantes
    plt.scatter(X[:, 0], X[:, 1], c=T, s=30, cmap="viridis")

    # Afficher la légende et le graphe
    plt.legend()
    plt.title("Frontières de Décision - Régression Logistique Multiclasse")
    plt.show()