import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

class Visualiseur:
    def __init__(self, nom_carte="viridis"):
        """
        Initialise le visualiseur avec une carte de couleurs par défaut.

        Paramètres:
            nom_carte (str): Nom de la carte de couleurs à utiliser (par défaut "viridis").
        """
        self.carte = cm.get_cmap(nom_carte)
        
    def tracer_progression_entrainement(self, donnees_erreur: list, donnees_precision: list) -> None:
        """
        Affiche côte à côte les courbes d'erreur et de précision durant l'entraînement.

        Paramètres:
            donnees_erreur (list): Liste de tuples (itération, valeur d'erreur).
            donnees_precision (list): Liste de tuples (itération, valeur de précision).
        """
        fig, (axe1, axe2) = plt.subplots(1, 2, figsize=(12, 5))

        iterations_erreur, erreurs = zip(*donnees_erreur)
        iterations_precision, precisions = zip(*donnees_precision)

        axe1.plot(iterations_erreur, erreurs, label="Erreur de cross-entropie", color='red')
        axe1.set_xlabel("Itérations")
        axe1.set_ylabel("Erreur")
        axe1.set_title("Erreur d'entraînement")
        axe1.legend()

        axe2.plot(iterations_precision, precisions, label=f"Précision (final = {precisions[-1]})", color='green')
        axe2.set_xlabel("Itérations")
        axe2.set_ylabel("Précision (%)")
        axe2.set_title("Précision d'entraînement")
        axe2.legend()

        plt.show()

    def tracer_frontieres_decision(self, poids: np.array, biais: np.array, X: np.array, T: np.array, architecure: list = None):
        """
        Affiche les frontières de décision et les points de données avec une carte de couleurs associée à chaque classe.

        Paramètres:
            poids (np.array): Matrice des poids du modèle.
            biais (np.array): Vecteur des biais du modèle.
            X (np.array): Matrice des caractéristiques.
            T (np.array): Étiquettes ou classes pour le nuage de points.
        """
        # Définition de l'intervalle pour l'axe des x
        valeurs_x = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        
        # Nombre de classes (K) d'après la matrice des poids
        K = poids.shape[1]
        
        carte = self.carte  
        couleurs = [carte(k / (K - 1)) for k in range(K)]  

        # Tracer la frontière de décision pour chaque classe (un contre tous)
        for k in range(K):
            # Récupérer les poids et le biais pour la classe k
            p1, p2 = poids[:, k]
            b_k = biais[0, k]
            
            # Calculer la frontière de décision pour la classe k
            valeurs_y = -(p1 * valeurs_x + b_k) / p2
            
            # Tracer la frontière de décision avec la couleur correspondante
            plt.plot(valeurs_x, valeurs_y, color=couleurs[k], label=f"Frontière classe {k}")
        
        # Tracer les points de données avec une couleur selon leur classe
        plt.scatter(X[:, 0], X[:, 1], c=T, s=30, cmap=carte)
        
                # Amélioration de l'affichage
        plt.xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
        plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max())
        plt.axis("equal")  # Échelle égale pour éviter l'écrasement
        plt.gca().set_aspect('auto')  

        plt.legend()
        plt.title(f"Frontières de décision - Reseau dense d'architecture: {architecure} " if architecure else "Frontières de décision - Régression logistique multiclasses")
        plt.show()
        
    def tracer_multiple(self, liste_courbes: list, dimensions: list = [], a_afficher: str = "error", modulation_taux: np.ndarray = None,  nom_carte="viridis") -> None:
        """
        Affiche plusieurs courbes sur un même graphique.

        Paramètres:
            liste_courbes (list): Liste des courbes à afficher, où chaque courbe est une liste de tuples (itération, valeur).
            a_afficher (str, optionnel): Type de courbe à afficher ("error" ou "precision", par défaut "error").
            modulation_taux (np.ndarray, optionnel): Tableau des taux d'apprentissage correspondant aux courbes (par défaut None).
            nom_carte (str, optionnel): Nom de la carte de couleurs utilisée pour les courbes (par défaut "viridis").
        """
        fig, axe = plt.subplots(figsize=(8, 6))
        nb_courbes = len(liste_courbes)
        couleurs = plt.cm.get_cmap(nom_carte, nb_courbes)  # Récupération de la carte de couleurs      
        
        # Pour chaque courbe dans la liste
        for i, courbe in enumerate(liste_courbes):
            # Séparation des itérations et des valeurs
            iterations, valeurs = zip(*(courbe[0] if a_afficher == "error" else courbe[1]))  
            
            # Vérifier si l'on doit afficher les courbes pour différents taux d'apprentissage
            if modulation_taux is not None and modulation_taux.size > 0:
                axe.plot(iterations, valeurs,
                          color=couleurs(i), 
                          label=f"lr: {modulation_taux[i]:.6f}, t_final ={valeurs[-1]}" if a_afficher != "error" else f"Architecture: {dimensions[i]}, e_final = {valeurs[-1]}" )
            else:
                axe.plot(iterations, valeurs, 
                        color=couleurs(i),  
                        label=f"Architecture: {dimensions[i]}, t_final = {valeurs[-1]}" if a_afficher != "error" else f"Architecture: {dimensions[i]}, e_final = {valeurs[-1]}" )
        
        # Configuration des axes et du titre en fonction du type de courbe à afficher
        if a_afficher == "error":
            axe.set_xlabel("Itérations")
            axe.set_ylabel("Erreur de cross-entropie")
            if modulation_taux is not None and modulation_taux.size > 0:
                 axe.set_title("Comparaison des erreurs pour différents taux d'apprentissage")
            else:
                axe.set_title("Comparaison des erreurs pour différentes architectures")
        elif a_afficher == "precision":
            axe.set_xlabel("Itérations")
            axe.set_ylabel("Taux de precision")
            if modulation_taux is not None and modulation_taux.size > 0:
                 axe.set_title("Comparaison des taux de precision pour différents taux d'apprentissage")
            else:
                axe.set_title("Comparaison des taux de precision pour différentes architectures")
        
        axe.legend()
        plt.show()
