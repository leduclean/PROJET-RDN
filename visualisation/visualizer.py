import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

class Visualizer:
    def __init__(self, cmap_name="viridis"):
        """Initializes the visualizer with a default colormap."""
        self.cmap = cm.get_cmap(cmap_name)
    
    def plot_error(self, error_series: list, label='Cross-entropy error', color='blue'):
        """
        Displays the error curve.
        
        Parameters:
            error_series (list): List of tuples (iteration, error value).
            label (str): Label for the curve.
            color (str): Color of the curve.
        """
        # Extraction des itérations et des valeurs d'erreur
        iterations, errors = zip(*error_series)
        fig, ax = plt.subplots()
        ax.plot(iterations, errors, color=color, label=label)
        ax.set_title("Cross-entropy Error")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Error")
        ax.legend()
        plt.show()

        
    def plot_precision(self, precision_sequence: list, label, color='blue'):
        """
        Displays the precision curve.
        
        Parameters:
            precision_sequence (list): List of precision values over iterations.
            label (str): Label to display on the curve.
            color (str): Color of the curve.
        """
        iterations, precisions = zip(*precision_sequence)
        fig, ax = plt.subplots()
        ax.plot(iterations, precisions, color=color, label=label)
        ax.set_title("Precision Rate")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Rate")
        ax.legend()
        plt.show()
        
    def plot_decision_boundaries(self, W: np.array, b: np.array, X: np.array, T: np.array):
        """
        Displays decision boundaries and data points with a colormap assigned to each class.
        
        Parameters:
            W (np.array): Model weight matrix.
            b (np.array): Model bias.
            X (np.array): Feature matrix.
            T (np.array): Targets or classes for the scatter plot.
        """
        x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        
        # Number of classes (K) from the weight matrix
        K = W.shape[1]
        
        cmap = self.cmap  
        colors = [cmap(k / (K - 1)) for k in range(K)]  

        # Plot decision boundary for each class (one-vs-all)
        for k in range(K):
            # Get the weights and bias for the k class
            w1, w2 = W[:, k]
            b_k = b[0, k]
            
            # Calculate the decision boundary for the k class
            y_vals = -(w1 * x_vals + b_k) / w2
            
            # Plot the decision boundary with the corresponding color for the class
            plt.plot(x_vals, y_vals, color=colors[k], label=f"Boundary class {k}")
        
        # Plot the data points with colors corresponding to their class labels
        plt.scatter(X[:, 0], X[:, 1], c=T, s=30, cmap=cmap)  # Use colormap for classes
        
        # Add legend, title, and show the plot
        plt.legend()
        plt.title("Decision Boundaries - Multiclass Logistic Regression")
        plt.show()
        
    def plot_multiple(self, list_of_curves: list, to_display: str = "error", learning_rate_modulation: np.ndarray = None, cmap: str = "viridis") -> None:
        """
        Displays multiple curves on the same plot.
        
        Parameters:
            list_of_curves (list): List of curves to display, where each curve is a list of tuples (iteration, value).
            to_display (str, optional): Type of curve to display ("error" or "precision", default: "error").
            learning_rate_modulation (np.ndarray, optional): Array of learning rates corresponding to the curves (default: None).
            cmap (str, optional): Colormap used for the curves (default: "viridis").
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        n_curves = len(list_of_curves)
        colors = plt.cm.get_cmap(cmap, n_curves)  # Getting colormap      
        
        # for each curve in list_of_curves
        for i, curve_data in enumerate(list_of_curves):
            iterations, values = zip(*curve_data)  # Séparation des itérations et des valeurs avec zip
            
            # Vérifie si l'on doit afficher les courbes pour différents taux d'apprentissage
            if learning_rate_modulation is not None and learning_rate_modulation.size > 0:
                # Si learning_rate_modulation est fourni et non vide, utilise la valeur du taux d'apprentissage pour l'étiquette
                ax.plot(iterations, values, color=colors(i),  label=f"lr: {learning_rate_modulation[i]:.6f}")
            else:
                ax.plot(iterations, values, color=colors(i),  label=f"Curve {i+1}")
        
        # Mise en place des axes et du titre en fonction de la courbe choisie à afficher
        if to_display == "error":
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Cross-entropy Error")
            ax.set_title("Error Comparison for Different Learning Rates")
        elif to_display == "precision":
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Precision Rate")
            ax.set_title("Precision Rate Comparison")
        
        ax.legend()
        plt.show()
