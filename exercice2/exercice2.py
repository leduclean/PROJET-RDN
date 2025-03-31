import numpy as np
import matplotlib.pyplot as plt 
from visualisation.visualizer import Visualizer

# Create an instance of the Visualizer class for plotting
viz = Visualizer()

#Question 1  
# Si on cherche à minimiser J, alors on cherche à maximiser les probabilités y_{nk}\in [0,1] car ln(1) = 0 alors que si y_{nk} proche de 0
# ln(y_{nk}) devient très petit.
# Ce qui veut dire qu'on cherche la plus grande probabibilité de y_{nk}, donc la plus grande probabilité à réaliser une bonne classification
#%% Introduction: Data Reading Function

def readdataset2d(fname):
    with open(fname, "r") as file:
        X, T = [], []
        for l in file:
            x = l.strip().split()
            X.append((float(x[0]), float(x[1])))
            T.append(int(x[2]))
        T = np.reshape(np.array(T), (-1,1)) 
    return np.array(X), T

#%% Function for converting labels to one-hot encoding
def convertit(T: np.array, K: int) -> np.array:
    new_T = np.zeros((len(T), K))  
    for i, classification in enumerate(T):
        new_T[i, classification - 1] = 1 
    return new_T

#%% Softmax Function for multiclass classification
def softmax(A: np.array) -> np.array:
    B = []
    for ligne in A:
        B.append(1/np.sum(np.exp(ligne))*np.exp(ligne))
    return np.array(B)

#%% Prediction Functions
def predit_proba(X: np.array, W: np.array, b: np.array):
    return softmax(X.dot(W) + b)

def predit_classe(Y: np.array, K: int) -> np.array:
    prediction_index = np.argmax(Y, axis = 1) + 1
    return convertit(prediction_index, K)

#%% Initialization of parameters for Logistic Regression
def initialise(D: int, K: int) -> tuple[np.array, np.array, np.array, np.array]:    
    W = np.random.uniform(-2, 2, size = (D, K))
    b = np.random.randn(1, K)
    Y = predit_proba(X_train, W, b)
    C = predit_classe(Y, K)
    return W, b, Y, C

#%% Cross-Entropy Loss Function
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

#%% Update Weights and Biases for Logistic Regression
def updateWb(W: np.array, b: np.array, X: np.array, Y: np.array, T: np.array, lr: float):
    W -= lr*(X.transpose()).dot((Y-T))
    for j in range(0, K):
        grad = 0
        for n in range(N):
            grad += np.sum(Y - T, axis = 1)[n]
        b[0, j] -= lr*grad

#%% Accuracy Calculation Function
def taux_precision(C: np.array, T: np.array):
    N = T.shape[0]
    well_classified = 0
    for i in range(N):
        if np.argmax(C[i]) == np.argmax(T[i]):
            well_classified += 1
    return well_classified*100/N

#%% Logistic Regression Function
def regression_logistique(W, b, X, Y, T, lr=0.1, nb_iter=1000, int_affiche=100, quiet = False ):
    suite_erreur = [(0, cross_entropy(Y,T))]
    suite_precision = [taux_precision(predit_classe(Y, K), T)]
    for i in range(1, nb_iter + 1):
        updateWb(W,b,X,Y,T,lr)
        Y[:] = predit_proba(X, W, b)
        if i % int_affiche == 0:
            erreur_iter = cross_entropy(Y, T)
            precision_iter = taux_precision(predit_classe(Y, K), T)
            if not quiet:
                print("Cross-entropy error at iteration ", i ," : " , erreur_iter)
                print("Accuracy at iteration ", i ," : " , precision_iter)
            suite_erreur.append((i, erreur_iter))
            suite_precision.append((i, precision_iter))
    return suite_erreur

#%% Import dataset for Logistic Regression - 4 Classes
# Uncomment the lines below to use the 4-class dataset for Logistic Regression
# X_train, T_train = readdataset2d("exercice2/probleme_4_classes")
# N, D = X_train.shape
# K = 4
# T_conserve = T_train.copy()
# W, b, Y_train, C_train_init = initialise(D, K)
# T_train = convertit(T_train, K)
# W_init = W.copy()
# b_init = b.copy()
# viz.plot_error(regression_logistique(W, b, X_train, Y_train, T_train, lr=0.01))

#%% Import dataset for Logistic Regression - 5 Classes
# Uncomment the lines below to use the 5-class dataset for Logistic Regression
# X_train, T_train = readdataset2d("exercice2/probleme_5_classes")
# N, D = X_train.shape
# K = 5
# T_conserve = T_train.copy()
# W, b, Y_train, C_train_init = initialise(D, K)
# T_train = convertit(T_train, K)
# W_init = W.copy()
# b_init = b.copy()
# viz.plot_error(regression_logistique(W, b, X_train, Y_train, T_train, 0.05, 1000, 100))
# viz.plot_decision_boundaries(W, b, X_train, T_conserve)
# C_final = predit_classe(Y_train, K)

#%% Import dataset for Logistic Regression - Harder 5-Class Problem
# Uncomment the lines below to use the more difficult 5-class dataset for Logistic Regression
# X_train, T_train = readdataset2d("exercice2/probleme_5_plus_difficile")
# N, D = X_train.shape
# K = 5
# T_conserve = T_train.copy()
# W, b, Y_train, C_train_init = initialise(D, K)
# T_train = convertit(T_train, K)
# viz.plot_error(regression_logistique(W, b, X_train, Y_train, T_train, lr=0.01))
# viz.plot_decision_boundaries(W, b , X_train, T_train)

#%% Neural Network for Harder 5-Class Problem

# Here we use a dense neural network (with multiple hidden layers) for the more difficult 5-class problem.
# Comment out this section if you want to test the Logistic Regression instead.
X_train, T_train = readdataset2d("exercice2/probleme_5_plus_difficile")
N, D = X_train.shape
K = 5
T_conserve = T_train.copy()
# Definition of the Sigmoid Activation Function
def sigma(x: float) -> float:
    return 1 / (1 + np.exp(-x))

# Neural Network Prediction with multiple hidden layers
def predit_proba_dense(parameters: list) -> np.array:
    W, b = parameters[0]
    datas = [sigma(X_train.dot(W) + b)]  # Sigmoid for hidden layers
    for i in range(1, len(parameters)):
        W, b = parameters[i]
        Wprev = datas[-1]
        if i == len(parameters) - 1:  # Final layer
            datas.append(softmax(Wprev.dot(W) + b))
        else:
            datas.append(sigma(Wprev.dot(W) + b))
    return datas

# Create parameters for the neural network
def create_parameters(dimensions: list) -> list:
    parameters = []
    for indice in range(len(dimensions) - 1):
        W = np.random.uniform(-2, 2, size = (dimensions[indice], dimensions[indice + 1]))
        b = np.random.randn(1, dimensions[indice+1])
        parameters.append((W, b))
    return parameters

# Initialization for the Neural Network
def initialise(dimensions: list, K) -> tuple [list, np.array, np.array]:
    parameters = create_parameters(dimensions)
    datas = predit_proba_dense(parameters)
    C = predit_classe(datas[-1], K)
    return parameters, datas, C

# Backpropagation for updating weights and biases in the Neural Network
def updateWb(X: np.array, parameters: list, datas: list, T: np.array, lr: float):
    Y = datas[-1]  # Output after softmax
    delta = Y - T  # Initial gradient (softmax derivative)

    # Update the last layer
    W_last, b_last = parameters[-1]
    Z_prev = datas[-2] if len(datas) >= 2 else X  # Activation of the previous layer
    parameters[-1] = (
        W_last - lr * np.dot(Z_prev.T, delta),  # Update W
        b_last - lr * np.sum(delta, axis=0, keepdims=True)  # Update b
    )

    # Backpropagation for hidden layers (sigmoid)
    for i in range(len(parameters) - 2, -1, -1):
        W_curr, b_curr = parameters[i]
        Z_prev = X if i == 0 else datas[i - 1]
        Z_curr = datas[i]

        # Sigmoid derivative for hidden layers
        delta = np.dot(delta, parameters[i + 1][0].T) * (Z_curr * (1 - Z_curr))

        # Update the parameters of layer i
        parameters[i] = (
            W_curr - lr * np.dot(Z_prev.T, delta),
            b_curr - lr * np.sum(delta, axis=0, keepdims=True)
        )

# Neural Network Training Function
def reseau_dense(X: np.array, parameters: list, datas: list, T: np.array, K, lr=0.001, nb_iter=1000, int_affiche=100, quiet=False):
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
                print("Cross-entropy error at iteration ", i, " : " , erreur_iter)
                print("Accuracy at iteration ", i, " : " , precision_iter)
            suite_erreur.append((i, erreur_iter))
            suite_precision.append((i, precision_iter))
    return suite_erreur, suite_precision

# Define neural network dimensions (First and last values must remain fixed)
dimensions = [2, 6 ,2 , 5]
T_train = convertit(T_train, K)
parameters, datas, C_train_init = initialise(dimensions, K)

# Train the neural network and track error and precision
suite_erreur, suite_precision = reseau_dense(X_train, parameters, datas, T_train, K)

# Final predictions after training
Y_fin = datas[-1]
C_train_final = predit_classe(Y_fin, K)
W_fin, b_fin = parameters[-1]

# Uncomment below to visualize results:
# viz.plot_precision(suite_precision, label=f'Architecture {dimensions}')
# viz.plot_error(suite_erreur, label=f'Architecture {dimensions}')

# ! Uncomment below to plot decision boundaries only if the second-to-last dimension is 2
# ! If the last parameter dimension is 2 x K, decision boundary plotting is possible.
# viz.plot_decision_boundaries(W_fin, b_fin, X_train, T_conserve)

