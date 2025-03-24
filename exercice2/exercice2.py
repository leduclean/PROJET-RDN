import numpy as np
import matplotlib.pyplot as plt 

#%% Introduction : Lecture des jeux de données fournis 
def readdataset2d(fname):
    with open(fname, "r") as file:
        X, T = [], []
        for l in file:
            x = l.strip().split()
            # print(x)
            X.append((float(x[0]), float(x[1])))
            T.append(int(x[2]))
        T = np.reshape(np.array(T), (-1,1)) 
    return np.array(X), T

#%% Fonction convertit de la question 2
def convertit(T: np.array, D: int) -> np.array:
    new_T= []
    for classification in T:
        line_i = np.zeros(D)
        line_i[classification - 1] = 1
        new_T.append(line_i)
    return np.array(new_T)
# print(convertit(np.array([1,2]), 2))

def softmax(A: np.array) -> np.array:
    B = []
    for ligne in A:
        B.append(1/np.sum(np.exp(ligne))*np.exp(ligne))
    return np.array(B)

def predit_proba(X: np.array, W: np.array, b: np.array):
    return softmax( X.dot(W) + b)

# idée -> parcourir la ligne associé à Xi dans Y et voir l'indice de la plus grande valeur et ensuite convertir comme avec T_train
def predit_classe(Y: np.array) -> np.array:
    prediction_index = np.argmax(Y, axis = 1) + 1
    return convertit(prediction_index, D)


    



#%% Import du jeu de données : probleme à 4 classes
X_train, T_train = readdataset2d("probleme_4_classes")
N, D = X_train.shape

# Pour la visualisation, on garde T_train sous sa forme originelle
plt.scatter(X_train[:,0], X_train[:,1], c=T_train, s = 30)
# plt.show()


# #%% Import du jeu de données : probleme à 5 classes
# X_train, T_train = readdataset2d("probleme_5_classes")
# N, D = X_train.shape

# # Pour la visualisation, on garde T_train sous sa forme originelle
# plt.scatter(X_train[:,0], X_train[:,1], c=T_train, s = 30)
# # plt.show()

#%% Import du jeu de données : probleme à 6 classes
X_train, T_train = readdataset2d("probleme_6_classes")
N, D = X_train.shape

# Pour la visualisation, on garde T_train sous sa forme originelle
plt.scatter(X_train[:,0], X_train[:,1], c=T_train, s = 30)
# plt.show()

