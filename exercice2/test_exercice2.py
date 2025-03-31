import numpy as np
import matplotlib.pyplot as plt
from exercice2 import convertit, softmax, cross_entropy, taux_precision, initialise, predit_classe, reseau

def test_convertit():
    T = np.array([[1], [2], [3], [1]])
    K = 3
    converted = convertit(T, K)
    expected = np.array([[1,0,0], [0,1,0], [0,0,1], [1,0,0]])
    assert np.array_equal(converted, expected), "Conversion incorrecte"

def test_softmax():
    A = np.array([[1, 2, 3], [1, 2, 1]])
    sm = softmax(A)
    print(sm)
    
    assert np.allclose(sm, np.array([[0.09003057, 0.24472847, 0.66524096],[0.21194156, 0.57611688, 0.21194156]]))

import numpy as np

def test_predit_classe():
    # Test case 1: Single sample, single class (K=1)
    Y1 = np.array([[0.7]])  # 2D array even for single sample
    K1 = 1
    expected1 = np.array([[1]])
    result1 = predit_classe(Y1, K1)
    assert np.allclose(result1, expected1), f"Failed for {Y1}. Expected {expected1}, got {result1}"

    # Test case 2: Multiple samples, binary classification (K=2)
    Y2 = np.array([[0.1, 0.9], [0.8, 0.2], [0.6, 0.5]])
    K2 = 2
    expected2 = np.array([[0., 1.], [1., 0.], [1, 0]])  # argmax returns first max in case of tie
    result2 = predit_classe(Y2, K2)
    assert np.allclose(result2, expected2), f"Failed for {Y2}. Expected {expected2}, got {result2}"

    # Test case 3: Multiple classes (K=3)
    Y3 = np.array([[0.1, 0.2, 0.7], [0.4, 0.5, 0.1], [0.3, 0.3, 0.4]])
    K3 = 3
    expected3 = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1]])
    result3 = predit_classe(Y3, K3)
    assert np.allclose(result3, expected3), f"Failed for {Y3}. Expected {expected3}, got {result3}"

    print("All test cases for predit_classe() passed successfully!")
    return True

def test_cross_entropy():
    T = np.array([[1, 0], [0, 1]])
    Y_perfect = np.array([[1, 0], [0, 1]])
    assert cross_entropy(Y_perfect, T) == 0
    return

def test_taux_precision():
    # Test case 1: Basic binary classification
    C1 = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # Predictions (one-hot)
    T1 = np.array([[1, 0], [0, 1], [1, 0], [1, 0]])  # True labels (one-hot)
    expected1 = 75.0  # 3 correct out of 4
    result1 = taux_precision(C1, T1)
    assert np.isclose(result1, expected1), f"Basic case failed. Expected {expected1}%, got {result1}%"

    # Test case 2: Perfect classification
    C2 = np.array([[1, 0], [0, 1], [1, 0]])
    T2 = np.array([[1, 0], [0, 1], [1, 0]])
    assert taux_precision(C2, T2) == 100.0, "Perfect case failed"

    # Test case 3: All wrong
    C3 = np.array([[0, 1], [1, 0], [0, 1]])
    T3 = np.array([[1, 0], [0, 1], [1, 0]])
    assert taux_precision(C3, T3) == 0.0, "All wrong case failed"

    # Test case 4: Multi-class (3 classes)
    C4 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    T4 = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
    assert taux_precision(C4, T4) == 50.0, "Multi-class case failed"

    print("All taux_precision() tests passed successfully!")

# def test_reseau():
#     # Test avec un petit réseau
#     dimensions = [2, 3, 2]  # 2 entrées, 3 neurones cachés, 2 classes
#     K = 2
#     parameters, datas, C = initialise(dimensions, K)
    
#     # Création de données de test
#     X_test = np.random.randn(10, 2)
#     T_test = convertit(np.random.randint(1, K+1, size=(10, 1)), K)
    
#     # Entraînement
#     suite_erreur, suite_precision = reseau(X_test, parameters, datas, T_test, K, nb_iter=50, quiet=True)
    
#     print("erreur:", suite_erreur)
#     print("precision:", suite_precision)
#     return


test_convertit()
test_softmax()
test_predit_classe()
test_cross_entropy()
test_taux_precision()
# test_reseau()
