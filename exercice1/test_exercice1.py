import numpy as np

from exercice1 import sigma, predit_classe, taux_precision, initialise, reseau


def test_sigma():
    ref_x = np.random.uniform(low=-20, high=20, size=5)
    ref_y = 1 / (1 + np.exp(-ref_x))
    assert np.allclose(sigma(ref_x), ref_y)
    return


def test_predit_classe():
    ref_x = np.random.uniform(low=0, high=1, size=5)
    ref_y = np.round(ref_x)
    assert np.array_equal(ref_y, predit_classe(ref_x))
    return


def test_taux_precision():
    C = np.array([[0], [1], [1], [0]])
    T = np.array([[0], [1], [0], [0]])
    assert taux_precision(C, T) == 75


# def test_reseau():
#     # Test avec un petit réseau
#     dimensions = [2, 3, 1]
#     parameters, datas, C_init = initialise(dimensions)

#     # Création de données de test
#     X_teste = np.random.randn(10, 2)
#     T_teste = np.random.randint(0, 2, size=(10, 1))


#     # Entraînement
#     suite_erreur, suite_precision = reseau(X_teste, parameters, datas, T_teste)
#     C_train_final = predit_classe(datas[-1])
#     print("taux de precision:",taux_precision(C_train_final,T_teste))
#     print("erreur:", suite_erreur)
#     print("precision:", suite_precision)
#     return

test_sigma()
test_predit_classe()
test_taux_precision()
# test_reseau()
