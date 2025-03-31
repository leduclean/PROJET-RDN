import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from exercice3 import get_block_values, pooling_max, pooling_mean, pooling_median, convolution1D, convolution1D_padding, convolution1D_stride, cross_correlation1D, cross_correlation2D


test_img = np.array([
    [10, 20, 30, 40, 50, 60],
    [15, 25, 35, 45, 55, 65],
    [20, 30, 40, 50, 60, 70],
    [25, 35, 45, 55, 65, 75],
    [30, 40, 50, 60, 70, 80],
    [35, 45, 55, 65, 75, 85]
], dtype=np.uint8)



def test_get_block_values():
    
    # Test avec un ratio qui divise parfaitement
    blocks = get_block_values(test_img, 2, 3)
    assert blocks.shape == (3, 2, 2, 3), "Mauvaise shape des blocs"
    
    # Vérification du premier bloc
    expected_first_block = np.array([[10, 20, 30], [15, 25, 35]])
    assert np.array_equal(blocks[0,0], expected_first_block), "Premier bloc incorrect"
    
    # Test avec padding nécessaire
    blocks_padded = get_block_values(test_img, 4, 5)
    print("Shape des blocs avec padding (4x5):", blocks_padded.shape)
    assert blocks_padded.shape == (2, 2, 4, 5), "Mauvaise shape avec padding"
    
    print("Test get_block_values réussi!\n")

def test_pooling_functions():
    
    # Test pooling max
    pooled_max = pooling_max(test_img, 2, 2)
    expected_max = np.array([[25, 45, 65], [35, 55, 75], [45, 65, 85]])
    assert np.array_equal(pooled_max, expected_max), "Pooling max incorrect"
    
    # Test pooling mean
    pooled_mean = pooling_mean(test_img, 2, 2)
    expected_mean = np.array([[17.5, 37.5, 57.5], [27.5, 47.5, 67.5], [37.5, 57.5, 77.5]])
    assert np.allclose(pooled_mean, expected_mean), "Pooling mean incorrect"
    
    print("Test des fonctions de pooling réussi!\n")

# def test_convolution1D():
#     print("\n=== Test de convolution1D ===")
#     X = [1, 2, 3, 4, 5]
#     F = [1, 0, -1]
    
#     # Test convolution de base
#     conv = convolution1D(X, F)
#     expected = [1*1 + 2*0 + 3*(-1),  # 1-3 = -2
#                 2*1 + 3*0 + 4*(-1),  # 2-4 = -2
#                 3*1 + 4*0 + 5*(-1)]  # 3-5 = -2
#     assert conv == expected, "Convolution1D de base incorrecte"
    
#     # Test avec padding
#     conv_pad = convolution1D_padding(X, F)
#     expected_pad = [1*1 + 2*0 + 0*(-1),  # 1+0+0 = 1
#                     1*1 + 2*0 + 3*(-1),  # 1-3 = -2
#                     2*1 + 3*0 + 4*(-1),  # 2-4 = -2
#                     3*1 + 4*0 + 5*(-1),  # 3-5 = -2
#                     4*1 + 5*0 + 0*(-1)]  # 4+0+0 = 4
#     assert conv_pad == expected_pad, "Convolution1D avec padding incorrecte"
    
#     # Test avec stride
#     conv_stride = convolution1D_stride(X, F, 2)
#     expected_stride = [1*1 + 2*0 + 3*(-1),  # 1-3 = -2
#                        3*1 + 4*0 + 5*(-1)]  # 3-5 = -2
#     assert conv_stride == expected_stride, "Convolution1D avec stride incorrecte"
    
#     print("Test convolution1D réussi!\n")

def test_cross_correlation():
    print("\n=== Test de cross_correlation ===")
    X = [1, 2, 3, 4, 5]
    F = [1, 0, -1]
    
    # Test 1D
    corr = cross_correlation1D(X, F)
    expected = [1*1 + 2*0 + 3*(-1),  # 1-3 = -2
                2*1 + 3*0 + 4*(-1),  # 2-4 = -2
                3*1 + 4*0 + 5*(-1)]  # 3-5 = -2
    assert corr == expected, "Cross-correlation1D incorrecte"
    
    # Test 2D
    F_2d = np.array([[1, 0], [0, -1]])
    corr_2d = cross_correlation2D(test_img, F_2d)
    expected_corr_2d = [
        [10*1 + 20*0 + 15*0 + 25*(-1),  # 10-25 = -15
         20*1 + 30*0 + 25*0 + 35*(-1)], # 20-35 = -15
        [15*1 + 25*0 + 20*0 + 30*(-1),  # 15-30 = -15
         25*1 + 35*0 + 30*0 + 40*(-1)]  # 25-40 = -15
    ]
    # On ne compare que les premiers éléments pour simplifier
    assert corr_2d[0][0] == expected_corr_2d[0][0], "Cross-correlation2D incorrecte"
    
    print("Test cross_correlation réussi!\n")

def test_filtres_image():
    
    # Filtre moyenne
    filtre_moyenne = np.ones((3,3))/9
    filtered = cross_correlation2D(test_img, filtre_moyenne)
    print("Image filtrée (moyenne) shape:", np.array(filtered).shape)
    
    # Filtre contour
    filtre_contour = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
    filtered = cross_correlation2D(test_img, filtre_contour)
    print("Image filtrée (contour) shape:", np.array(filtered).shape)
    
    print("Test des filtres réussi (vérifiez visuellement)!\n")


test_get_block_values()
test_pooling_functions()
#test_convolution1D()
test_cross_correlation()
test_filtres_image()