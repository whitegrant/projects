#Grant White
#November 2019
"""
Singular Value Decomposition (SVD)
"""

import numpy as np
from scipy import linalg as la
import math
from matplotlib import pyplot as plt


def compact_svd(A, tol=1e-6):
    """Computes the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    #calculate singulat values and eigenvectos
    lamda, V = np.linalg.eig(A.conj().T @ A)
    sigma = np.sqrt(lamda)

    #keep only non-zero singular values and corresponding eigenvectors
    sigma_1 = []
    V_1 = []
    order = sigma.argsort()
    for i in reversed(order):
        if sigma[i] > tol:
            sigma_1.append(sigma[i])
            V_1.append(V[:,i])
        else:
            break   #the rest should be zeros
    V_1 = np.array(V_1).T

    #construct U
    U_1 = A @ V_1 / sigma_1

    sigma_1 = np.array(sigma_1)

    return U_1, sigma_1, V_1.conj().T


def visualize_svd(A):
    """Plots the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    #calculate S and E
    circle = np.linspace(0, 2*math.pi, 200)
    S = np.array([[math.sin(theta), math.cos(theta)] for theta in circle]).T
    E = np.array([[1, 0, 0], [0, 0, 1]])

    #calculate the SVD
    U, s, Vh = la.svd(A)
    s = np.diag(s)

    #plot
    plt.subplot(221)
    plt.plot(S[0], S[1])
    plt.plot(E[0], E[1])
    plt.axis('equal')

    S = Vh@S
    E = Vh@E

    plt.subplot(222)
    plt.plot(S[0], S[1])
    plt.plot(E[0], E[1])
    plt.axis('equal')

    S = s@S
    E = s@E

    plt.subplot(223)
    plt.plot(S[0], S[1])
    plt.plot(E[0], E[1])
    plt.axis('equal')

    S = U@S
    E = U@E

    plt.subplot(224)
    plt.plot(S[0], S[1])
    plt.plot(E[0], E[1])
    plt.axis('equal')

    plt.show()


def svd_approx(A, s):
    """Returns the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    #check s
    if s > np.linalg.matrix_rank(A):
        raise ValueError("s > rank(A)")

    #calculate A_s
    U, S, Vh = la.svd(A, full_matrices=False)
    U_hat = U[:,:s]
    S_hat = np.diag(S[:s])
    Vh_hat = Vh[:s]

    A_s = U_hat @ S_hat @ Vh_hat

    #get number of entries
    num = U_hat.size + S[:s].size + Vh_hat.size
    
    return A_s, num


def lowest_rank_approx(A, err):
    """Returns the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    #calculate SVD of A
    U, S, Vh = la.svd(A, full_matrices=False)

    #check if A can be approximated
    if err <= S[-1]:
        raise ValueError("A cannot be approximated.")

    #find lowest rank aproximation of A
    s = np.argmax(np.where(S > err, 0, S))
    return svd_approx(A, s)


# #check compact_svd
# A = np.random.random((10,5))
# u,s,vh = la.svd(A, full_matrices=False)
# U, S, Vh = compact_svd(A)
# if (np.allclose(u, U) and np.allclose(s, S) and np.allclose(vh, Vh)):
#     print(True)
# else:
#     print(False)

# #check visualize_svd
# A = np.array([[3, 1], [1, 3]])
# visualize_svd(A)

# #check svd_approx
# A = np.random.rand(20,8)
# s = 8
# print(svd_approx(A, s))
# print(np.linalg.matrix_rank(A))

# #check lowest_rand_approx
# A = np.random.rand(20,8)
# e = 5
# print(lowest_rank_approx(A, e))
