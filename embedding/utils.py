import scipy.sparse.linalg as sLa
import numpy as np
import scipy.sparse as sp 
from time import time

def spectral_radius(adj):
    """
    calculating the spectral raidus of a square matrix,
    which is the largest abosolute value  of its eigenvalues

    parameters:
    ------------
    adj - adjacency matrix, sparse coo_matrix
    """

    vals, _ = sLa.eigs(adj)
    return np.amax(np.abs(vals))

def gsvd(A, d=100, const_b=0.8):
    """
    generalised svd - using kats index

    parameters
    --------------------
    A - adjacency matrix
    d - dimension
    """
    #s_r is spectral radius of the adjacency matrix
    s_r = spectral_radius(A)
    #b is the decay parameter beta
    b = const_b / s_r

    print("spectral radius: {0:.3f}".format(s_r))
    print("beta: {0:.3f}".format(b))
    print("------------------------------------")

    start = time()

    #identity matrix
    I = sp.eye(A.shape[0]).tocsc()
    #Kats index - S = inverse(Ma) * Mb
    Ma = I - b * A
    Mb = b * A

    #similarity matrix
    S = sLa.inv(Ma).multiply(Mb)

    #decomposition
    u, s, v = sLa.svds(S, k=d//2)

    end = time()

    print("GSVD: {0:.3f}s".format(end-start))
    print("------------------------------------")

    return u, s, v
    


