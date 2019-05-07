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

    


