import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sLa
from .utils import spectral_radius
from .graph_embedding import GraphEmbedding




class DHPE(GraphEmbedding):

    def __init__(self, file, type='undirected'):
        super().__init__(file, type=type)

    def decay_param(self, A, const_b=0.8):
        # s_r is spectral radius of the adjacency matrix
        s_r = spectral_radius(A)
        # b is the decay parameter beta
        return const_b / s_r

    def static_embedding(self, d=100):
        b = self.decay_param(self.A)
        # identity matrix
        I = sp.eye(A.shape[0]).tocsc()
        # Kats index - S = inverse(Ma) * Mb
        Ma = I - b * A
        Mb = b * A

        # similarity matrix
        S = sLa.inv(Ma).multiply(Mb)

        # decomposition
        l, s, r = sLa.svds(S, k=d)
        U = np.dot(l, np.diag(np.sqrt(s)))

        return l, s, r, Ma, Mb, U, b

    def dynamic_embedding(self, l, s, r, Ma, Mb, A, d, U, Sigma, Vl, Vr, Fa, Fb):
        l = np.dot(s, np.sign(l, r.T))
        L = np.diag(l)

        # dMa is change of Ma, dMb is change of Mb
        dMa = -b * A
        dMb = b * A

        Ha =
        Hb

    def embed(self):
        l, s, r, Ma, Mb, U, b = self.static_embedding()
        Sigma = np.diag(s)

        #dA is change of adjacency matrix
        dA = self.build_adj();

        Fa = np.dot()




