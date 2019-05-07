from .graph_embedding import GraphEmbedding
from .utils import spectral_radius
import scipy.sparse as sp
import scipy.sparse.linalg as sLa
import numpy as np

class HOPE(GraphEmbedding):

    def __init__(self, file, type='directed'):
        super().__init__(file, type=type)
        X, self.Y = self.data_split(self.G)
        self.A = self.build_adj(np.array(X))

    def data_split(self, graph, ratio=0.8):
        #traning set
        X = []
        #test set
        Y = []
        for edge in graph:
            if np.random.uniform() <= ratio:
                X.append(tuple(edge))
            else:
                Y.append(tuple(edge))

        return X, Y

    def embed(self, d=100, const_b=0.8):
        s_r = spectral_radius(self.A)
        b = const_b / s_r
        # identity matrix
        I = sp.eye(self.A.shape[0]).tocsc()
        # Kats index - S = inverse(Ma) * Mb
        Ma = I - b * self.A
        Mb = b * self.A

        # similarity matrix
        S = sLa.inv(Ma) @ (Mb)

        # decomposition
        l, s, r = sLa.svds(S, k=d)

        self.U = np.dot(l, np.diag(np.sqrt(s)))
        self.V = np.dot(r.T, np.diag(np.sqrt(s)))
        
    def get_edge_weight(self, i, j):
        return np.dot(self.U[i, :], self.U[j, :])

    def get_testing_data(self):
        return self.Y.copy()
