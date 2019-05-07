import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sLa
from .utils import spectral_radius
from .graph_embedding import GraphEmbedding




class HPPE(GraphEmbedding):

    def __init__(self, file, type='undirected'):
        super().__init__(file, type=type)
        X, self.Y, Z = self.data_split(self.G)
        self.A = self.build_adj(np.array(X))
        #divide the growing set into 10 subsets
        step = len(Z) // 10
        self.growing = [Z[i:i+step] for i in range(0, len(Z), step)]

    def data_split(self, graph):
        #traning set
        X = []
        #testing set
        Y = []
        #growing set
        Z = []

        for edge in graph:
            #p is the probability
            p = np.random.uniform()
            if p <= 0.6:
                X.append(tuple(edge))
            elif p > 0.6 and p <= 0.8:
                Z.append(tuple(edge))
            else:
                Y.append(tuple(edge))

        return X, Y, Z

    def decay_param(self, A, const_b=0.8):
        # s_r is spectral radius of the adjacency matrix
        s_r = spectral_radius(A)
        # b is the decay parameter beta
        self.b = const_b / s_r

    def static_embedding(self, A, d=100):
        self.decay_param(A)
        # identity matrix
        I = sp.eye(A.shape[0]).tocsc()
        # Kats index - S = inverse(Ma) * Mb
        Ma = I
        Mb = A.multiply(A)

        # similarity matrix
        S = sLa.inv(Ma) @ (Mb)

        # decomposition
        l, s, r = sLa.svds(S, k=d)

        U = np.dot(l, np.diag(np.sqrt(s)))

        return U

    def embed(self, d=100):
        U = self.static_embedding(self.A)

        for change in self.growing:
            #dA is the change of adjacency matrix
            dA = self.build_adj(np.array(change))
            dU = self.static_embedding(dA)

            U = U + dU

        self.U = U

    def get_edge_weight(self, i, j):
        return np.dot(self.U[i, :], self.U[j, :])

    def get_testing_data(self):
        return self.Y.copy()






