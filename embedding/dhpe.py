import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sLa
from .utils import spectral_radius
from .graph_embedding import GraphEmbedding




class DHPE(GraphEmbedding):

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

    def static_embedding(self, d=100):
        self.decay_param(self.A)
        # identity matrix
        I = sp.eye(self.A.shape[0]).tocsc()
        # Kats index - S = inverse(Ma) * Mb
        Ma = I - self.b * self.A
        Mb = self.b * self.A

        # similarity matrix
        S = sLa.inv(Ma) @ (Mb)

        # decomposition
        l, s, r = sLa.svds(S, k=d)

        U = np.dot(l, np.diag(np.sqrt(s)))

        return l, s, r, Ma, Mb, U

    def dynamic_embedding(self, d, dA, L, X, Fa, Fb):
        # dMa is change of Ma, dMb is change of Mb
        dMa = -self.b * dA
        dMb = self.b * dA

        Ha = self.get_HF(dMa, X)
        Hb = self.get_HF(dMb, X)
        for i in range(0, d):
            #change of eigenvalues
            dl = (Hb[i, i] - L[i, i]*Ha[i, i])/Fa[i, i]
            L[i, i] += dl
            B = Hb - L[i, i] * Ha - dl * Fa
            W = L[i, i] * Ha - Hb + L[i, i] * Fa - Fb
            a = np.linalg.inv(W) * B
            X[i] = np.dot(a[i, i], X[i, :])

        return L, X


    #calculate H and F.
    def get_HF(self, M, X):
        return X.T @ M @ X

    def singular_to_eigen(self, Vl, s, Vr):
        l = [np.dot(s[i], np.sign(np.dot(Vl[:, i], Vr[i, :]))) for i in range(0, s.shape[0])]
        return np.diag(np.array(l)), Vl

    def eigen_to_singular(self, L, X):
        singular_vals = np.abs(L)
        return X, singular_vals

    def embed(self, d=100):
        #Vl-left singular vector, Vr-right singular vector, s-singular values, U-embedding matrix
        Vl, s, Vr, Ma, Mb, U = self.static_embedding()

        # singular values
        Sigma = np.diag(s)

        # transform singular to eigenvalue problems
        L, X = self.singular_to_eigen(Vl, s, Vr)
        #
        for change in self.growing:
            #dA is the change of adjacency matrix
            dA = self.build_adj(np.array(change))
            Fa = self.get_HF(Ma, X)
            Fb = self.get_HF(Mb, X)
            nL, nX = self.dynamic_embedding(d, dA, L, X, Fa, Fb)

            X, s = self.eigen_to_singular(nL, nX)

            U = U + np.dot(X, np.sqrt(s))

        self.U = U

    def get_edge_weight(self, i, j):
        return np.dot(self.U[i, :], self.U[j, :])

    def get_testing_data(self):
        return self.Y.copy()






