from .graph_embedding import GraphEmbedding
from .utils import gsvd
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

    def embed(self):
        u, s, v = gsvd(self.A)
        #embedding matrix U and U'
        self.U = np.dot(u, np.diag(np.sqrt(s)))
        self.V = np.dot(v.T, np.diag(np.sqrt(s)))
        
    def get_edge_weight(self, i, j):
        return np.dot(self.U[i, :], self.V[j, :])

    def get_testing_data(self):
        return self.Y.copy()
