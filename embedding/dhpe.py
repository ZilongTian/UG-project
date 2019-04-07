from .graph_embedding import GraphEmbedding
from .utils import gsvd
import numpy as np

class DHPE(GraphEmbedding):

	def __init__(self, file, type='undirected'):
		super().__init__(file, type=type)

		self.A = self.build_adj(np.array(X))

	def data_split(self, graph):
		#traning set
        X = []
        #test set
        Y = []
        # growing set
        Z = []

        for edge in graph:
            if np.random.uniform() <= ratio:
                X.append(tuple(edge))
            else:
                Y.append(tuple(edge))

        return X, Y

    def embed(self):
    	u, s, v = gsvd(self.A)

    	#embedding matrix U and U'
	    U = np.dot(u, np.diag(np.sqrt(s)))
	    V = np.dot(v.T, np.diag(np.sqrt(s)))
    	