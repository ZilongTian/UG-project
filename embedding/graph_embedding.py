from abc import ABC
import numpy as np
from scipy import sparse as sp

class GraphEmbedding(ABC):
	"""
	graph embedding methods

	attributes
	-------------------------
	G - graph 
	V - number of nodes
	N - max node 
	E - number of edges
	type - type of graph, directed or undirected
	"""

	def __init__(self, file, type='directed'):
		#import data
		self.G = np.loadtxt(file, delimiter=',', dtype=int)
		#number of nodes
		self.V = self.number_of_nodes(self.G)
		print('number of nodes: ', self.V)
		#max node number
		self.N = self.max_node(self.G)
		#number of edges
		self.E = self.number_of_edges(self.G)
		print('number of edges: ', self.E)

		print("------------------------------------")
		
		#property of graph
		self.type = type

	def number_of_edges(self, graph):
		return graph.shape[0]

	def number_of_nodes(self, graph):
		"""
		calculating the number of nodes of this graph
		"""
		return len(set(graph[:, 0]) | set(graph[:, 1]))

	def max_node(self, graph):
		"""
		working out thee biggest id of node
		"""
		return max(max(graph[:, 0]), max(graph[:, 1]))

	def build_adj(self, graph):
		n = self.N + 1
		E = self.number_of_edges(graph)
		if self.type == 'directed':
			print('directed')
			i = graph[:, 0]
			j = graph[:, 1]
			A = sp.coo_matrix((np.ones(E), (i,j)), shape=(n,n)).asfptype()
		else:
			print('undirected')
			i = np.concatenate((graph[:, 0], graph[:,1]), axis=0)
			j = np.concatenate((graph[:, 1], graph[:,0]), axis=0)
			A = sp.coo_matrix((np.ones(E*2),(i,j)),shape=(n,n)).asfptype()

		return A

	def data_split(self, graph):
		pass

