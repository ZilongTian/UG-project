import numpy as np
import operator

def link_predict(embedding):	
	actual = embedding.get_testing_data()

	N = embedding.N

	pred = []
	for i in set(np.array(actual)[:, 0]):
		for j in range(0, N):
			weight = embedding.get_edge_weight(i, j)
			if weight > 0.0:
				pred.append((i, j, weight))

	#sort in decending order
	#pred.sort(key=operator.itemgetter(2))
	predicted = []
	for i,j,w in pred:
		predicted.append((i,j))

	#filter
	# actual = graph.get_train_data()
	# predicted = list(set(predicted) - set(actual))
	# print(len(predicted))
	return predicted