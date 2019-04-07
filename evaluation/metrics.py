
def pak(actual, predicted, k=1):
	"""
	precision at k - to measure the performance of link prediction
	"""
	k = 10 ** k

	if len(predicted) > k:
		predicted = predicted[:k]

	return len(set(actual) & set(predicted)) / len(predicted)