import matplotlib.pyplot as plt

def results_plot(x, y, title):
	#plot x and y
	plt.plot(x, y)
	#title of the plot
	plt.title(title)
	#x and y labels
	plt.ylabel('Precision@k')
	plt.xlabel('k')
	
	plt.show()