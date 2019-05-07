from embedding.dhpe import DHPE
from evaluation.metrics import pak
from evaluation.visualise import results_plot
from evaluation.evaluations import link_predict

file = 'data/fb.csv'

embedding = DHPE(file, type='undirected')
embedding.embed()

pred = link_predict(embedding)
actual = embedding.get_testing_data()
result = []
for i in range(2, 8):
    result.append(pak(actual, pred, k=i))

x = ['$10^' + str(i) + '$' for i in range(2, 8)]
results_plot(x, result, 'Facebook Dataset')