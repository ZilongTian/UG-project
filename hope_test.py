from embedding.hope import HOPE
from evaluation.metrics import pak
from evaluation.visualise import results_plot
from evaluation.evaluations import link_predict

file = 'data/BlogCatalog.csv'

# for i in range(0, 5):

embedding = HOPE(file, type='undirected')
embedding.embed()

pred = link_predict(embedding)
actual = embedding.get_testing_data()
result = []
for j in range(1, 7):
  result.append(pak(actual, pred, k=j))

x = ['$10^' + str(i) + '$' for i in range(1, 7)]
results_plot(x, result, 'BlogCatalog Dataset')
