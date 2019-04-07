from embedding.hope import HOPE
from evaluation.metrics import pak
from evaluation.visualise import results_plot
from evaluation.evaluations import link_predict

file = 'data/fb.csv'
embedding = HOPE(file, type='directed')
embedding.embed()
pred = link_predict(embedding)
actual = embedding.get_testing_data()
result = []
x = []
for i in range(1, 10):
    result.append(pak(actual, pred, k=i))
    x.append('$10^' + str(i) + '$')

print(result)
results_plot(x, result, 'Link prediction')
