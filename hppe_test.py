from embedding.hppe import HPPE
from embedding.hope import HOPE
from embedding.dhpe import DHPE
from evaluation.metrics import pak
from evaluation.visualise import results_plot
from evaluation.evaluations import link_predict
import matplotlib.pyplot as plt

file = 'data/fb.csv'
#set up for hppe
hppe = HPPE(file, type='undirected')
hppe.embed()

hppe_pred = link_predict(hppe)
hppe_actual = hppe.get_testing_data()
hppe_result = []

#set up for hope
hope = DHPE(file, type='undirected')
hope.embed()
hope_pred = link_predict(hope)
hope_actual = hope.get_testing_data()
hope_result = []

for i in range(2, 8):
    hppe_result.append(pak(hppe_actual, hppe_pred, k=i))
    hope_result.append(pak(hope_actual, hope_pred, k=i))

x = ['$10^' + str(i) + '$' for i in range(2, 8)]
#plotting
plt.title('Facebook Dataset')
plt.ylabel('Precision@k')
plt.xlabel('k')

plt.plot(x, hppe_result, 'r', label='HPPE')
plt.plot(x, hope_result, 'b', label='DHPE')
plt.gca().legend()

plt.show()