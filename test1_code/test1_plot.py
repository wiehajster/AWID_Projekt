import pandas as pd
import matplotlib.pyplot as plt

ns = pd.read_csv('test1/test1_ns.csv')
ok = pd.read_csv('test1/test1_ok.csv')
sk = pd.read_csv('test1/test1_sk.csv')

ns['mean_times'] /= 1e9
ok['mean_times'] /= 1e9
ns['memory'] /= 1e9
ok['memory'] /= 1e9
sk['current'] /= 1e6
sk['peak'] /= 1e6

print(ns)
print(ok)
print(sk)

plt.figure()
plt.plot(ns['hidden_neurons'], ns['mean_times'])
plt.plot(ok['hidden_neurons'], ok['mean_times'])
plt.plot(sk['hidden_neurons'], sk['mean_times'])
plt.legend(['Method A', 'Method B', 'MLPClassifier'])
plt.xlabel('hidden neurons')
plt.ylabel('mean time [s]')
plt.savefig('test1/test1_time.png', bbox_inches='tight', pad_inches = 0)

plt.figure()
plt.plot(ns['hidden_neurons'], ns['memory'])
plt.plot(ok['hidden_neurons'], ok['memory'])
plt.legend(['Method A', 'Method B'])
plt.xlabel('hidden neurons')
plt.ylabel('memory [GB]')
plt.savefig('test1/test1_memory.png', bbox_inches='tight', pad_inches = 0)

plt.figure()
plt.plot(sk['hidden_neurons'], sk['current'])
plt.plot(sk['hidden_neurons'], sk['peak'])
plt.legend(['MLPClassifier - current', 'MLPClassifier - peak'])
plt.xlabel('hidden neurons')
plt.ylabel('memory [MB]')
plt.savefig('test1/test1_memory_python.png', bbox_inches='tight', pad_inches = 0)

plt.figure()
plt.plot(ns['hidden_neurons'], ns['allocs'])
plt.plot(ok['hidden_neurons'], ok['allocs'])
plt.legend(['Method A', 'Method B'])
plt.xlabel('hidden neurons')
plt.ylabel('allocs')
plt.savefig('test1/test1_allocs.png', bbox_inches='tight', pad_inches = 0)

plt.figure()
plt.plot(ns['hidden_neurons'], ns['test_acc'])
plt.plot(ok['hidden_neurons'], ok['test_acc'])
plt.plot(sk['hidden_neurons'], sk['test_acc'])
plt.legend(['Method A', 'Method B', 'MLPClassifier'])
plt.xlabel('hidden neurons')
plt.ylabel('test accuracy')
plt.savefig('test1/test1_accuracy.png', bbox_inches='tight', pad_inches = 0)


