import pandas as pd
import matplotlib.pyplot as plt

ns = pd.read_csv('test3/test3_ns.csv')
ok = pd.read_csv('test3/test3_ok.csv')
sk = pd.read_csv('test3/test3_sk.csv')

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
plt.plot(ns['train_size'], ns['mean_times'])
plt.plot(ok['train_size'], ok['mean_times'])
plt.plot(sk['train_size'], sk['mean_times'])
plt.legend(['Method A', 'Method B', 'MLPClassifier'])
plt.xlabel('train size')
plt.ylabel('mean time [s]')
plt.xscale('log')
plt.savefig('test3/test3_time.png', bbox_inches='tight', pad_inches = 0)

plt.figure()
plt.plot(ns['train_size'], ns['memory'])
plt.plot(ok['train_size'], ok['memory'])
plt.legend(['Method A', 'Method B'])
plt.xlabel('train size')
plt.ylabel('memory [GB]')
plt.xscale('log')
plt.savefig('test3/test3_memory.png', bbox_inches='tight', pad_inches = 0)

plt.figure()
plt.plot(sk['train_size'], sk['current'])
plt.plot(sk['train_size'], sk['peak'])
plt.legend(['MLPClassifier - current', 'MLPClassifier - peak'])
plt.xlabel('train size')
plt.ylabel('memory [MB]')
plt.xscale('log')
plt.savefig('test3/test3_memory_python.png', bbox_inches='tight', pad_inches = 0)

plt.figure()
plt.plot(ns['train_size'], ns['allocs'])
plt.plot(ok['train_size'], ok['allocs'])
plt.legend(['Method A', 'Method B'])
plt.xlabel('train size')
plt.ylabel('allocs')
plt.xscale('log')
plt.savefig('test3/test3_allocs.png', bbox_inches='tight', pad_inches = 0)

plt.figure()
plt.plot(ns['train_size'], ns['test_acc'])
plt.plot(ok['train_size'], ok['test_acc'])
plt.plot(sk['train_size'], sk['test_acc'])
plt.legend(['Method A', 'Method B', 'MLPClassifier'])
plt.xlabel('train size')
plt.ylabel('test accuracy')
plt.xscale('log')
plt.savefig('test3/test3_accuracy.png', bbox_inches='tight', pad_inches = 0)


