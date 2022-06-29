import pandas as pd
import matplotlib.pyplot as plt

ok = pd.read_csv('test2/test2_ok.csv')
sk = pd.read_csv('test2/test2_sk.csv')

ok['mean_times'] /= 1e9
ok['memory'] /= 1e9
sk['current'] /= 1e6
sk['peak'] /= 1e6

print(ok)
print(sk)

plt.figure()
plt.plot(ok['batch_size'], ok['mean_times'])
plt.plot(sk['batch_size'], sk['mean_times'])
plt.legend(['Method B', 'MLPClassifier'])
plt.xlabel('batch size')
plt.ylabel('mean time [s]')
plt.xscale('log')
plt.savefig('test2/test2_time.png', bbox_inches='tight', pad_inches = 0)

plt.figure()
plt.plot(ok['batch_size'], ok['memory'])
plt.legend(['Method B'])
plt.xlabel('batch size')
plt.ylabel('memory [GB]')
plt.xscale('log')
plt.savefig('test2/test2_memory.png', bbox_inches='tight', pad_inches = 0)

plt.figure()
plt.plot(sk['batch_size'], sk['current'])
plt.plot(sk['batch_size'], sk['peak'])
plt.legend(['MLPClassifier - current', 'MLPClassifier - peak'])
plt.xlabel('batch size')
plt.ylabel('memory [MB]')
plt.xscale('log')
plt.savefig('test2/test2_memory_python.png', bbox_inches='tight', pad_inches = 0)

plt.figure()
plt.plot(ok['batch_size'], ok['allocs'])
plt.legend(['Method B'])
plt.xlabel('batch size')
plt.ylabel('allocs')
plt.xscale('log')
plt.savefig('test2/test2_allocs.png', bbox_inches='tight', pad_inches = 0)

plt.figure()
plt.plot(ok['batch_size'], ok['test_acc'])
plt.plot(sk['batch_size'], sk['test_acc'])
plt.legend(['Method B', 'MLPClassifier'])
plt.xlabel('batch size')
plt.ylabel('test accuracy')
plt.xscale('log')
plt.savefig('test2/test2_accuracy.png', bbox_inches='tight', pad_inches = 0)


