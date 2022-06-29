import pandas as pd
import matplotlib.pyplot as plt

ns = pd.read_csv('test4/test4_ns.csv')
ok = pd.read_csv('test4/test4_ok.csv')
sk = pd.read_csv('test4/test4_sk.csv')
cnn = pd.read_csv('test4/test4_cnn.csv')

ns['test_acc'] = ns.loc[:, ns.columns.str.startswith("test_acc")].mean(axis=1)
ok['test_acc'] = ok.loc[:, ok.columns.str.startswith("test_acc")].mean(axis=1)
sk['test_acc'] = sk.loc[:, sk.columns.str.startswith("test_acc")].mean(axis=1)

print(ns)
print(ok)
print(sk)
print(cnn)

plt.figure()
plt.plot(ns['epochs'], ns['test_acc'])
plt.plot(ok['epochs'], ok['test_acc'])
plt.plot(sk['epochs'], sk['test_acc'])
plt.plot(cnn['epochs'], cnn['valid-epoch-acc'])
plt.legend(['Method A', 'Method B', 'MLPClassifier', 'ResNet18'])
plt.xlabel('epoch')
plt.ylabel('test accuracy')
plt.savefig('test4/test4_accuracy.png', bbox_inches='tight', pad_inches = 0)


