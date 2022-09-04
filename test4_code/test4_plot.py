import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

ns = pd.read_csv("test4/test4_ns.csv")
ok = pd.read_csv("test4/test4_ok.csv")
sk = pd.read_csv("test4/test4_sk.csv")
cnn = pd.read_csv("test4/test4_cnn.csv")

output_dir = Path("test4/plots")
if output_dir.is_dir():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

ns["test_acc"] = ns.loc[:, ns.columns.str.startswith("test_acc")].mean(axis=1)
ok["test_acc"] = ok.loc[:, ok.columns.str.startswith("test_acc")].mean(axis=1)
sk["test_acc"] = sk.loc[:, sk.columns.str.startswith("test_acc")].mean(axis=1)

ns.drop(columns=ns.loc[:, ns.columns.str.startswith("test_acc_")], inplace=True)
ok.drop(columns=ok.loc[:, ok.columns.str.startswith("test_acc_")], inplace=True)
sk.drop(columns=sk.loc[:, sk.columns.str.startswith("test_acc_")], inplace=True)

ns.to_csv("test4/test4_ns_mean.csv")
ok.to_csv("test4/test4_ok_mean.csv")
sk.to_csv("test4/test4_sk_mean.csv")

print(ns["test_acc"])
print(ok["test_acc"])
print(sk["test_acc"])
print(cnn["valid-epoch-acc"])

print("ns", ns["test_acc"].max(), ns["test_acc"].argmax())
print("ok", ok["test_acc"].max(), ok["test_acc"].argmax())
print("sk", sk["test_acc"].max(), sk["test_acc"].argmax())
print("cnn", cnn["valid-epoch-acc"].max(), cnn["valid-epoch-acc"].argmax())


plt.figure()
plt.plot(ns["epochs"], ns["test_acc"])
plt.plot(ok["epochs"], ok["test_acc"])
plt.plot(sk["epochs"], sk["test_acc"])
plt.plot(cnn["epochs"], cnn["valid-epoch-acc"])
plt.legend(["Method A", "Method B", "MLPClassifier", "ResNet18"])
plt.xlabel("epoch")
plt.ylabel("test accuracy")
plt.grid()
plt.savefig(
    output_dir.joinpath("test4_accuracy.png"), bbox_inches="tight", pad_inches=0
)
