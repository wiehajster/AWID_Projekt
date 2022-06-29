from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import torchvision
import numpy as np
import torch
import tracemalloc
import time
import pandas as pd

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True)
train_x = trainset.data.numpy()
train_y = trainset.targets.numpy()

test_x = testset.data.numpy()
test_y = testset.targets.numpy()

print(train_x.shape)
print(train_y.shape)

image_size = train_x.shape[1] * train_x.shape[2]
train_x = train_x.reshape((train_x.shape[0], image_size))/255
test_x = test_x.reshape((test_x.shape[0], image_size))/255

print(train_x.shape)
print(test_x.shape)

encoder = OneHotEncoder()
encoder.fit(train_y.reshape(-1, 1))
train_y_encoded = encoder.transform(train_y.reshape(-1, 1)).toarray()
test_y_encoded = encoder.transform(test_y.reshape(-1, 1)).toarray()

print(train_y_encoded.shape)
print(test_y_encoded.shape)
print(train_y[:5])
print(train_y_encoded[:5])

batch_size=1
learning_rate=0.01
epochs=1
hidden_neurons = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
times = [7, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4]
d = {"mean_times" : [],
     "current" : [],
     "peak" : [],
     "hidden_neurons" : hidden_neurons,
     "train_loss": [],
     "train_acc": [],
     "test_acc": []}

for hn, t in zip(hidden_neurons, times):
    print('hidden neurons:', hn)

    time_res = 0
    mean_train_loss = 0
    mean_train_acc = 0
    mean_test_acc = 0

    for i in range(t):
        print(i)
        clf = MLPClassifier(hidden_layer_sizes=(hn),
                            activation="relu",
                            solver="sgd",
                            alpha=0,
                            batch_size=batch_size,
                            learning_rate="constant",
                            learning_rate_init=learning_rate,
                            max_iter=epochs,
                            shuffle=False,
                            tol=0,
                            verbose=True,
                            warm_start=False,
                            momentum=0,
                            nesterovs_momentum=False,
                            early_stopping=False,
                            validation_fraction=0)
        start = time.time()
        clf = clf.fit(train_x, train_y_encoded)
        end = time.time()
        time_res = time_res + end - start

        mean_train_loss += clf.loss_

        predicted = clf.predict(train_x)
        predicted = np.argmax(predicted, axis=1)
        accuracy = accuracy_score(train_y, predicted)
        mean_train_acc += accuracy

        predicted = clf.predict(test_x)
        predicted = np.argmax(predicted, axis=1)
        accuracy = accuracy_score(test_y, predicted)
        mean_test_acc += accuracy

    time_res /= t
    d['mean_times'].append(time_res)
    d['train_loss'].append(mean_train_loss/t)
    d['train_acc'].append(mean_train_acc/t)
    d['test_acc'].append(mean_test_acc/t)

    print('tracemalloc')
    
    clf = MLPClassifier(hidden_layer_sizes=(hn),
                            activation="relu",
                            solver="sgd",
                            alpha=0,
                            batch_size=batch_size,
                            learning_rate="constant",
                            learning_rate_init=learning_rate,
                            max_iter=epochs,
                            shuffle=False,
                            tol=0,
                            verbose=10,
                            warm_start=False,
                            momentum=0,
                            nesterovs_momentum=False,
                            early_stopping=False,
                            validation_fraction=0)

    tracemalloc.start()
    clf = clf.fit(train_x, train_y_encoded)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracemalloc.clear_traces()
    d['current'].append(current)
    d['peak'].append(peak)

df = pd.DataFrame(d)
df.to_csv('test1/test1_sk.csv', index=False)