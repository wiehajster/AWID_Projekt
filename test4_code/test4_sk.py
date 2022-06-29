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
epochs=10
hidden_neurons = 50

d = {"epochs" : list(range(1,epochs+1))}

tests_number = 5

for i in range(tests_number):
    print(f"Test {i}")

    clf = MLPClassifier(hidden_layer_sizes=(hidden_neurons),
                        activation="relu",
                        solver="sgd",
                        alpha=0,
                        batch_size=batch_size,
                        learning_rate="constant",
                        learning_rate_init=learning_rate,
                        max_iter=1,
                        shuffle=False,
                        tol=0,
                        verbose=True,
                        warm_start=False,
                        momentum=0,
                        nesterovs_momentum=False,
                        early_stopping=False,
                        validation_fraction=0)

    epoch_train_loss = []
    epoch_train_acc = []
    epoch_test_acc = []

    for j in range(epochs):
        clf = clf.partial_fit(train_x, train_y_encoded, classes=np.unique(train_y))
        epoch_train_loss.append(clf.loss_)

        predicted = clf.predict(train_x)
        predicted = np.argmax(predicted, axis=1)
        accuracy = accuracy_score(train_y, predicted)
        epoch_train_acc.append(accuracy)

        predicted = clf.predict(test_x)
        predicted = np.argmax(predicted, axis=1)
        accuracy = accuracy_score(test_y, predicted)
        epoch_test_acc.append(accuracy)

    d[f'train_loss_{i}'] = epoch_train_loss
    d[f'train_acc_{i}'] = epoch_train_acc
    d[f'test_acc_{i}'] = epoch_test_acc
    

df = pd.DataFrame(d)
df.to_csv('test4/test4_sk.csv', index=False)