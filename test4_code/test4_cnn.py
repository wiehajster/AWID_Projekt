from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score
import torchvision
import numpy as np
import torch
import pandas as pd

def train(model: torch.nn.Module, epoch: int):
  train_loss = train_acc = 0
  batch_loss = batch_acc = []

  model.train()
  print('='*25, 'EPOCH', epoch, '='*25)
  for batch_idx, (data, label) in enumerate(train_loader, 1):
    optimizer.zero_grad()
    data = data.to(DEVICE)
    label = label.to(DEVICE)

    output = model(data)
    loss = criterion(output, label)
    loss.backward()
    cur_loss = loss.item()
    avg_batch_loss = cur_loss/len(data)
    batch_loss.append(avg_batch_loss)
    train_loss += cur_loss
    optimizer.step()

    _, pred = torch.max(output, 1)
    correct = np.squeeze(pred.eq(label.data.view_as(pred)))
    cur_acc = correct.sum().item()
    step_acc = cur_acc/len(data)
    batch_acc.append(step_acc)
    train_acc += cur_acc

    if batch_idx % LOG_INTERVAL == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100.*batch_idx / len(train_loader),
          avg_batch_loss, step_acc))
      
  avg_epoch_loss = train_loss / len(train_loader.dataset)
  epoch_acc = train_acc / len(train_loader.dataset)
  print('====> Trainset loss: {:.4f}\tAccuracy: {:.4f}'.format(
      avg_epoch_loss, epoch_acc
  ))
  HISTORY["train-batch-loss"].append(batch_loss) 
  HISTORY["train-epoch-loss"].append(avg_epoch_loss)
  HISTORY["train-batch-acc"].append(batch_acc)
  HISTORY["train-epoch-acc"].append(epoch_acc)
  return  model

def test(model):
  test_loss = test_acc = 0.0
  batch_loss = batch_acc = []
  predictions = targets = []

  model.eval()

  for data, label in test_loader:
    data = data.to(DEVICE)
    label = label.to(DEVICE)
    output = model(data)

    loss = criterion(output, label)
    cur_loss = loss.item()
    avg_batch_loss = cur_loss/len(data)
    batch_loss.append(avg_batch_loss)
    test_loss += cur_loss

    _, pred = torch.max(output, 1)
    correct = np.squeeze(pred.eq(label.data.view_as(pred)))
    cur_acc = correct.sum().item()
    step_acc = cur_acc/len(data)
    batch_acc.append(step_acc)
    test_acc += cur_acc

  predictions.extend(list(pred.squeeze()))
  targets.extend(list(label.squeeze()))

  avg_epoch_loss = test_loss / len(test_loader.dataset)
  epoch_acc = test_acc / len(test_loader.dataset)
  print('====> Testset loss: {:.4f}\tAccuracy: {:.4f}'.format(avg_epoch_loss, epoch_acc))
  HISTORY["valid-batch-loss"].append(batch_loss) 
  HISTORY["valid-epoch-loss"].append(avg_epoch_loss)
  HISTORY["valid-batch-acc"].append(batch_acc)
  HISTORY["valid-epoch-acc"].append(epoch_acc)
  return predictions, targets

def set_history():
    history = {
        "train-batch-loss": [],
        "train-epoch-loss": [],
        "valid-batch-loss": [],
        "valid-epoch-loss": [],
        "train-batch-acc": [],
        "train-epoch-acc": [],
        "valid-batch-acc": [],
        "valid-epoch-acc": [],
    }
    return history

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
HISTORY = set_history()
BATCH_SIZE = 32
NUM_WORKERS = 2
LR = 0.01
EPOCHS = 10
LOG_INTERVAL = 50

model = torchvision.models.resnet18(num_classes=10)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model = model.to(DEVICE)

transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=NUM_WORKERS)
test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=NUM_WORKERS)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    model = train(model, epoch)
    predictions, targets = test(model)

HISTORY['epochs'] = list(range(1,EPOCHS+1))
df = pd.DataFrame(HISTORY)
df.to_csv('test4/test4_cnn.csv', index=False)