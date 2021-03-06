import torch

def activation(x):
    return 1/(1+torch.exp(-x))

torch.manual_seed(7)
features = torch.randn((1, 5))
weights = torch.randn_like(features)
bias = torch.randn((1, 1))

y = activation(torch.sum(features * weights) + bias)
y1 = activation((features * weights).sum() + bias)
print(y, y1)

print(weights.shape) # shape
#weights.reshape(5, 1)
#weights.resize_(5, 1)
print(weights.view(5, 1).shape)

# torch.mm(a, b) # more effient
y = activation(torch.mm(features, weights.view(5, 1)) + bias)
print(y)

torch.manual_seed(7)
features = torch.randn((1, 3))
n_input = features.shape[1]
n_hidden = 2
n_output = 1
w1 = torch.randn((n_input, n_hidden))
w2 = torch.randn((n_hidden, n_output))
b1 = torch.randn((1, n_hidden))
b2 = torch.randn(1, n_output)
h = activation(torch.mm(features, w1) + b1)
y2 = activation(torch.mm(h, w2) + b2)
print(y2)

import numpy as np
a = np.random.rand(4, 3)
b = torch.from_numpy(a)
aa = b.numpy()
print(aa)

from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('../test-me/data', download=False,
                          train=True, transform=transform)
testset = datasets.MNIST('../test-me/data', download=False,
                          train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                          shuffle=True)
dataiter = iter(trainloader)
images, label = dataiter.next()
print(images.shape, label.shape)

def sofmax(x):
    return torch.exp(x)/(torch.sum(torch.exp(x), dim=1)).view(-1, 1)

for images, label in trainloader:
    inputs = images.view(images.shape[0], -1)
    w1 = torch.randn((784, 256))
    b1 = torch.randn((1, 256))
    w2 = torch.randn((256, 10))
    b2 = torch.randn((1, 10))
    h =  activation(torch.mm(inputs, w1) + b1)
    y = torch.mm(h, w2) + b2
    print(y.shape)
    print(sofmax(y).sum(dim=1))
    break

import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 256)
        self.output = nn.Linear(256, 10)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        return x
model = Net()
print(model)

import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 256)
        self.output = nn.Linear(256, 10)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.sigmoid(self.hidden(x))
        x = F.softmax(self.output(x), dim=1)
        return x
model2 = Net()
print(model2)

#actionvation functions are non-linear
# Sigmoid
# TanH
# ReLU, rectified linear unit

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.h1 = nn.Linear(784, 128)
        self.h2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        x = F.ReLU(self.h1(x))
        x = F.ReLU(self.h2(x))
        x = F.softmax(self.output, dim=1)
        return x
# 11.PyTorch中的网络架构

model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))
criterion = nn.CrossEntropyLoss()
dataiter = iter(trainloader)
images, label = dataiter.next()
images = images.view(images.shape[0], -1)
y_pred = model(images)
loss = criterion(y_pred, label)
print(loss)

import torch.nn.functional as F
criterion = nn.NLLLoss()
dataiter = iter(trainloader)
images, label = dataiter.next()
images = images.view(images.shape[0], -1)
y_pred = model(images)
loss = criterion(y_pred, label)
print(loss)

# autograd
x = torch.randn(2,2, requires_grad=True)
y = x**2
z = y.mean()
print(y.grad_fn, z)
z.backward()
print(x.grad)
print(x/2)

from torch import optim
optimizer = optim.SGD(model.parameters(), lr=0.003)
epochs = 5
f_out = open('loss.csv', 'w')
for e in range(epochs):
    train_loss, test_loss = [], []
    running_loss, testing_loss = 0, 0
    for images, label in trainloader:
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        y_pred = model.forward(images)
        loss = criterion(y_pred, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        accuracy = 0
        with torch.no_grad():
            for images, label in testloader:
                images = images.view(images.shape[0], -1)
                y_pred = model.forward(images)
                loss = criterion(y_pred, label)
                testing_loss += loss.item()
                ps = torch.exp(y_pred) # get probabilities
                top_p, top_class = ps.topk(1, dim=1)
                # print(top_class[:5, :], )
                equals = top_class == label.view(top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        train_loss.append(running_loss/len(trainloader))
        test_loss.append(testing_loss/len(testloader))
        print(f"Epoch: {e+1}/{epochs}")
        print(f"Training loss: {running_loss/len(trainloader)}")
        print(f"Testing  loss: {testing_loss/len(testloader)}")
        print(f"Accuracy: {accuracy.item()/len(trainloader)*100}%")
        f_out.write(str(running_loss/len(trainloader))+' '+str(testing_loss/len(testloader))+'\n')
f_out.close()
import matplotlib.pyplot as plt
plt.figure()
plt.plot(train_loss, label='train loss')
plt.plot(test_loss, label='test_loss')
plt.legend()
plt.savefig('train-test_loss.pdf')

# 如何处理过拟合？drop
