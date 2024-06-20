#!/usr/bin/env python
# coding: utf-8

# In[47]:


import torch
import os
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from datetime import datetime


# In[48]:


os.makedirs("data", exist_ok=True)


# In[49]:


learning_rate = 0.0002
epochs = 1 # 1 epoch to test
batch_size = 64
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
num_workers = 0
pin_memory = True if device.type == 'cuda' else False
print('Device:', device)


# In[50]:


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

source_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
source_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


# In[51]:


source_train = DataLoader(source_trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
source_test = DataLoader(source_testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)


# In[52]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# In[53]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1, 20, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(20, 50, 5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(50 * 4 * 4, 200),
            nn.ReLU(),

            nn.Linear(200, 10)
        )

    def forward(self, x):
        return self.main(x)


# In[54]:


model = CNN().to(device)
model.apply(weights_init)

optim = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
criterion_label = nn.CrossEntropyLoss()


# In[55]:


model.train()

for epoch in range(epochs):
    for i, ((X, y)) in enumerate(source_train):
        X, y = X.to(device), y.to(device)

        model.zero_grad()

        output = model(X)

        loss_prediction = criterion_label(output, y)
        loss = loss_prediction

        accuracy = torch.sum(output.argmax(dim=1) == y)/batch_size

        loss.backward()
        optim.step()

        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [Prediction loss %f] [Accuracy %f]"
            % (
                epoch, epochs - 1, i, len(source_train),
                loss_prediction.item(), accuracy.item()
            )
        )


# In[56]:

now = str(datetime.now()).replace(' ', '_')
path = f'/usr/saved_models/model_{now}'
print('Saving model to:', path)
torch.save(model.state_dict(), path)


# In[57]:


model.eval()
accuracy = 0
for X, y in source_test:
    X, y = X.to(device), y.to(device)

    preds = model(X)
    accuracy += torch.sum(preds.argmax(dim=1) == y)
accuracy = accuracy / (len(source_test) * batch_size)
print('\nTest accuracy:', accuracy)


# In[ ]:




