import torch
import os
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np

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

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load('saved_models/model_2024-06-20_13:15:17.342383'))
model.eval()

test_data = np.loadtxt('test_input.txt', dtype=np.float64)
test_data = torch.from_numpy(test_data).to(device).reshape(1, 1, 28, 28).float()
prediction = int(model(test_data).argmax(dim=1))
print(prediction)
