import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleCNN(nn.Module):
    def __init__(self, classnum = 3):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=4, padding=1)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=4, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(160, 128)
        self.fc2 = nn.Linear(128+960, 960)
        self.fc3 = nn.Linear(960, classnum)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x1, x2 = x[0], x[1]
        # Apply convolutions
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        x1 = self.dropout(x1)
        x1 = F.relu(self.conv4(x1))

        # Flatten the output for the fully connected layers
        x1 = x1.view(x1.size(0), -1)

        # Fully connected layers
        x1 = F.relu(self.fc1(x1))

        x1 = self.fc2(torch.cat((x1, x2), dim=1))
        x1 = self.dropout(x1)
        yh = self.fc3(x1)
        
        return yh, x1