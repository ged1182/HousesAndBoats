import torch.nn as nn
from torch import functional

class Baseline_CNN_Model(nn.Module):
    def __init__(self):
        super(Baseline_CNN_Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=1),  #126x126
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # 63x63
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1),  #61x61
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), #32x32
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1),  #30x30
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # 14x14
        )
        self.classifer = nn.Sequential(
            nn.Linear(in_features=14*14*128, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=4)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifer(x)
        return x
