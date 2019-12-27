import torch
import torch.nn as nn
# import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class BaselineCNNModel(nn.Module):
    def __init__(self):
        super(BaselineCNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1),  # 126x126
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # 63x63
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1),  # 61x61
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # 32x32
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1),  # 30x30
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # 14x14
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=14 * 14 * 128, out_features=1024),
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
        x = self.classifier(x)
        return x


def squash(x, axis=-1):
    norm = torch.norm(x, dim=axis, keepdim=True)
    scale = norm ** 2 / (1 + norm ** 2)
    return scale * x


class PrimaryCapsuleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, caps_dim,
                 kernel_size=9, stride=1, padding=0):
        super(PrimaryCapsuleLayer, self).__init__()

        self.caps_dim = caps_dim
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        outputs = self.conv(x)
        outputs = outputs.view(x.size(0), -1, self.caps_dim)
        outputs = squash(outputs)
        return outputs


class CapsNet(nn.Module):

    def __init__(self, input_size, classes, routing_iters):
        super(CapsNet, self).__init__()

        self.input_size = input_size
        self.classes = classes
        self.routing_iters = routing_iters

        self.conv = nn.Conv2d(in_channels=input_size[0], out_channels=256,
                              kernel_size=9, stride=1, padding=0)

        self.primaryCaps = PrimaryCapsuleLayer(in_channels=256, out_channels=256,
                                               kernel_size=9, stride=2, padding=0,
                                               caps_dim=8)
        self.relu = nn.ReLU()

        self.weightMatrices = torch.nn.Parameter(torch.randn(classes, 32*6*6, 16, 8),
                                                 requires_grad=True)
        self.weightMatrices.to(device)

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)


    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.primaryCaps(x)
        x = x.transpose(0, 1).transpose(1, 2)
        u_hat = self.weightMatrices @ x
        b = nn.Parameter(torch.zeros_like(u_hat))
        softmax = nn.Softmax(dim=1)
        for i in range(self.routing_iters):
            probs = softmax(b)
            outputs = self.squash((probs * u_hat).sum(dim=1, keepdim=True))

            if i != self.routing_iters - 1:
                delta_b = (u_hat * outputs).sum(dim=-1, keepdim=True)
                b = b + delta_b

        outputs = outputs.squeeze()
        outputs = outputs.transpose(1, 2)
        outputs = outputs.transpose(0, 1)
        outputs = (outputs **2).sum(dim=2)
        return outputs