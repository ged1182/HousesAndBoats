import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.classifer = nn.Sequential(
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
        x = self.classifer(x)
        return x


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    output = output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)
    return output


class CapsuleLayer(nn.Module):

    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels,
                 kernel_size=None, stride=None, num_iterations=3):
        super(CapsuleLayer, self).__init__()

        self.num_rout_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=0) for _ in range(num_capsules)
                ]
            )

    def squash(self, tensor, dim=-1):

        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):

        if self.num_rout_nodes != -1:
            priors = x[None, :, :, None] @ self.route_weights[:, None, :, :, :]
            logits = torch.zeros(*priors.size())

            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dime=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class CapsuleNet(nn.Module):

    def __init__(self):
        super(CapsuleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.shape_capsules = CapsuleLayer(num_capsules=4, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)

        self.decoder = nn.Sequential(
            nn.Linear(in_features=16 * 4, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128 * 128),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.shape_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        if y is None:
            _, max_length_indices = classes.max(dim=1)
            y = torch.eye(4).index_select(dim=0, index=max_length_indices.data)
        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))

        return classes, reconstructions


class CapsuleLoss(nn.ModuleList):

    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):

        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1.0 - labels) * right
        margin_loss = margin_loss.sum()

        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)
