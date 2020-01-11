import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils import compute_accuracy, get_predictions
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def max_pool_out_size(in_size, nb_max_pool_layers=2):
    h, w = in_size
    for i in range(nb_max_pool_layers):
        h = int(np.floor((h - 2) / 2 + 1))
        w = int(np.floor((w - 2) / 2 + 1))
    return h, w


class BaselineCNNModel(pl.LightningModule):
    def __init__(self,
                 train_dataset,
                 testing_dataset,
                 input_size,
                 validation_proportion=0.1,
                 train_batch_size=16,
                 val_batch_size=16,
                 test_batch_size=16):
        self.train_dataset = train_dataset
        self.testing_dataset = testing_dataset
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        np.random.seed(143)
        indices = np.arange(len(train_dataset))
        indices = np.random.permutation(indices)
        num_train = int(np.floor((1 - validation_proportion) * len(train_dataset)))
        self.train_indices = indices[0:num_train]
        self.val_indices = indices[num_train:]
        self.num_workers = 4 if device == 'cpu' else 1
        self.input_size = input_size
        self.in_channels, self.input_height, self.input_width = input_size
        self.h, self.w = self.input_height, self.input_width

        super(BaselineCNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=256, kernel_size=(5, 5),
                      stride=1, padding=2),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=1,
                      padding=2),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(5, 5),
                      stride=1, padding=2),
            nn.ReLU()  # ,
            # nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        # self.h, self.w = max_pool_out_size([self.h, self.w], nb_max_pool_layers=3)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.h * self.w * 128, out_features=328),
            nn.ReLU(),
            nn.Linear(in_features=328, out_features=192),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=192, out_features=10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        predictions = get_predictions(y_hat)
        accuracy = compute_accuracy(predictions, y)
        tensorboard_logs = {'train_loss': loss, 'train_accuracy': accuracy}
        self.current_epoch
        return {'loss': loss, 'log': tensorboard_logs, 'accuracy': accuracy}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        predictions = get_predictions(y_hat)
        accuracy = compute_accuracy(predictions, y)
        return {'val_loss': loss, 'val_acc': accuracy}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_accuracy': avg_accuracy}
        return {'val_loss': avg_loss, 'log': tensorboard_logs,
                'val_accuracy': avg_accuracy}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.2)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(dataset=self.train_dataset, batch_size=self.train_batch_size,
                          sampler=SubsetRandomSampler(self.train_indices),
                          num_workers=self.num_workers)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(dataset=self.train_dataset, batch_size=self.val_batch_size,
                          sampler=SubsetRandomSampler(self.val_indices),
                          num_workers=self.num_workers)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(dataset=self.testing_dataset, batch_size=self.test_batch_size,
                          num_workers=self.num_workers)