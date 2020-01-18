import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import compute_accuracy, get_predictions
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from argparse import ArgumentParser
from collections import OrderedDict
import logging
import os
from torchvision.datasets import MNIST, ImageFolder
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.logging import TestTubeLogger

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def max_pool_out_size(in_size, nb_max_pool_layers=2):
    h, w = in_size
    for i in range(nb_max_pool_layers):
        h = int(np.floor((h - 2) / 2 + 1))
        w = int(np.floor((w - 2) / 2 + 1))
    return h, w


class BaselineCNNModel(LightningModule):
    def __init__(self, hparams):
        super(BaselineCNNModel, self).__init__()

        self.hparams = hparams
        self.get_datasets()
        self.train_batch_size = self.hparams.train_batch_size
        self.val_batch_size = self.hparams.val_batch_size
        self.test_batch_size = self.hparams.test_batch_size
        np.random.seed(143)
        self.num_workers = 4 if device == 'cpu' else 1
        self.in_channels, self.input_height, self.input_width = self.input_size
        self.h, self.w = self.input_height, self.input_width
        self.nb_classes = 10
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
        tqdm_dict = {'train_loss': loss, 'train_acc': accuracy}
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        predictions = get_predictions(y_hat)
        accuracy = compute_accuracy(predictions, y)
        output = OrderedDict({
            'val_loss': loss,
            'val_acc': accuracy
        })
        return output

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['val_acc'] for x in outputs]).mean()
        tqdm_dict = {'val_loss': avg_loss, 'val_acc': avg_accuracy}
        output = OrderedDict({
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
            'val_loss': avg_loss
        })
        return output

    def test_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        predictions = get_predictions(y_hat)
        loss = F.cross_entropy(y_hat, y)
        accuracy = compute_accuracy(predictions, y)
        output = OrderedDict({
            'test_loss': loss,
            'test_acc': accuracy
        })
        return output

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['test_acc'] for x in outputs]).mean()
        tqdm_dict = {'test_loss': avg_loss, 'test_acc': avg_accuracy}
        output = OrderedDict({
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
            'test_loss': avg_loss
        })
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def get_datasets(self):
        if self.hparams.dataset == "MNIST":
            self.train_dataset = MNIST(root='./data', train=True,
                                       transform=transforms.ToTensor(),
                                       target_transform=None, download=True)
            self.testing_dataset = MNIST(root='./data', train=False,
                                      transform=transforms.ToTensor(),
                                      target_transform=None, download=True)
            self.input_size = [1, 28, 28]
            self.nb_classes = 10
        elif self.hparams.dataset == "HB":
            t = transforms.Compose(
                [transforms.Resize([28, 28]),
                transforms.ToTensor()])
            self.train_dataset = ImageFolder(root='./data/HB/training',
                                             transform=t)
            self.testing_dataset = ImageFolder(root='./data/HB/testing',
                                               transform=t)
            self.input_size = [3, 28, 28]
            self.nb_classes = 4
        else:
            self.train_dataset = None
            self.testing_dataset = None

        if not(self.train_dataset == None):
            N = len(self.train_dataset)
            idx = np.arange(N)
            train_prop = self.hparams.train_prop
            indices = np.random.permutation(idx)
            num_train = int(np.floor(train_prop * N))
            train_indices = indices[0:num_train]
            val_indices = indices[num_train:]
            self.train_sampler = SubsetRandomSampler(train_indices)
            self.val_sampler = SubsetRandomSampler(val_indices)


    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.train_batch_size,
                          sampler=self.train_sampler,
                          num_workers=self.num_workers)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(dataset=self.train_dataset, batch_size=self.val_batch_size,
                          sampler=self.val_sampler,
                          num_workers=self.num_workers)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(dataset=self.testing_dataset, batch_size=self.test_batch_size,
                          num_workers=self.num_workers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--epochs', default=10, type=int,
                            help="the max number of epochs (default: 10)",
                            metavar="max_nb_epochs")
        parser.add_argument('--train_b', default=32, type=int,
                            help="batch_size used for training (default: 32)",
                            metavar="train_batch_size",
                            dest="train_batch_size")
        parser.add_argument('--val_b', default=64, type=int,
                            help="batch_size used during validation (default: 64)",
                            metavar="val_batch_size",
                            dest="val_batch_size")
        parser.add_argument('--test_b', default=64, type=int,
                            help="batch_size used during testing (default: 64)",
                            metavar="test_batch_size",
                            dest="test_batch_size")
        parser.add_argument('--lr', default=1e-3, type=float,
                            help="initial learning rate (default: 1e-3)",
                            metavar='lr')
        parser.add_argument('--train_prop', default=0.9, type=float,
                            help="the proportion of the data to use for training (default: 0.9)",
                            metavar='train_prop')
        parser.add_argument('--eval', default=False, type=bool,
                            help="whether or not to evaluate the model (default: False)",
                            metavar='evaluate',
                            dest='evaluate')
        parser.add_argument('--overfit_pct', default=0.0, type=float,
                            help="the proportion of the data to use to overfit (default=0.0)\nuse this to see if things are working",
                            dest='overfit_pct')
        return parser

def get_args():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--dataset', default='MNIST', type=str,
                               help="The dataset to use (default: MNIST, choices: ['MNIST', 'HB])",
                               choices=['MNIST', 'HB'],
                               metavar='dataset')
    parser = BaselineCNNModel.add_model_specific_args(parent_parser)
    return parser.parse_args()

def main(hparams):
    model = BaselineCNNModel(hparams)
    save_path = os.path.join('./Logs', hparams.dataset, "BaselineCNN")
    # tt_logger = TestTubeLogger(save_dir=save_dir, name="BaselineCNNModel")

    if not(torch.cuda.is_available()):

        trainer = Trainer(overfit_pct=hparams.overfit_pct, default_save_path=save_path)
    else:

        trainer = Trainer(overfit_pct=hparams.overfit_pct, default_save_path=save_path,
                          gpus=1)
    if hparams.evaluate:
        trainer.run_evaluation()
    else:
        trainer.fit(model)

if __name__ == '__main__':
    hparams = get_args()
    print(hparams)
    main(hparams)
