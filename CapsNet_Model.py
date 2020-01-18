from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import compute_accuracy, get_predictions, squash_fn
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
        outputs = squash_fn(outputs)
        return outputs


class ShapeCaps(nn.Module):
    def __init__(self, in_caps, out_caps, in_caps_dim, out_caps_dim, routing_iters):
        super(ShapeCaps, self).__init__()
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.in_caps_dim = in_caps_dim
        self.out_caps_dim = out_caps_dim
        self.routing_iters = routing_iters
        self.weightMatrices = torch.nn.Parameter(torch.randn(self.out_caps,
                                                             self.in_caps,
                                                             self.out_caps_dim,
                                                             self.in_caps_dim),
                                                 requires_grad=True)
        self.weightMatrices.to(device)

    def forward(self, x):
        x = x[:, None, :, :, None]

        u_hat = torch.matmul(self.weightMatrices, x)

        b = nn.Parameter(torch.zeros_like(u_hat))
        softmax = nn.Softmax(dim=1)

        for i in range(self.routing_iters):
            probs = softmax(b)

            outputs = squash_fn((probs * u_hat).sum(dim=2, keepdim=True))

            if i != self.routing_iters - 1:
                delta_b = (u_hat * outputs).sum(dim=-1, keepdim=True)
                b = b + delta_b

        outputs = outputs.squeeze()
        return outputs


class ReconstructionLayers(nn.Module):
    def __init__(self, in_features, out_features):
        super(ReconstructionLayers, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layers = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=self.out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        outputs = self.layers(x)

        return outputs


class MarginLoss(nn.Module):
    def __init__(self, nb_classes = 10, margin_pos=0.9, margin_neg = 0.1, p = 2):
        super(MarginLoss, self).__init__()
        self.nb_classes = nb_classes
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg
        self.p = p

    def forward(self, norms, labels):
        loss = torch.sum(torch.mul(torch.eye(self.nb_classes)[labels], (torch.mul(self.margin_pos-norms, norms<self.margin_pos))**2),dim=1)
        loss += torch.sum(torch.mul
                          (1-torch.eye(self.nb_classes)[labels],
                           (torch.mul(norms-self.margin_neg, norms > self.margin_neg)) ** 2),
                          dim=1)
        loss = loss.mean().squeeze()

        return loss


class CapsNet(pl.LightningModule):

    def __init__(self, hparams):

        super(CapsNet, self).__init__()
        self.hparams = hparams
        self.reconstruction = hparams.reconstruction
        self.train_dataset = None
        self.testing_dataset = None
        self.get_datasets()
        self.train_batch_size = self.hparams.train_batch_size
        self.val_batch_size = self.hparams.val_batch_size
        self.test_batch_size = self.hparams.test_batch_size
        np.random.seed(143)
        # self.num_workers = 4 if device == 'cpu' else 1
        self.num_workers = 0
        self.in_channels, self.input_height, self.input_width = self.input_size
        self.h, self.w = self.input_height, self.input_width
        self.nb_classes = 10
        self.routing_iters = self.hparams.routing_iters
        self.out_caps_dim = self.hparams.out_caps_dim
        self.feature_caps_dim = self.hparams.feature_caps_dim

        self.convLayer = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=256,
                      kernel_size=9, stride=1, padding=0),
            nn.ReLU()
        )

        self.primaryCaps = PrimaryCapsuleLayer(in_channels=256, out_channels=256,
                                               kernel_size=9, stride=2, padding=0,
                                               caps_dim=self.feature_caps_dim)

        self.h, self.w = self.conv_out_size([self.h, self.w], kernel_sizes=[9, 9], paddings=[0, 0],
                                            strides=[1, 2])
        self.num_feature_caps = int((self.h * self.w) * 256 / self.feature_caps_dim)

        self.shapeCaps = ShapeCaps(in_caps=self.num_feature_caps,
                                   out_caps=self.nb_classes,
                                   in_caps_dim=self.feature_caps_dim,
                                   out_caps_dim=self.out_caps_dim,
                                   routing_iters=self.routing_iters)
        self.reconstructionLayer = None
        if self.reconstruction:
            out_features = int(torch.prod(torch.tensor(self.input_size, requires_grad=False), dtype=int).squeeze())
            self.reconstructionLayer = ReconstructionLayers(
                in_features=self.out_caps_dim,
                out_features=out_features)

    def forward(self, x, y=torch.Tensor([])):
        x = self.convLayer(x)
        x = self.primaryCaps(x)
        shape_caps = self.shapeCaps(x)

        norms = self.l2_norm(shape_caps)
        predicted_classes = get_predictions(norms)
        outputs = {'norms': norms, 'predicted_classes': predicted_classes}
        if self.reconstruction:

            if y.shape != [0]:

                batch_size, _, caps_dim = shape_caps.shape
                selected_capsules = torch.zeros([batch_size, 1, caps_dim])
                for e in np.arange(batch_size):
                    selected_capsules[e] = shape_caps[e, y[e]]


            else:
                selected_capsules = shape_caps[:, predicted_classes]

            reconstructions = self.reconstructionLayer(selected_capsules)
            outputs['reconstructions'] = reconstructions

        return outputs

    def compute_loss(self, x, y, outputs):
        norms = outputs['norms']
        margin_loss = MarginLoss(nb_classes=self.nb_classes)

        total_loss = margin_loss(norms, y)
        loss_dict = {'margin_loss': total_loss}
        if 'reconstructions' in outputs:
            mse_loss = nn.MSELoss()
            reconstructions = outputs['reconstructions']
            l_mse = mse_loss(reconstructions, x.view(reconstructions.shape))
            total_loss += 0.0005 * l_mse
            loss_dict['mse_loss'] = l_mse
        loss_dict['total_loss'] = total_loss
        return loss_dict

    def l2_norm(self, outputs):
        return torch.sqrt((outputs ** 2).sum(dim=2))

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.reconstruction:
            outputs = self.forward(x, y)
        else:
            outputs = self.forward(x)
        losses = self.compute_loss(x, y, outputs)
        total_loss = losses['total_loss']
        predictions = outputs['predicted_classes']
        accuracy = compute_accuracy(predictions, y)
        tqdm_dict = {'train_loss': total_loss, 'train_acc': accuracy}
        output = OrderedDict({
            'loss': total_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
            'margin_loss': losses['margin_loss']
        })
        if 'mse_loss' in losses:
            output['mse_loss'] = losses['mse_loss']
        return output

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.reconstruction:
            outputs = self.forward(x, y)
        else:
            outputs = self.forward(x)
        losses = self.compute_loss(x, y, outputs)
        total_loss = losses['total_loss']
        predictions = outputs['predicted_classes']
        accuracy = compute_accuracy(predictions, y)
        output = OrderedDict({
            'val_loss': total_loss,
            'val_acc': accuracy,
            'val_margin_loss': losses['margin_loss']
        })
        if 'mse_loss' in losses:
            output['val_mse_loss'] = losses['mse_loss']
        return output

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_margin_loss = torch.stack([x['val_margin_loss'] for x in outputs]).mean()
        tqdm_dict = {'val_loss': avg_loss, 'val_acc': avg_accuracy}
        output = OrderedDict({
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
            'val_loss': avg_loss,
            'val_acc': avg_accuracy,
            'val_margin_loss': avg_margin_loss
        })
        if 'val_mse_loss' in outputs[0]:
            avg_mse_loss = torch.stack([x['val_mse_loss'] for x in outputs]).mean()
            output['val_mse_loss'] = avg_mse_loss
        return output

    def test_step(self, batch):
        x, y = batch
        if self.reconstruction:
            outputs = self.forward(x, y)
        else:
            outputs = self.forward(x)
        losses = self.compute_loss(x, y, outputs)
        total_loss = losses['total_loss']
        predictions = outputs['predicted_classes']
        accuracy = compute_accuracy(predictions, y)
        output = OrderedDict({
            'test_loss': total_loss,
            'test_acc': accuracy,
            'test_margin_loss': losses['margin_loss']
        })
        if 'mse_loss' in losses:
            output['test_mse_loss'] = losses['mse_loss']
        return output

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['test_acc'] for x in outputs]).mean()
        avg_margin_loss = torch.stack([x['test_margin_loss'] for x in outputs]).mean()
        tqdm_dict = {'test_loss': avg_loss, 'test_acc': avg_accuracy}
        output = OrderedDict({
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
            'test_loss': avg_loss,
            'test_acc': avg_accuracy,
            'test_margin_loss': avg_margin_loss
        })
        if 'test_mse_loss' in outputs[0]:
            avg_mse_loss = torch.stack([x['test_mse_loss'] for x in outputs]).mean()
            output['test_mse_loss'] = avg_mse_loss
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def get_datasets(self):
        if self.hparams.dataset == "MNIST":
            self.train_dataset = MNIST(root='./data', train=True,
                                       transform=transforms.ToTensor(),
                                       target_transform=None,download=True)
            self.testing_dataset = MNIST(root='./data', train=False,
                                      transform=transforms.ToTensor(),
                                      target_transform=None, download=True)
            self.input_size = [1, 28, 28]
            self.nb_classes = 10
        elif self.hparams.dataset == "HB":
            t = transforms.Compose([
                transforms.Resize([28, 28]),
                transforms.ToTensor()])
            self.train_dataset = ImageFolder(root='./data/HB/training',
                                             transform=t)
            self.testing_dataset = ImageFolder(root='./data/HB/testing',
                                               transform=t)
            self.input_size = [3, 28, 28]
            self.nb_classes = 4
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
    def conv_out_size(in_size, kernel_sizes=[9, 9], paddings=[0, 0], strides=[1, 2]):
        h, w = in_size

        for i in range(len(kernel_sizes)):
            k = kernel_sizes[i]
            p = paddings[i]
            s = strides[i]
            h = int(np.floor((h + 2 * p - k) / s + 1))
            w = int(np.floor((w + 2 * p - k) / s + 1))
        return h, w

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
        parser.add_argument('--lr', default=1e-4, type=float,
                            help="initial learning rate (default: 1e-4)",
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
        parser.add_argument('--feature_caps_dim', default=8, type=int,
                            help="the dimension of the primary capsules (default: 8)",
                            dest='feature_caps_dim')
        parser.add_argument('--out_caps_dim', default=16, type=int,
                            help="the dimension of the output capsules (default: 16)",
                            dest='out_caps_dim')
        parser.add_argument('--routing_iters', default=3, type=int,
                            help="number of iterations for the routing algorithm (default: 3)",
                            dest='routing_iters')
        # parser.add_argument('--loss', default='cross_entropy', type=str,
        #                     help="type of loss (default: cross_entropy), choices: []\nIf reconstruction then this option is ignored",
        #                     choices=['cross_entropy', 'margin'],
        #                     dest='loss_fn')
        parser.add_argument('--reconstruction', default=False, type=bool,
                            help="whether to include reconstruction (default: False)",
                            dest='reconstruction')

        return parser

def get_args():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--dataset', default='MNIST', type=str,
                               help="The dataset to use (default: MNIST, choices: ['MNIST', 'HB])",
                               choices=['MNIST', 'HB'],
                               metavar='dataset')
    parser = CapsNet.add_model_specific_args(parent_parser)
    return parser.parse_args()

def main(hparams):
    model = CapsNet(hparams)
    save_path = os.path.join('./Logs', hparams.dataset, "CapsNet")
    print(save_path)
    # checkpoint_path = os.path.join(save_dir, "checkpoint")
    # print(checkpoint_path)
    # tt_logger = TestTubeLogger(save_dir=save_dir, name="CapsNet2")
    # tt_logger = TestTubeLogger(name="CapsNet")
    # print('tt_logger.version', tt_logger.version)
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=checkpoint_path)
    if not(torch.cuda.is_available()):

        trainer = Trainer(overfit_pct=hparams.overfit_pct, default_save_path=save_path)
    else:

        trainer = Trainer(overfit_pct=hparams.overfit_pct,
                          gpus=1)
    if hparams.evaluate:
        trainer.run_evaluation()
    else:
        trainer.fit(model)

if __name__ == '__main__':
    hparams = get_args()
    print(hparams)
    main(hparams)