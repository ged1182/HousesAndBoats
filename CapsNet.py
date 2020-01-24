from torch import cuda
from torch.nn import Conv2d, ReLU, Parameter, Sequential, Linear, Sigmoid, ModuleList
from torch import stack, transpose, sum, sqrt, randn, matmul, zeros_like, tensor, argmax, mean
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Grayscale
from torchvision.datasets import MNIST, ImageFolder
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import EarlyStopping
from torch.nn import Module, MSELoss
from torch.nn.functional import softmax, one_hot
from collections import OrderedDict
from argparse import ArgumentParser
import os

EPSILON = 1e-8
DATA_FOLDER = './data'


def count_parameters(model):
    return np.sum([p.numel() for p in model.parameters() if p.requires_grad])


def compute_dims_after_conv(input_size, conv=Conv2d(1, 4, 3)):
    """
    Compute the new height and width after performing the convolution.

    :param input_size: [channels_in, h_in, w_in] (int,int,int)
    :param conv: (Conv2d) the convolution used
    :return: out_size [channels_out, h_out, w_out] (int, int, int)
    """
    _, h_in, w_in = input_size
    kernel_size = conv.kernel_size
    channels_out = conv.out_channels
    padding = conv.padding
    stride = conv.stride
    dilation = conv.dilation

    h_out = (h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / (stride[0]) + 1
    w_out = (w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / (stride[1]) + 1
    h_out = int(np.floor(h_out))
    w_out = int(np.floor(w_out))
    out_size = [channels_out, h_out, w_out]
    return out_size


def squash(capsules_in, dim=-1):
    """
    This is a vector activation function we rescales the capsules.
    The length of the squashed vector is interpreted as the probability that the feature represented by the capsule\
    is included in the input.

    :param capsules_in: tensor of [batch_size, nb_capsules, caps_dim]
    :param dim: the dimension along which to squash (default: -1)
    :return: capsules_out: tensor of [batch_size, nb_capsules, caps_dim] of squashed tensors
    """

    squared_norms = sum(capsules_in ** 2, dim=dim, keepdim=True)
    squashing_factors = squared_norms / (1 + squared_norms)
    unit_capsules = capsules_in / sqrt(squared_norms + EPSILON)
    capsules_out = squashing_factors * unit_capsules
    return capsules_out


def build_trainer(min_epochs=1, max_epochs=100, patience=5, gpus=0, overfit_pct=0.0, dataset="HB", decoder=False,
                  resume_from_checkpoint=""):
    """
    Returns a pytorch_lightning_Trainer object with early stopping enabled

    :param min_epochs: (int) The minimum number of epochs to train for (early stopping doesn't apply) (default: 10)
    :param max_epochs: (int) The maximum number of epochs to train for (default: 20)
    :param patience: (int) The number of epochs with no improvement after which training is stopped (default: 5)
    :param gpus: the number of gpus to train on (for cpu put gpus=0) (default: 1)
    :param overfit_pct: (float b/w 0 and 1) Overfits the model to a portion of the training set (default: 0.0)
    :param dataset: (string) the name of the dataset used; used to set  save path for logs and checkpoints (default: HB)
    :param resume_from_checkpoint: (str) path to a checkpoint from which to resume training (default: None)
    :param decoder: (bool): whether the model contains a decoder. This affects the save_path for the logs/checkpoints
    :return: a Trainer object
    """
    early_stop_callback = EarlyStopping(patience=patience)
    checkpoint_path = resume_from_checkpoint if len(resume_from_checkpoint) > 0 else None
    save_path = os.path.join('./Logs', dataset, 'CapsNet-D') if decoder else os.path.join('./Logs', dataset, 'CapsNet')
    trainer = Trainer(early_stop_callback=early_stop_callback, min_epochs=min_epochs,
                      max_epochs=max_epochs, gpus=gpus, overfit_pct=overfit_pct,
                      resume_from_checkpoint=checkpoint_path, default_save_path=save_path, num_sanity_val_steps=0)
    return trainer


class MarginLoss(Module):
    """
    Margin loss defined by the sum of t_k*relu(m^+-prob)^2+lmbda*(1-t_k)*relu(prob-m^-)^2, k=0,1,..,nb_classes-1
    """

    def __init__(self, nb_classes=10, m_plus=0.9, m_minus=0.1, lmbda=0.5):
        """
        Initializes the MarginLoss module

        :param nb_classes: (int) the number of classes
        :param m_plus: (float b/w 0 and 1) indicating the positive margin (default: 0.9)
        :param m_minus: (float b/w 0 and 1 < m_plus) indicating the negative (class absence) margin (default: 0.1)
        :param lmbda: (float) the weight for the class absence portion of the loss
        """
        super(MarginLoss, self).__init__()

        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lmbda = lmbda
        self.nb_classes = nb_classes
        self.relu = ReLU(inplace=True)

    def forward(self, probs, labels):
        """
        Computes the margin loss of a batch
        :param probs: [batch_size, nb_classes] the probabilities of each class
        :param labels: [batch_size, ] the labels for image from the batch
        :return: (float) the mean of the losses from all the batches
        """
        t_k = one_hot(labels, num_classes=self.nb_classes).float()  # [batch_size, nb_classes]
        loss = t_k * self.relu(self.m_plus - probs) ** 2 + self.lmbda * (1.0 - t_k) * self.relu(
            probs - self.m_minus) ** 2
        loss = mean(loss)
        loss = loss.squeeze()
        return loss


class Squash(Module):
    """
    Squash is a capsule (vector) activation function which rescales the capsules.
    The length of the squashed capsule (vector) represents the probability that the feature represented by the capsules
    is included in the input.
    """

    def __init__(self, dim=-1):
        """
        Initializes the Squash Layer
        :rtype: Tensor
        :param dim: the dimension along which to squash the tensors
        """
        super(Squash, self).__init__()
        self.dim = dim

    def forward(self, capsules_in):
        """
        Computes the squashing activation of the capsules
        :param capsules_in: [batch_size, nb_capsules, caps_dim] the capsules that require squashing
        :return: capsules_out: [batch_size, nb_capsules, caps_dim] the capsules after getting squashed
        """

        squared_norms = sum(capsules_in ** 2, dim=self.dim, keepdim=True)
        squashing_factors = squared_norms / (1 + squared_norms)
        unit_capsules = capsules_in / sqrt(squared_norms + EPSILON)
        capsules_out = squashing_factors * unit_capsules
        return capsules_out


class PrimaryCaps(Module):
    """
    PrimaryCaps layer is a collection of capsule layers built from Conv2d
    """

    def __init__(self, caps_dim=8, kernel_size=9, in_channels=256, stride=2, padding=0, nb_capsule_blocks=32):
        """
        Initializes the PrimaryCaps layer

        :param caps_dim: (int) the dimension of the capsules (default: 8)
        :param kernel_size: (int or [int, int]) the kernel size of the Conv2d layers (default: 9)
        :param stride: (int or [int, int]) the stride of the Conv2d Layers (default: 2)
        :param padding: (int or [int, int]) the padding of the Conv2d Layers (default: 0)
        :param in_channels: (int) the number of channels of the previous layer (default: 256)
        :param nb_capsule_blocks: (int) the number of capsule blocks (default: 32)
        :return: None
        """

        super(PrimaryCaps, self).__init__()

        self.caps_dim = caps_dim
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        self.nb_capsule_blocks = nb_capsule_blocks

        self.capsule_blocks = ModuleList([Conv2d(in_channels=self.in_channels, out_channels=self.caps_dim,
                                                 kernel_size=self.kernel_size, stride=self.stride,
                                                 padding=self.padding) for _ in np.arange(self.nb_capsule_blocks)])

        self.relu = ReLU(inplace=True)
        self.squash = Squash()

    def forward(self, x):
        """
        Computes the output of the PrimaryCaps layer

        :param x: [batch_size, in_channels, height, width] the input tensor
        :return: output [batch_size, nb_capsules, caps_dim] the output tensor
        """
        batch_size, in_channels, height, width = x.shape
        assert in_channels == self.in_channels, f"number of channels in x ({in_channels}) don't match the expected " \
                                                f"in_channels ({self.in_channels}) "

        capsules_out = [c_b(x) for c_b in self.capsule_blocks]
        capsules_out = stack(capsules_out, dim=1)  # [batch_size, nb_capsule_blocks, caps_dim, h, w]
        capsules_out = self.relu(capsules_out)
        capsules_out = transpose(capsules_out, 3, 2)  # [batch_size, nb_capsule_blocks, h, caps_dim, w]
        capsules_out = transpose(capsules_out, 4, 3)  # [batch_size, nb_capsule_blocks, h, w, caps_dim]
        capsules_out = capsules_out.reshape(batch_size, -1,
                                            self.caps_dim)  # [batch_size, nb_capsule_blocks*h*w, caps_dim]
        # output = squash(output)
        capsules_out = self.squash(capsules_out)
        return capsules_out


class ClassCaps(Module):
    """
    The ClassCaps is a set of capsules whose activations
    """

    def __init__(self, nb_caps_in=1152, caps_dim_in=8, nb_classes=10, caps_dim_out=16, routing_iters=3):
        """
        Initializes the ClassCaps Layer

        :param nb_caps_in: [int] the number of capsules this module is expecting (default: 1152)
        :param caps_dim_in: [int] the dimension of the incoming capsules (default: 8)
        :param nb_classes: [int] the number of classes (default: 10)
        :param caps_dim_out: [int] the dimension of the class capsules (default: 16)
        :param routing_iters: [int] the number of routing iterations (default: 3)
        """
        super(ClassCaps, self).__init__()

        self.nb_caps_in = nb_caps_in
        self.caps_dim_in = caps_dim_in
        self.nb_classes = nb_classes
        self.caps_dim_out = caps_dim_out
        self.routing_iters = routing_iters

        self.w_ij = Parameter(0.05 * randn([1, nb_classes, nb_caps_in, caps_dim_in, caps_dim_out]), requires_grad=True)
        self.squash = Squash()

    def forward(self, caps_in):
        """
        Outputs the class capsules whose length represents the probability that the the input contains the entities
        represented by the capsules.

        :param caps_in: [batch_size, nb_caps_in, caps_dim_in]
        :return: v_j: [batch_size, nb_classes, caps_dim_out]
        """

        u_hat = matmul(caps_in[:, None, :, None, :], self.w_ij)  # [batch_size, nb_classes, nb_caps_in, 1, caps_dim_out]
        u_hat = u_hat.squeeze()  # [batch_size, nb_classes, nb_caps_in, caps_dim_out]
        u_hat = transpose(u_hat, 2, 1)  # [batch_size, nb_capsules_in, nb_classes, caps_dim_out]
        b_ij = zeros_like(u_hat)
        u_no_grad = u_hat.clone().detach()
        v_j = zeros_like(u_hat)

        for i in np.arange(self.routing_iters):
            c_i = softmax(b_ij, dim=-2)  # [batch_size, nb_capsules_in, nb_classes, caps_dim_out]

            if i == self.routing_iters - 1:
                s_j = sum(c_i * u_hat, dim=1, keepdim=True)  # [batch_size, nb_capsules_in, nb_classes, caps_dim_out]
            else:
                s_j = sum(c_i * u_no_grad, dim=1,
                          keepdim=True)  # [batch_size, nb_capsules_in, nb_classes, caps_dim_out]

            v_j = self.squash(s_j)  # [batch_size, nb_capsules_in, nb_classes, caps_dim_out]

            if i < self.routing_iters - 1:
                b_ij += sum(u_no_grad * v_j, dim=-1, keepdim=True)

        return v_j.squeeze()


class Decoder(Module):
    """
    The Decoder reconstructs the input image from the ClassCaps.
    """

    def __init__(self, in_features=16, fc_1=512, fc_2=1024, out_features=784):
        """
        Initializes the Decoder Module

        :param in_features: (int) this is the dimension of the ClassCaps (default: 16)
        :param fc_1: (int) the number of nodes in the first fully-connected layer (default: 512)
        :param fc_2: (int) the number of nodes in the second fully-connected layer (default: 1024)
        :param out_features: the number of nodes in the output layer (default: 784)
        """
        super(Decoder, self).__init__()

        self.in_features = in_features
        self.fc_1 = fc_1
        self.fc_2 = fc_2
        self.out_features = out_features

        self.layers = Sequential(
            Linear(in_features=self.in_features,
                   out_features=self.fc_1),
            ReLU(inplace=True),
            Linear(in_features=self.fc_1,
                   out_features=self.fc_2),
            ReLU(inplace=True),
            Linear(in_features=fc_2,
                   out_features=self.out_features),
            Sigmoid()
        )

    def forward(self, capsule):
        """
        Takes in a capsule an return a reconstruction

        :param capsule: [batch_size, caps_dim] the capsule from which to decode
        :return: reconstruction: [batch_size, out_features] the reconstruction
        """
        reconstruction = self.layers(capsule)
        return reconstruction


# noinspection PyTypeChecker
class CapsNet(LightningModule):
    """
    This is the main model which optionally includes a decoder.
    """

    def __init__(self, hparams):
        """
        :param nb_classes:
        :param train_batch_size:
        :param val_batch_size:
        :param reconstruction:
        :param lr:
        """
        super(CapsNet, self).__init__()
        self.hparams = hparams
        self.train_dataset, self.val_dataset = self.get_datasets(dataset_name=self.hparams.dataset)
        self.nb_classes = len(self.train_dataset.classes)
        self.input_size = self.train_dataset[0][0].shape
        self.conv_channels = hparams.conv_channels
        conv_layer = Conv2d(self.input_size[0],
                   out_channels=hparams.conv_channels,
                   kernel_size=9)

        primaryCaps_layer = PrimaryCaps(caps_dim=hparams.primary_caps_dim,
                        kernel_size=9,
                        in_channels=self.conv_channels,
                        stride=2,
                        padding=0,
                        nb_capsule_blocks=hparams.nb_capsule_blocks)
        [c, h, w] = compute_dims_after_conv(self.input_size, conv_layer)
        [c, h, w] = compute_dims_after_conv([c, h, w], Conv2d(1, hparams.class_caps_dim, kernel_size=9, stride=2))
        nb_caps_in = hparams.nb_capsule_blocks * h * w

        classCaps_layer = ClassCaps(nb_caps_in=nb_caps_in,
                      caps_dim_in=hparams.primary_caps_dim,
                      nb_classes=self.nb_classes,
                      caps_dim_out=hparams.class_caps_dim,
                      routing_iters=hparams.routing_iters)
        self.encoder = Sequential(
            conv_layer,
            primaryCaps_layer,
            classCaps_layer
        )

        self.decoder = None
        self.margin_loss = MarginLoss(nb_classes=self.nb_classes)
        self.mse_loss = MSELoss(reduction='mean')

        if hparams.reconstruction:
            out_features = int(np.prod(self.input_size))
            decoder_layer = Decoder(in_features=hparams.class_caps_dim,
                                    fc_1=self.hparams.fc1,
                                    fc_2=self.hparams.fc2,
                                    out_features=np.prod(self.input_size))
            self.decoder = decoder_layer

    def forward(self, images, labels=tensor([])):
        """
        Does the forward pass of the CapsNet model.

        :param images: (batch_size, *input_size) The images from a batch
        :param labels: (batch_size, int) The ground truth labels for the images in the batch
        :return: OrderedDict containing class_caps, class_probs, 'predictions', and optionally 'reconstructions'
        """
        class_caps = self.encoder(images)  # [batch_size, nb_classes, caps_dim]
        class_probs = sqrt(sum(class_caps ** 2, dim=-1, keepdim=False))  # [batch_size, nb_classes]
        predictions = argmax(class_probs, dim=1)
        output = OrderedDict({
            'class_caps': class_caps,
            'class_probs': class_probs,
            'predictions': predictions
        })
        if self.hparams.reconstruction:
            if labels.shape[0] > 0:
                mask = one_hot(labels, num_classes=self.nb_classes).float()  # [batch_size, nb_classes]
            else:
                mask = one_hot(predictions, num_classes=self.nb_classes).float()  # [batch_size, nb_classes]

            mask = mask[:, :, None].expand_as(class_caps)
            caps_to_decode = sum(class_caps * mask, dim=1)  # [batch_size, caps_dim]

            reconstructions = self.decoder(caps_to_decode)
            output.update({'reconstructions': reconstructions})
            output.move_to_end('reconstructions', last=True)

        return output

    def training_step(self, batch, batch_nb):
        """
        Compute the loss and other metrics for a batch

        :param batch: a batch containing images, labels
        :param batch_nb: the number of the batch within the epoch
        :return: OrderedDict object containing the loss and other metrics from the batch
        """

        images, labels = batch
        output = self.forward(images, labels)
        class_probs = output['class_probs']
        margin_loss_ = self.margin_loss(class_probs, labels)
        log = {'margin_loss': margin_loss_}
        loss = margin_loss_
        predictions = output['predictions']
        accuracy = mean((predictions == labels).float()).squeeze()
        log['accuracy'] = accuracy

        if 'reconstructions' in output.keys():
            mse_loss_ = self.mse_loss(images.view(images.shape[0], -1), output['reconstructions'])
            loss = margin_loss_ + 0.0005 * mse_loss_
            log['mse_loss'] = mse_loss_

        log['loss'] = loss
        metrics_dict = OrderedDict({
            'loss': loss,
            'progress_bar': log,
            'log': log,
        })
        return metrics_dict

    # def training_end(self, outputs):
    #     """
    #     Compute the aggregate metrics for all the validation batches.
    #
    #     :param outputs: the ordered dictionaries from the training_step
    #     :return: OrderedDict object containing the average metrics from the values in outputs
    #     """
    #     print(outputs)
    #     avg_loss = stack([x['loss'] for x in outputs]).mean()
    #     avg_margin_loss = stack([x['log']['margin_loss'] for x in outputs]).mean()
    #     avg_accuracy = stack([x['log']['accuracy'] for x in outputs]).mean()
    #     log = {'avg_loss': avg_loss, 'avg_margin_loss': avg_margin_loss, 'avg_accuracy': avg_accuracy}
    #     if 'mse_loss' in outputs[0]['log'].keys():
    #         avg_mse_loss = stack([x['log']['mse_loss'] for x in outputs]).mean()
    #         log['avg_mse_loss'] = avg_mse_loss
    #
    #     metrics_dict = OrderedDict({
    #         'loss': avg_loss,
    #         'progress_bar': log,
    #         'log': log
    #     })
    #     return metrics_dict

    def validation_step(self, batch, batch_nb):
        """
        Compute the validation loss and other metrics for a batch

        :param batch: a batch containing images, labels
        :param batch_nb: the number of the batch within the epoch
        :return: OrderedDict object containing the loss and other metrics from the batch
        """

        images, labels = batch
        output = self.forward(images, labels)
        class_probs = output['class_probs']
        margin_loss_ = self.margin_loss(class_probs, labels)
        log = {'margin_loss': margin_loss_}
        loss = margin_loss_
        predictions = output['predictions']
        accuracy = mean((predictions == labels).float()).squeeze()
        log['accuracy'] = accuracy

        if 'reconstructions' in output.keys():
            mse_loss_ = self.mse_loss(images.view(images.shape[0], -1), output['reconstructions'])
            loss = margin_loss_ + 0.0005 * mse_loss_
            log['mse_loss'] = mse_loss_

        log['loss'] = loss
        metrics_dict = OrderedDict({
            'loss': loss,
            'progress_bar': log,
            'log': log,
        })
        return metrics_dict

    # noinspection PyTypeChecker
    def validation_end(self, outputs):
        """
        Compute the aggregate metrics for all the validation batches.

        :param outputs: the ordered dictionaries from the validation_step
        :return: OrderedDict object containing the average metrics from the values in outputs
        """

        avg_loss = stack([x['loss'] for x in outputs]).mean()
        avg_margin_loss = stack([x['log']['margin_loss'] for x in outputs]).mean()
        avg_accuracy = stack([x['log']['accuracy'] for x in outputs]).mean()
        log = {'val_loss': avg_loss, 'val_margin_loss': avg_margin_loss, 'val_accuracy': avg_accuracy}
        if 'mse_loss' in outputs[0]['log'].keys():
            avg_mse_loss = stack([x['log']['mse_loss'] for x in outputs]).mean()
            log['val_mse_loss'] = avg_mse_loss

        metrics_dict = OrderedDict({
            'val_loss': avg_loss,
            'progress_bar': log,
            'log': log
        })
        return metrics_dict

    @pl.data_loader
    def train_dataloader(self):
        """
        Build and return the training dataloader

        :return: a DataLoader object
        """

        loader = DataLoader(dataset=self.train_dataset, batch_size=self.hparams.train_batch_size, shuffle=True)
        return loader

    @pl.data_loader
    def val_dataloader(self):
        """
        Build and return the validation dataloader

        :return: a DataLoader object
        """
        loader = DataLoader(dataset=self.val_dataset, batch_size=self.hparams.val_batch_size, shuffle=True)
        return loader

    def configure_optimizers(self):
        """
        Configure the optimizer to use in the training loop

        :return: Adam optimizer with lr=self.hparams.lr
        """
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)

        # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.gamma)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, min_lr=1e-6)
        return [optimizer], [scheduler]

    @staticmethod
    def get_datasets(dataset_name="HB"):
        if dataset_name == "MNIST":
            transform = Compose([ToTensor(), Normalize((0.5,), (1.0,))])
            train_dataset = MNIST(root=DATA_FOLDER, train=True, transform=transform, target_transform=None,
                                  download=True)
            val_dataset = MNIST(root=DATA_FOLDER, train=False, transform=transform, target_transform=None,
                                download=True)
        else:
            if dataset_name == "HB":
                transform = Compose([Grayscale(), ToTensor()])
            else:
                transform = ToTensor()
            train_dataset = ImageFolder(root=os.path.join(DATA_FOLDER, dataset_name, "training"), transform=transform)
            val_dataset = ImageFolder(root=os.path.join(DATA_FOLDER, dataset_name, "validation"), transform=transform)

        print(f"Dataset: {dataset_name}")
        print(f"Training Examples: {'{: ,}'.format(len(train_dataset))}")
        print(f"Validation Examples: {'{: ,}'.format(len(val_dataset))}")

        return train_dataset, val_dataset

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=True)
        parser.add_argument('--min_epochs', default=1, type=int,
                            help="min number of epochs (default: 1)",
                            metavar="min_epochs")
        parser.add_argument('--max_epochs', default=100, type=int,
                            help="the max number of epochs (default: 10)",
                            metavar="max_epochs")
        parser.add_argument('--train_batch_size', default=128, type=int,
                            help="batch_size used for training (default: 128)",
                            metavar="train_batch_size",
                            dest="train_batch_size")
        parser.add_argument('--val_batch_size', default=1024, type=int,
                            help="batch_size used during validation (default: 1024)",
                            metavar="val_batch_size",
                            dest="val_batch_size")
        parser.add_argument('--lr', default=1e-3, type=float,
                            help="initial learning rate (default: 1e-3)",
                            metavar='lr')
        # parser.add_argument('--gamma', default=0.95, type=float,
        #                     help="lr decay multiplier after every epoch (default: 0.95)",
        #                     metavar='gamma')
        parser.add_argument('--patience', default=5, type=int,
                            help="number of epochs without improvement to 'val_loss' before early stopping.",
                            dest='patience')
        parser.add_argument('--overfit_pct', default=0.0, type=float,
                            help="the proportion of the data to use to overfit (default=0.0)",
                            dest='overfit_pct')
        parser.add_argument('--conv_channels', default=256, type=int,
                            help="the number of out_channels in Conv2d for the first hidden layer (default: 256)",
                            dest='conv_channels')
        parser.add_argument('--nb_capsule_blocks', default=32, type=int,
                            help="the number of capsule blocks in the PrimaryCaps layer (default: 32)",
                            dest='nb_capsule_blocks')
        parser.add_argument('--primary_caps_dim', default=8, type=int,
                            help="the dimension of the primary capsules (default: 8)",
                            dest='primary_caps_dim')
        parser.add_argument('--class_caps_dim', default=16, type=int,
                            help="the dimension of the output capsules (default: 16)",
                            dest='class_caps_dim')
        parser.add_argument('--routing_iters', default=3, type=int,
                            help="number of iterations for the routing algorithm (default: 3)",
                            dest='routing_iters')
        parser.add_argument('--reconstruction', default=False, type=bool,
                            help="whether to include reconstruction (default: False)",
                            dest='reconstruction')
        parser.add_argument('--resume_from_ckpt', default="", type=str,
                            help="the path of a checkpoint to resume training from",
                            dest='resume_from_checkpoint')
        parser.add_argument('--fc1', default=512, type=int,
                            help="the number of nodes in the first fc layer of the decoder (default: 512)",
                            dest='fc1')
        parser.add_argument('--fc2', default=1024, type=int,
                            help="the number of nodes in the first fc layer of the decoder (default: 1024)",
                            dest='fc2')


        return parser


def get_args():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--dataset', default='HB', type=str,
                               help="The dataset to use (default: HB, choices: ['MNIST', 'HB', 'HB_Colored'])",
                               choices=['MNIST', 'HB', 'HB_Colored'],
                               metavar='dataset')
    parser = CapsNet.add_model_specific_args(parent_parser)
    return parser.parse_args()


def main(hparams):
    model = CapsNet(hparams)
    print(f"The model has {'{:,}'.format(count_parameters(model))} parameters.")

    if cuda.is_available():
        print("GPU Found!")
        trainer = build_trainer(min_epochs=hparams.min_epochs,
                                max_epochs=hparams.max_epochs,
                                patience=hparams.patience,
                                gpus=1,
                                overfit_pct=hparams.overfit_pct,
                                dataset=hparams.dataset,
                                resume_from_checkpoint=hparams.resume_from_checkpoint,
                                decoder=hparams.reconstruction
                                )
    else:
        print("GPU Not Found! Will use CPU.")
        trainer = build_trainer(min_epochs=hparams.min_epochs,
                                max_epochs=hparams.max_epochs,
                                patience=hparams.patience,
                                gpus=0,
                                overfit_pct=hparams.overfit_pct,
                                dataset=hparams.dataset,
                                resume_from_checkpoint=hparams.resume_from_checkpoint,
                                decoder=hparams.recontruction
                                )

    trainer.fit(model)


if __name__ == '__main__':
    hparams = get_args()
    print(hparams)
    main(hparams)
