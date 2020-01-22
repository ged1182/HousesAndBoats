import torch.cuda as cuda
from torch.nn import Conv2d, ReLU, Parameter, Sequential, Linear, Sigmoid
from torch import stack, transpose, sum, sqrt, randn, matmul, zeros_like, tensor, argmax, mean
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets import MNIST
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import EarlyStopping
from torch.nn import Module, MSELoss
from torch.nn.functional import softmax, one_hot
from collections import OrderedDict

EPSILON = 1e-8


def get_device():
    return 'cuda:0' if cuda.is_available() else 'cpu'


def get_new_dim(h_in, w_in, conv=Conv2d(1, 4, 3)):
    """
    Compute the new height and width after performing the convolution.

    :param h_in: (int) the height of the input tensor
    :param w_in: (int) the width of the input tensor
    :param conv: (Conv2d) the convolution used
    :return: [h_out, w_out] ([int int]) the height and width after applying Conv2d
    """

    kernel_size = conv.kernel_size
    padding = conv.padding
    stride = conv.stride
    dilation = conv.dilation

    h_out = (h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / (stride[0]) + 1
    w_out = (w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / (stride[1]) + 1
    h_out = int(np.floor(h_out))
    w_out = int(np.floor(w_out))
    return h_out, w_out


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


def build_trainer(min_epochs=10, max_epochs=20, patience=5, gpus=1, overfit_pct=0.0, resume_from_checkpoint=None):
    """
    Returns a pytorch_lightning_Trainer object with early stopping enabled

    :param min_epochs: (int) The minimum number of epochs to train for (early stopping doesn't apply) (default: 10)
    :param max_epochs: (int) The maximum number of epochs to train for (default: 20)
    :param patience: (int) The number of epochs with no improvement after which training is stopped (default: 5)
    :param gpus: the number of gpus to train on (for cpu put gpus=0) (default: 1)
    :param overfit_pct: (float b/w 0 and 1) Overfits the model to a portion of the training set (default: 0.0)
    :param resume_from_checkpoint: (str) path to a checkpoint from which to resume training (default: None)
    :return: a Trainer object
    """
    # logger = TensorBoardLogger(save_dir='./Logs/MNIST/', name='CapsNet')
    early_stop_callback = EarlyStopping(patience=patience)
    trainer = Trainer(early_stop_callback=early_stop_callback, min_epochs=min_epochs,
                      max_epochs=max_epochs, gpus=gpus, overfit_pct=overfit_pct,
                      resume_from_checkpoint=resume_from_checkpoint, default_save_path='./Logs/MNIST/CapsNet')
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
        t_k = one_hot(labels, num_classes=self.nb_classes)  # [batch_size, nb_classes]
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

        self.capsule_blocks = [Conv2d(in_channels=self.in_channels,
                                      out_channels=self.caps_dim,
                                      kernel_size=self.kernel_size,
                                      stride=self.stride,
                                      padding=self.padding) for _ in np.arange(self.nb_capsule_blocks)]

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

        self.w_ij = Parameter(randn([1, nb_classes, nb_caps_in, caps_dim_in, caps_dim_out]), requires_grad=True)
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

    def __init__(self, nb_classes, train_batch_size, val_batch_size, reconstruction=True, lr=1e-3):
        """
        Initialize and build the CapsNet model.
        :param hparams: the hyperparameters of the model
        """
        super(CapsNet, self).__init__()
        # self.hparams = hparams
        # self.nb_classes = hparams.nb_classes
        self.nb_classes = nb_classes
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.reconstruction = reconstruction
        self.lr = lr
        self.input_size = [1, 28, 28]
        self.conv = Conv2d(self.input_size[0], out_channels=256, kernel_size=9)

        self.encoder = Sequential(self.conv, PrimaryCaps(), ClassCaps())
        self.decoder = None
        self.margin_loss = MarginLoss(nb_classes=self.nb_classes)
        self.mse_loss = MSELoss(reduction='mean')
        if self.reconstruction:
            self.decoder = Decoder()

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
        if self.reconstruction:
            if labels.shape[0] > 0:
                mask = one_hot(labels, num_classes=self.nb_classes)  # [batch_size, nb_classes]
            else:
                mask = one_hot(predictions, num_classes=self.nb_classes)  # [batch_size, nb_classes]

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
        Compute the loss and other metrics for a batch

        :param batch: a batch containing images, labels
        :param batch_idx: the number of the batch within the epoch
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

        transform = Compose([ToTensor(), Normalize((0.5,), (1.0,))])
        data_dir = './data/'
        dataset = MNIST(root=data_dir, train=True, transform=transform, target_transform=None, download=True)
        loader = DataLoader(dataset=dataset, batch_size=self.train_batch_size, shuffle=True)
        return loader

    @pl.data_loader
    def val_dataloader(self):
        """
        Build and return the validation dataloader

        :return: a DataLoader object
        """
        transform = Compose([ToTensor(), Normalize((0.5,), (1.0,))])
        data_dir = './data/'
        dataset = MNIST(root=data_dir, train=False, transform=transform, target_transform=None, download=True)
        loader = DataLoader(dataset=dataset, batch_size=self.val_batch_size, shuffle=True)
        return loader

    def configure_optimizers(self):
        """
        Configure the optimizer to use in the training loop

        :return: Adam optimizer with lr=self.lr
        """
        return Adam(self.parameters(), lr=self.lr)
