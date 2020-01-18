from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def show_images(data_loader, classes, num_imgs=8):
    batch = next(iter(data_loader))
    samples, targets = batch
    if len(samples) > num_imgs:
        samples = samples[0:num_imgs]
        targets = targets[0:num_imgs]
    else:
        num_imgs = len(samples)
    figsize = (10, 8)
    ncols = 4
    nrows = num_imgs // 4
    if ncols * nrows < num_imgs:
        nrows += 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = axs.flat
    for ax in axs[num_imgs:]:
        ax.remove()
    axs = axs[:num_imgs]
    for i in range(num_imgs):
        sample, target = samples[i], targets[i]
        ax = axs[i]
        img = transforms.ToPILImage()(sample)
        ax.set_title(classes[target])
        ax.imshow(img)


def squash_fn(tensor, dim=-1):
    if torch.sum(torch.isnan(tensor)).squeeze() > 0:
        print('squash_fn input tensor contains nan')
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    s = scale * tensor / torch.sqrt(1e-4 + squared_norm)
    if torch.sum(torch.isnan(s)).squeeze() > 0:
        print('squash_fn output tensor contains nan')
    return s
def get_predictions(y_hat):

    predictions = torch.argmax(F.softmax(y_hat, dim=1), dim=1)


    return predictions


def compute_accuracy(predictions, targets):

    return torch.mean((predictions == targets).double())