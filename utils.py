from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


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
