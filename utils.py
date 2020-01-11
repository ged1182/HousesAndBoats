from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_train_val_data(data_dir, verbose=False):
    training_data = ImageFolder(root=os.path.join(data_dir, "training"), transform=transforms.ToTensor())
    validation_data = ImageFolder(root=os.path.join(data_dir, "validation"), transform=transforms.ToTensor())
    training_data
    if verbose:
        print(len(training_data.classes), "Classes:", training_data.classes)
        print("Training Size: {0:d}".format((len(training_data))))
        print("Validation Size: {0:d}".format((len(validation_data))))
    return training_data, validation_data


def get_test_data(data_dir, verbose=False):
    testing_data = ImageFolder(root=os.path.join(data_dir, "testing"), transform=transforms.ToTensor())
    if verbose:
        print(len(testing_data.classes), "Classes:", testing_data.classes)
        print("Testing Size: {0:d}".format((len(testing_data))))
    return testing_data


def get_loader(data, batch_size=16, shuffle=True, num_workers=4):
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


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


def get_predictions(y_hat):
    return torch.argmax(F.softmax(y_hat, dim=1), dim=1)


def compute_accuracy(predictions, targets):
    return torch.mean((predictions == targets).double())