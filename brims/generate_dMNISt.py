# @author: mcaandewiel

import os
import time

import numpy as np

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from random import randint

from tqdm import tqdm


# Batch size (no. samples per subfolder)
batch_size = 1

# Image size of the generated image, higher number means a greater transformation
WIDTH = 60
HEIGHT = 60

# Image size of the dataset, MNIST is 28x28
data_x = 28
data_y = 28

dir_name = 'data/tMNIST'

is_train = False

if is_train:
    folder_filename = 'images/train'
    name_file = 'train_tMNIST.npz'
else:
    folder_filename = 'images/test'
    name_file = 'test_tMNIST.npz'



path_save = os.path.join(os.getcwd(), dir_name, folder_filename)

#print(path_save)

#exit()
if not os.path.exists(path_save):
    os.makedirs(path_save)


assert data_x < WIDTH
assert data_y < HEIGHT

try:
    os.mkdir(dir_name)
except FileExistsError:
    print("Directory %s already exists." % dir_name)

# Initialize dataframe for labels
#data = np.array(['filename', 'label'])

dataset = datasets.MNIST(root='/home/brain/alana/brims/delta-MNIST/data/', train=is_train, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

#print(len(data_loader))
#exit()
base_image = np.zeros((len(data_loader), HEIGHT, WIDTH))
target = np.zeros((len(data_loader)))


for iteration, (x, y) in enumerate(data_loader):
    print(iteration)
    rand_x = randint(0, WIDTH - (data_x + 1))
    rand_y = randint(0, HEIGHT - (data_y + 1))

    base_image[iteration, rand_y:rand_y + data_y, rand_x:rand_x + data_x] = x.detach().numpy().reshape(data_y, data_x)
    target[iteration] = y.item()
    img = base_image[iteration, :, :]
    filename = f'{str(iteration)}.png'
    print(img.shape)
    print(os.path.join(path_save, filename))
    save_image(torch.Tensor(img), os.path.join(path_save, filename))

np.savez(os.path.join(dir_name, name_file), x=base_image, y=target)






