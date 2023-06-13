
import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn import preprocessing
from torchvision import transforms


class DatasetdMNIST(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, root, transform=None):
        'Initialization'

        self.root = root
        self.transform = transform
        self.dataset = np.load(self.root)
        self.imgs = self.dataset['x']
        self.targets = self.dataset['y']

    def __len__(self):
        return len(self.dataset['x'])

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        #img, target = self.imgs[index, :], self.targets[index]
        img = self.imgs[index, :, :]
        target = self.targets[index]
        target = torch.tensor(target)
        img = Image.fromarray(img)
        img = np.array(img)
        if self.transform:
            img = self.transform(img)

        return img, target


class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, size_output=112):
        self.size_output = size_output

    def __call__(self, image):
        d = image
        #if self.size_output != 112:
            #d = np.array(Image.fromarray(d).resize((self.size_output,
            #                                                self.size_output),
            #                                               Image.NEAREST))
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        d = min_max_scaler.fit_transform(d)
        d = d.astype(np.float32)
        #d = np.rint(d)
        #d = d.astype(int)
        return d

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #patches_torch = []
        #for patch in patches:
        img = img.reshape(img.shape[0], img.shape[1], 1)
        img = img.transpose((2, 0, 1))
        #patches_torch.append(patch)

        img = np.asarray(img)
        return torch.from_numpy(img)


def get_dmnist_data(batch_size=64, lens=[60, 64]):
    mnist_trainset, mnist_valset, mnist_testset = {}, {}, {}
    train_loaders, val_loaders, test_loaders = {}, {}, {}

    for l in lens:
        mnist_trainset[l] = DatasetdMNIST(root=f'/home/brain/alana/brims/delta-MNIST/data/dMNIST/processed/training.npz',
                                         transform=transforms.Compose([Rescale(size_output=l), ToTensor()]))
        mnist_valset[l] = DatasetdMNIST(root=f'/home/brain/alana/brims/delta-MNIST/data/dMNIST/processed/val.npz',
                                         transform = transforms.Compose([Rescale(size_output=l), ToTensor()]))
        mnist_testset[l] = DatasetdMNIST(root=f'/home/brain/alana/brims/delta-MNIST/data/dMNIST/processed/test.npz',
                                         transform=transforms.Compose([Rescale(size_output=l), ToTensor()]))

        train_loaders[l] = torch.utils.data.DataLoader(mnist_trainset[l], batch_size=batch_size, drop_last=True)
        val_loaders[l] = torch.utils.data.DataLoader(mnist_valset[l], batch_size=batch_size, drop_last=True)
        test_loaders[l] = torch.utils.data.DataLoader(mnist_testset[l], batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loaders, val_loaders, test_loaders


def plot_batch(loader, len):
    i_batch = 0
    #print(len(loader))
    for images_batched, targets_batched in loader:
        print(images_batched.shape, targets_batched)

        print(f'unique values: {torch.unique(images_batched[0, :, :, :])}')
        images_batched = images_batched.reshape((images_batched.shape[0], 1, len, len))

        i_batch += 1
        # observe 4th batch and stop.
        if i_batch == 1:
            #images_batch = images_batched[idx]
            plt.figure(i_batch)
            grid_img = torchvision.utils.make_grid(images_batched.cpu().detach(),
                                                   nrow=10,
                                                   padding=1, pad_value=50)
            plt.imshow(255*grid_img.permute(1, 2, 0))
            plt.axis('off')
            plt.ioff()
            plt.show()
        break


if __name__ == "__main__":
    batch_size = 10
    lens = [112]

    train_loaders, val_loaders, test_loaders = get_dmnist_data(batch_size=batch_size, lens=lens)

    plot_batch(val_loaders[112], lens[0])
    #plot_batch(val_loaders[64], lens[1])

