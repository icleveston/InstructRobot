
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


class tMNIST(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, root, len):
        'Initialization'

        self.root = root
        self.dataset = np.load(self.root)
        self.imgs = self.dataset['x']
        self.targets = self.dataset['y']
        self.len = len

    def __len__(self):
        return self.dataset['x'].shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        img = self.imgs[index, :, :]

        target = self.targets[index]
        target = torch.tensor(target)
        img = np.asarray(img)
        img = torch.from_numpy(img)
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        #print(img.shape)
        #print(type(img))
        if self.len != 60:
            img = F.interpolate(img, size=(self.len, self.len), mode='nearest')

        img = img.reshape(1, img.shape[2], img.shape[3])

        return img, target





def get_tmnist_data(batch_size=64, lens=[60]):
    tmnist_trainset, tmnist_testset = {}, {}
    train_loaders, val_loaders, test_loaders = {}, {}, {}

    for l in lens:
        tmnist_trainset[l] = tMNIST(root=f'/home/brain/alana/brims/delta-MNIST/data/tMNIST/train_tMNIST.npz', len=l)
        #print('saiu aqui')
        tmnist_testset[l] = tMNIST(root=f'/home/brain/alana/brims/delta-MNIST/data/tMNIST/test_tMNIST.npz', len=l)
        #print('saiu aqui')
        num_val = len(tmnist_trainset[l]) // 5
        np.random.seed(0)
        num_train = len(tmnist_trainset[l])
        idxs = np.random.choice(num_train, num_train, replace=False)

        train_sampler = SubsetRandomSampler(idxs[num_val:])
        val_sampler = SubsetRandomSampler(idxs[:num_val])

        train_loaders[l] = torch.utils.data.DataLoader(tmnist_trainset[l], batch_size=batch_size, drop_last=True, sampler=train_sampler)
        val_loaders[l] = torch.utils.data.DataLoader(tmnist_trainset[l], batch_size=batch_size, drop_last=True, sampler=val_sampler)
        test_loaders[l] = torch.utils.data.DataLoader(tmnist_testset[l], batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loaders, val_loaders, test_loaders


def plot_batch(loader, lens):
    i_batch = 0
    #print(len(loader))
    for images_batched, targets_batched in loader:
        print(images_batched.shape, targets_batched)

        print(f'unique values: {torch.unique(images_batched[0, :, :, :])}')
        #images_batched = images_batched.reshape((images_batched.shape[0], 1, lens, lens))

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
    batch_size = 5
    lens = [68]

    train_loaders, val_loaders, test_loaders = get_tmnist_data(batch_size=batch_size, lens=lens)

    print(len(train_loaders[68]))
    plot_batch(train_loaders[68], lens)
    plot_batch(val_loaders[68], lens)
    plot_batch(test_loaders[68], lens)


