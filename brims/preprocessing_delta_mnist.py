import torch
import os
import torchvision.transforms.functional as FF
import torchvision.transforms as t
import torch.nn.functional as F
from scipy import io
import numpy as np


#mat_data_test = scipy.io.loadmat('MNIST_data_test_re.mat')

#test_x = mat_data_test['X_test'][:9900, :]
#test_y = mat_data_test['Y_test'][:9900, :]

#validation_x = mat_data_test['X_test'][9900:, :]
#validation_y = mat_data_test['Y_test'][9900:, :]


dataset = io.loadmat(os.path.join('/home/brain/alana/brims/delta-MNIST/data/dMNIST/raw/', 'MNIST_data_train_re.mat'))
dataset['Y_train'] = np.argmax(dataset['Y_train'], axis=1)
np.savez(os.path.join('/home/brain/alana/brims/delta-MNIST/data/dMNIST/processed/', f'training.npz'), x=dataset['X_train'], y=dataset['Y_train'])

dataset = io.loadmat(os.path.join('/home/brain/alana/brims/delta-MNIST/data/dMNIST/raw/', 'MNIST_data_test_re.mat'))
dataset['Y_test'] = np.argmax(dataset['Y_test'], axis=1)
#print(dataset['Y_test'].shape)
np.savez(os.path.join('/home/brain/alana/brims/delta-MNIST/data/dMNIST/processed/', f'test.npz'), x=dataset['X_test'][:5000, :], y=dataset['Y_test'][:5000])
#print(dataset['Y_test'][9900:].shape)
np.savez(os.path.join('/home/brain/alana/brims/delta-MNIST/data/dMNIST/processed/', f'val.npz'), x=dataset['X_test'][5000:, :], y=dataset['Y_test'][5000:])
