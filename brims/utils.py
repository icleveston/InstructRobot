import numpy as np
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import datetime
import shutil
import pickle
import rnn_models
import baseline_lstm_model
import random
import mixed
from mnist_seq_data_classify import mnist_data
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import argparse
import os
import json
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch




def get_args():
    # same hyperparameter scheme as word-language-model
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=0.00007,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=110,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA', default=True)
    parser.add_argument('--cudnn', action='store_true',
                        help='use cudnn optimized version. i.e. use RNN instead of RNNCell with for loop', default=True)
    parser.add_argument('--log-interval', type=int, default=750, metavar='N',
                        help='report interval')
    parser.add_argument('--algo', type=str, choices=('blocks', 'lstm', 'mixed'), default='blocks')
    parser.add_argument('--num_blocks', nargs='+', type=int, default=[6])
    parser.add_argument('--nhid', nargs='+', type=int, default=[300])
    parser.add_argument('--topk', nargs='+', type=int, default=[4])
    parser.add_argument('--block_dilation', nargs='+', type=int, default=-1)
    parser.add_argument('--layer_dilation', nargs='+', type=int, default=-1)

    parser.add_argument('--use_inactive', action='store_true',
                        help='Use inactive blocks for higher level representations too', default=True)
    parser.add_argument('--blocked_grad', action='store_true',
                        help='Block Gradients through inactive blocks', default=False)
    parser.add_argument('--multi', action='store_true',
                        help='Train for Multi MNIST')
    parser.add_argument('--scheduler', action='store_true',
                        help='Scheduler for Learning Rate', default=True)
    parser.add_argument('--test_lens', nargs='+', type=int, default=[60, 68, 76, 84, 92, 112])
    parser.add_argument('--in_size', type=int, default=60)
    parser.add_argument('--glimpse_size', type=int, default=10)
    parser.add_argument('--quant_glimpses', type=int, default=8)
    parser.add_argument('--ntokens', type=int, default=2)
    parser.add_argument('--n_out', type=int, default=10)
    parser.add_argument('--discrete_input', action='store_true', default=False)

    # parameters for adaptive softmax
    parser.add_argument('--adaptivesoftmax', action='store_true',
                        help='use adaptive softmax during hidden state to output logits.'
                             'it uses less memory by approximating softmax of large vocabulary.')
    parser.add_argument('--cutoffs', nargs="*", type=int, default=[10000, 50000, 100000],
                        help='cutoff values for adaptive softmax. list of integers.'
                             'optimal values are based on word frequencey and vocabulary size of the dataset.')

    # experiment name for this run
    #
    parser.add_argument('--name', type=str, default='results/teste_bobo',
                        help='name for this experiment. generates folder with the name if specified.')

    args = parser.parse_args()

    return args




def repackage_hidden(h, args):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if args.algo == "lstm":
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v, args) for v in h)
    hidden = []
    if args.nlayers==1:
         if isinstance(h, torch.Tensor):
             return h.detach()
         else:
             return tuple(repackage_hidden(v, args) for v in h)
    for i in range(args.nlayers):
        if isinstance(h[i], torch.Tensor):
            hidden.append(h[i].detach())
        else:
            hidden.append(tuple((h[i][0].detach(), h[i][1].detach())))
    return hidden


def plot_grad_flow(named_parameters, name, plot=True):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in named_parameters:

        #if (p.requires_grad) and ("bias" not in n):
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if (p.grad is not None):
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())

                print(f'{n}: {p.grad.abs().mean()}')
            else:
                ave_grads.append(0)
                max_grads.append(0)
                print(f'{n}: {0}')

    if plot:
        #plt.figure(figsize=(12, 8))
        fig_m = plt.gcf()
        plt.bar(np.arange(len(max_grads)), max_grads, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))

        # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions

        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

        plt.tight_layout(rect=[0.03, 0, 1, 0.95], pad=2.0, w_pad=4.0, h_pad=3.0)
        plt.show()
        plt.draw()
        fig_m.savefig(name)





def mnist_prep(x, do_upsample=True, test_upsample=-1):
    d = x
    if do_upsample:
        d = F.upsample(d.round(), size=(14,14), mode='nearest')
        d = d.reshape((d.shape[0],784//4)).round().to(dtype=torch.int64)
    else:
        d = F.upsample(d.round(), size=(test_upsample,test_upsample), mode='nearest')
        d = d.reshape((d.shape[0],test_upsample*test_upsample)).round().to(dtype=torch.int64)
    d = d.permute(1,0)
    return d

def multimnist_prep(x, do_upsample=True, test_upsample=-1):

    d = x*1.0
    if do_upsample:
        d = F.upsample(d.round(), size=(14,14), mode='nearest')
        d = d.reshape((d.shape[0],784//4)).round().to(dtype=torch.int64)
    else:
        x_small = F.upsample(x.round(), size=(14,14), mode='nearest')
        d = torch.zeros(size=(d.shape[0], 1, test_upsample, test_upsample)).float()
        a = (test_upsample - 14)//2
        d[:,:, 0:0+14, 0:0+14] = x_small
        d = d.reshape((d.shape[0],test_upsample*test_upsample)).round().to(dtype=torch.int64)

    d = d.permute(1,0)
    return d

def convert_to_multimnist(d,t, t_len=14):
    d = d.permute(1,0)
    d = d.reshape((64,1,t_len,t_len))
    perm = torch.randperm(d.shape[0])
    drand = d[perm]
    #save_image(d.data.cpu(), 'multimnist_0.png')
    #save_image(drand.data.cpu(), 'multimnist_1_preshift.png')

    for i in range(0,64):
      dr = drand[i]
      rshiftx1 = random.randint(0,3)
      rshiftx2 = random.randint(0,3)
      rshifty1 = random.randint(0,3)
      rshifty2 = random.randint(0,3)
      dr = dr[:, rshiftx1 : t_len-rshiftx2, rshifty1 : t_len-rshifty2]

      if random.uniform(0,1) < 0.5:
        padl = rshiftx1 + rshiftx2
        padr = 0
      else:
        padl = 0
        padr = rshiftx1 + rshiftx2

      if random.uniform(0,1) < 0.5:
        padt = rshifty1 + rshifty2
        padb = 0
      else:
        padt = 0
        padb = rshifty1 + rshifty2

      dr = torch.cat([torch.zeros(1,padl,dr.shape[2]).long(),dr,torch.zeros(1,padr,dr.shape[2]).long()],1)
      dr = torch.cat([torch.zeros(1,dr.shape[1], padt).long(),dr,torch.zeros(1,dr.shape[1], padb).long()],2)
      #print('dr shape', dr.shape)
      drand[i] = dr*1.0

    #save_image(drand.data.cpu(), 'multimnist_1.png')

    d = torch.clamp((d + drand),0,1)

    tr = t[perm]
    #print(t[0:5], tr[0:5])

    new_target = t*0.0

    for i in range(t.shape[0]):
      if t[i] >= tr[i]:
        nt = int(str(t[i].item()) + str(tr[i].item()))
      else:
        nt = int(str(tr[i].item()) + str(t[i].item()))

      new_target[i] += nt

    #print('new target i', new_target[0:5])
    #save_image(d.data.cpu(), 'multimnist.png')
    #raise Exception('done')

    d = d.reshape((64, t_len*t_len))
    d = d.permute(1,0)

    return d, new_target


