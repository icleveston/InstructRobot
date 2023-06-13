import torch.nn as nn
import torch
from attention import MultiHeadAttention
from layer_conn_attention import LayerConnAttention
from BlockLSTM import BlockLSTM
import torch.nn.functional as F
import random
import time
from GroupLinearLayer import GroupLinearLayer
from sparse_grad_attn import blocked_grad
from torch.autograd import Variable

from blocks import Blocks

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ninp, nhid, nlayers, dropout=0.5, num_blocks=[6], topk=[4], use_inactive=False, blocked_grad=False):
        super(RNNModel, self).__init__()

        self.nhid = nhid
        self.topk = topk
        print('Top k Blocks: ', topk)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Linear(self.glimpse_size*self.glimpse_size + 156, ninp)
        self.num_blocks = num_blocks
        self.nhid = nhid

        self.sigmoid = nn.Sigmoid()
        self.sm = nn.Softmax(dim=1)
        self.use_inactive = use_inactive
        self.blocked_grad = blocked_grad
        self.nlayers = nlayers


        print("Dropout rate", dropout)

        self.init_weights()

        self.model = Blocks(ninp, nhid, nlayers, num_blocks, topk, use_inactive, blocked_grad)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        hx, cx = hidden

        self.model.blockify_params()

        hx, cx = self.model(emb, hx, cx, idx_step)


        hidden = (hx,cx)




        return hidden


    def init_hidden(self, bsz):
        hx, cx = [],[]
        weight = next(self.model.bc_lst[0].block_lstm.parameters())
        for i in range(self.nlayers):
            hx.append(weight.new_zeros(bsz, self.nhid[i]))
            cx.append(weight.new_zeros(bsz, self.nhid[i]))

        return (hx,cx)


