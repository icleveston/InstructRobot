
import torch
import torch.nn as nn

from Main.Models.attention import MultiHeadAttention
from Main.Models.BlockLSTM import BlockLSTM
from Main.Models.BlockGRU import BlockGRU
from Main.Models.sparse_grad_attn import blocked_grad

from Main.Models.blocks_core_all import BlocksCore

'''
Core blocks module.  Takes:
    input: (ts, mb, h)
    hx: (ts, mb, h)
    cx: (ts, mb, h)

    output:
    output, hx, cx

'''

class Blocks(nn.Module):

    def __init__(self, ninp, nhid, nlayers, num_blocks, top_k, use_inactive, blocked_grad, step_att=True, do_gru=False):
        super(Blocks, self).__init__()
        self.nhid = nhid
        self.ninp = ninp
        self.top_k = top_k
        self.step_att = step_att
        self.do_gru = do_gru
        self.nlayers = nlayers
        self.num_blocks = num_blocks
        self.use_inactive = use_inactive
        self.blocked_grad = blocked_grad

        print("Number of Layers: ", nlayers)
        print("Input Dimension: ", ninp)
        print("Hidden Dimensions: ", nhid)
        print("Number of Blocks: ", num_blocks)
        print("Top k Blocks: ", top_k)
        print('Is the model using inactive blocks for higher representations? ', use_inactive)
        print('Is the model blocking gradients down inactive blocks? ', blocked_grad)

        self.bc_lst = []
        self.dropout_lst = []

        for i in range(nlayers):
            if i==0:

                self.bc_lst.append(BlocksCore(ninp, nhid[i], 1, num_blocks[i], top_k[i], True, do_gru=do_gru, use_higher=True))
            else:

                self.bc_lst.append(BlocksCore(nhid[i-1], nhid[i], 1, num_blocks[i], top_k[i], True, do_gru=do_gru))

        self.bc_lst = nn.ModuleList(self.bc_lst)

    def blockify_params(self):
        for i in range(self.nlayers):
            self.bc_lst[i].block_lstm.blockify_params()

    def forward(self, inp, hx, cx):
        inp_use = inp
        #print(inp_use.shape)
        hx_new, cx_new, mask_new = [],[],[]

        for idx_layer in range(self.nlayers):
            hx_, cx_ = self.bc_lst[idx_layer](inp_use, hx, cx, idx_layer)

            hx_new.append(hx_)
            cx_new.append(cx_)

            inp_use = hx_

        return hx_new, cx_new


if __name__ == "__main__":
    bc = BlocksCore(512, 1, 4, 4)

    inp = torch.randn(10, 512)
    hx = torch.randn(10,512)
    cx = torch.randn(10,512)

    hx, cx = bc(inp, hx, cx)

    print('hx cx shape', hx.shape, cx.shape)
