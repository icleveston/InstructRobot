import torch.nn as nn
from Main.Models.blocks import Blocks

class Brims(nn.Module):
    def __init__(self, ninp=256, nhid=[256], nlayers=1, dropout=0.2, num_blocks=[4], topk=[2],
                 use_inactive=False, blocked_grad=True):
        super(Brims, self).__init__()

        self.topk = topk
        self.num_blocks = num_blocks
        self.nhid = nhid
        self.use_inactive = use_inactive
        self.blocked_grad = blocked_grad
        self.nlayers = nlayers
        self.drop = nn.Dropout(dropout)

        self.model = Blocks(ninp, nhid, nlayers, num_blocks, topk, use_inactive, blocked_grad)


    def forward(self, input, hidden):
        hx, cx = hidden

        self.model.blockify_params()
        #print(f'input: {input.shape}')
        hx, cx = self.model(input, hx, cx)
        output = hx[-1]
        #print(f'output shape: {output.shape}')
        output = self.drop(output)
        hidden = (hx, cx)

        return output, hidden


    def init_hidden(self, bsz):
        hx, cx = [],[]
        weight = next(self.model.bc_lst[0].block_lstm.parameters())
        for i in range(self.nlayers):
            hx.append(weight.new_zeros(bsz, self.nhid[i]))
            cx.append(weight.new_zeros(bsz, self.nhid[i]))

        return (hx,cx)
