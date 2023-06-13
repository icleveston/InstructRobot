import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from brims.blocks import Blocks



def layer_init(layer, std=np.sqrt(2), bias_const=0.0, init_w=3e-3):
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer



class AgentCuriosity(nn.Module):
    def __init__(self, num_actions, ninp=512, nhid=[512, 512], nlayers=2, dropout=0.5, num_blocks=[8, 8], topk=[4, 4], use_inactive=False, blocked_grad=False):
        super(AgentCuriosity, self).__init__()
        self.nhid = nhid
        self.topk = topk
        print('Top k Blocks: ', topk)
        self.drop = nn.Dropout(dropout)
        self.num_blocks = num_blocks
        self.ninp = ninp
        self.use_inactive = use_inactive
        self.blocked_grad = blocked_grad
        self.nlayers = nlayers
        self.num_actions = num_actions
        print("Dropout rate", dropout)
        self.encoder = nn.Sequential(nn.Linear(172, ninp), nn.ReLU())
        self.brims_p = Blocks(ninp, nhid, nlayers, num_blocks, topk, use_inactive, blocked_grad)
        #self.brims_f = Blocks(ninp, nhid, nlayers, num_blocks, topk, use_inactive, blocked_grad)
        #self.decoder = layer_init(nn.Linear(self.nhid[-1], self.nhid[-1] // 2))
        self.critic = nn.Linear(self.nhid[-1], 1)
        self.mu = layer_init(nn.Linear(self.nhid[-1], num_actions))
        self.std = layer_init(nn.Linear(self.nhid[-1], num_actions))
        self.device = torch.device("cuda")

        #self.action_var = torch.full((num_actions,), 0.5).to(self.device)
        #self.action_var = torch.tensor([0.5]*num_actions).to(self.device)

    def init_hidden_p(self, bsz):
        hx, cx = [], []
        weight = next(self.brims_p.bc_lst[0].block_lstm.parameters())
        for i in range(self.nlayers):
            hx.append(weight.new_zeros(bsz, self.nhid[i]))
            cx.append(weight.new_zeros(bsz, self.nhid[i]))
        return (hx, cx)

    ''' 
    def init_hidden_f(self, bsz):
        hx, cx = [], []
        weight = next(self.brims_f.bc_lst[0].block_lstm.parameters())
        for i in range(self.nlayers):
            hx.append(weight.new_zeros(bsz, self.nhid[i]))
            cx.append(weight.new_zeros(bsz, self.nhid[i]))
        return (hx, cx)'''


    def brims_p_blockify_params(self):
        self.brims_p.blockify_params()

    def forward(self, state, lstm_state, bz=1):
        embs = self.encoder(state)
        #embs = embs.reshape((-1, bz, 132))
        #embs = embs.view(-1, bz, 132)
        new_hidden = []
        self.brims_p_blockify_params()
        for emb in embs:
            lstm_state, _ = self.brims_p(emb, lstm_state)
            new_hidden.append(lstm_state[0][-1])
        new_hidden = torch.stack(new_hidden)
        new_hidden = new_hidden.squeeze(0)

        mu = self.mu(new_hidden)
        log_std = self.std(new_hidden)
        log_std = torch.clamp(log_std, -20, 2)
        value = self.critic(new_hidden)

        return mu, log_std, value, lstm_state

    def get_action(self, state, lstm_state):
        mean, log_std, value, lstm_state = self.forward(state, lstm_state)
        std = log_std.exp()

        dist = Normal(mean, std)
        action = dist.rsample()
        action_tanh = torch.tanh(action)

        action_logprob = dist.log_prob(action)
        action_logprob = torch.sum(action_logprob, dim=1)

        # normal = Normal(0, 1)
        # z = normal.sample()
        # action_0 = torch.tanh(mean + std * z.to(self.device))  # TanhNormal distribution as actions; reparameterization trick
        #
        # log_prob = Normal(mean, std).log_prob(mean + std * z.to(self.device)) - torch.log(
        #     1. - action_0.pow(2) + 1e-6) - np.log(1)
        # log_prob = torch.sum(log_prob)

        return action_tanh.detach().cpu().numpy().flatten(), value.detach().cpu().numpy()[-1].item(), action_logprob.detach().cpu().numpy().item(), lstm_state

    def evaluate(self, state, action, lstm_state):
        mean, log_std, value, lstm_state = self.forward(state, lstm_state)
        std = log_std.exp()

        dist = Normal(mean, std)
        action_logprob = dist.log_prob(action)
        action_logprob = torch.sum(action_logprob, dim=2)

        # normal = Normal(0, 1)
        # z = normal.sample()
        #
        # dist = Normal(mean, std)
        # log_prob = dist.log_prob(mean + std * z.to(self.device)) - torch.log(
        #     1. - action.pow(2) + 1e-6) - np.log(1)
        #
        # log_prob = torch.sum(log_prob, dim=2)

        dist_entropy = dist.entropy().mean()

        return action_logprob, value, dist_entropy

    ''' 
    def compute_intrinsic_reward(self, enc_ts, actual_brims_p, exp_hidden_f, lstm_state):
        batch_size = lstm_state[0][0].shape[0]
        input_size = lstm_state[0][0].shape[1]
        enc_ts = enc_ts.reshape((-1, batch_size, input_size))
        new_hidden_f = []
        self.brims_f_blockify_params()
        for enc_t in enc_ts:
            lstm_state, _ = self.brims_f(enc_t, lstm_state)
            new_hidden_f.append(lstm_state[0][-1])
        new_hidden_f = torch.stack(new_hidden_f)
        new_hidden_f = new_hidden_f.view(enc_ts.shape[0] * batch_size, self.nhid[-1])
        intrinsic_reward = F.mse_loss(actual_brims_p, exp_hidden_f, reduction='none').mean(-1)
        return intrinsic_reward.detach().cpu().numpy(), lstm_state, new_hidden_f'''
