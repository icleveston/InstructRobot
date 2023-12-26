import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.autograd import Function


class ActorCritic(nn.Module):
    def __init__(self, action_dim, action_std, env_min_values, env_max_values, device):
        super(ActorCritic, self).__init__()

        self.device = device
        self.action_std = action_std

        self.actor = Actor(action_dim)
        self.critic = Critic(action_dim)
        self.intrinsic = Intrinsic(env_min_values, env_max_values)

        self.action_var = torch.full((action_dim,), self.action_std).to(self.device)

    def forward(self):
        raise NotImplementedError

    def next_state(self, state_vision, state_proprioception, action):
        state_pred = self.intrinsic(state_vision, state_proprioception, action)

        return state_pred.detach()

    def act(self, state_vision, state_proprioception):

        action_mean = self.actor(state_vision, state_proprioception)

        cov_mat = torch.diag(self.action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat, validate_args=False)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob

    def evaluate(self, state_vision, state_proprioception, action):

        # Actor
        action_mean = self.actor(state_vision, state_proprioception)

        action_var = self.action_var.expand_as(action_mean).to(self.device)
        cov_mat = torch.diag_embed(action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat, validate_args=False)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        # Critic
        state_value = self.critic(state_vision, state_proprioception)

        # Intrinsic
        state_pred = self.intrinsic(state_vision, state_proprioception, action)

        return action_logprobs, torch.squeeze(state_value), dist_entropy, state_pred


class ConvNet(nn.Module):
    def __init__(self, num_channels, num_output):
        super(ConvNet, self).__init__()

        self.dconv_down1 = double_conv(num_channels, 12)
        self.dconv_down2 = double_conv(12, 9)
        self.dconv_down3 = double_conv(9, 3)

        self.maxpool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(in_features=768, out_features=num_output)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class Actor(nn.Module):

    def __init__(self, action_dim):
        super().__init__()

        self.actor_vision = ConvNet(9, 128)
        self.actor_proprioception = nn.Linear(3 * 26, 128)

        self.actor = nn.Sequential(
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state_vision, state_proprioception):

        x_vision = self.actor_vision(state_vision)
        x_proprioception = self.actor_proprioception(state_proprioception)

        x = torch.cat((x_vision, x_proprioception), dim=1)

        return self.actor(x)


class Critic(nn.Module):

    def __init__(self, action_dim):
        super().__init__()

        self.critic_vision = ConvNet(9, 128)
        self.critic_proprioception = nn.Linear(3 * action_dim, 128)

        self.critic = nn.Sequential(
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, state_vision, state_proprioception):

        x_vision_critic = self.critic_vision(state_vision)
        x_proprioception_critic = self.critic_proprioception(state_proprioception)
        x_critic = torch.cat((x_vision_critic, x_proprioception_critic), dim=1)

        return self.critic(x_critic)


class Intrinsic(nn.Module):

    def __init__(self, env_min_values, env_max_values):
        super().__init__()

        self.env_min_values = env_min_values
        self.env_max_values = env_max_values

        self.proprioception = nn.Linear(3 * 26, 512)
        self.action = nn.Linear(26, 512)

        self.dconv_down1 = double_conv(9, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(128 + 128 + 1, 128)
        self.dconv_up2 = double_conv(64 + 128, 64)
        self.dconv_up1 = double_conv(64 + 32, 16)

        self.conv_last = nn.Conv2d(16, 3, 1)

    def forward(self, state_vision, state_proprioception, action):
        p = self.proprioception(state_proprioception)
        a = self.action(action)

        t = torch.stack((p, a), dim=1)

        t = t.reshape(-1, 1, 32, 32)

        conv1 = self.dconv_down1(state_vision)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.upsample(x)

        x = torch.cat([x, t, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        y = self.conv_last(x)

        out = torch.stack(
            [F.hardtanh(y[:, i, :, :], min_val=min_value, max_val=max_value) for i, (min_value, max_value) in
             enumerate(zip(self.env_min_values, self.env_max_values))]).permute(1, 0, 2, 3)

        return out


class Intrinsic2(nn.Module):
    def __init__(self, env_min_values, env_max_values, C=64, M=192, in_chan=9, out_chan=3):
        super(Intrinsic2, self).__init__()

        self.env_min_values = env_min_values
        self.env_max_values = env_max_values

        self.proprioception = nn.Linear(3 * 26, 256)
        self.action = nn.Linear(26, 256)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=M, kernel_size=5, stride=2, padding=2, bias=False),
            GDN(M),
            nn.Conv2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, bias=False),
            GDN(M),
            nn.Conv2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, bias=False),
            GDN(M),
            nn.Conv2d(in_channels=M, out_channels=C, kernel_size=5, stride=2, padding=2, bias=False)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=C + 8, out_channels=M, kernel_size=5, stride=2, padding=2, output_padding=1,
                               bias=False),
            GDN(M, inverse=True),
            nn.ConvTranspose2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, output_padding=1,
                               bias=False),
            GDN(M, inverse=True),
            nn.ConvTranspose2d(in_channels=M, out_channels=M, kernel_size=5, stride=2, padding=2, output_padding=1,
                               bias=False),
            GDN(M, inverse=True),
            nn.ConvTranspose2d(in_channels=M, out_channels=out_chan, kernel_size=5, stride=2, padding=2,
                               output_padding=1, bias=False)
        )

    def forward(self, state_vision, state_proprioception, action):
        p = self.proprioception(state_proprioception)
        a = self.action(action)

        t = torch.stack((p, a), dim=1)

        t = t.reshape(-1, 8, 8, 8)

        code = self.encoder(state_vision)

        x = torch.cat([code, t], dim=1)

        y = self.decoder(x)

        out = torch.stack(
            [F.hardtanh(y[:, i, :, :], min_val=min_value, max_val=max_value) for i, (min_value, max_value) in
             enumerate(zip(self.env_min_values, self.env_max_values))]).permute(1, 0, 2, 3)

        return out


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        ctx.save_for_backward(inputs, inputs.new_ones(1) * bound)
        return inputs.clamp(min=bound)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, bound = ctx.saved_tensors

        pass_through_1 = (inputs >= bound)
        pass_through_2 = (grad_output < 0)

        pass_through = (pass_through_1 | pass_through_2)
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    def __init__(self,
                 num_features,
                 inverse=False,
                 gamma_init=.1,
                 beta_bound=1e-6,
                 gamma_bound=0.0,
                 reparam_offset=2 ** -18,
                 ):
        super(GDN, self).__init__()
        self._inverse = inverse
        self.num_features = num_features
        self.reparam_offset = reparam_offset
        self.pedestal = self.reparam_offset ** 2

        beta_init = torch.sqrt(torch.ones(num_features, dtype=torch.float) + self.pedestal)
        gama_init = torch.sqrt(torch.full((num_features, num_features), fill_value=gamma_init, dtype=torch.float)
                               * torch.eye(num_features, dtype=torch.float) + self.pedestal)

        self.beta = nn.Parameter(beta_init)
        self.gamma = nn.Parameter(gama_init)

        self.beta_bound = (beta_bound + self.pedestal) ** 0.5
        self.gamma_bound = (gamma_bound + self.pedestal) ** 0.5

    def _reparam(self, var, bound):
        var = LowerBound.apply(var, bound)
        return (var ** 2) - self.pedestal

    def forward(self, x):
        gamma = self._reparam(self.gamma, self.gamma_bound).view(self.num_features, self.num_features, 1, 1)
        beta = self._reparam(self.beta, self.beta_bound)
        norm_pool = F.conv2d(x ** 2, gamma, bias=beta, stride=1, padding=0)
        norm_pool = torch.sqrt(norm_pool)

        if self._inverse:
            norm_pool = x * norm_pool
        else:
            norm_pool = x / norm_pool
        return norm_pool
