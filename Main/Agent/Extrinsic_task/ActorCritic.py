import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.autograd import Function


class ActorCritic(nn.Module):
    def __init__(self, action_dim, action_std, device):
        super(ActorCritic, self).__init__()

        self.device = device
        self.action_std = action_std

        self.actor = Actor(action_dim)
        self.critic = Critic(action_dim)


        self.action_var = torch.full((action_dim,), self.action_std).to(self.device)

    def forward(self):
        raise NotImplementedError

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

        return action_logprobs, torch.squeeze(state_value), dist_entropy


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


