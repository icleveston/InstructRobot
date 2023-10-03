import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class ActorCritic(nn.Module):
    def __init__(self, action_dim, action_std, device):
        super(ActorCritic, self).__init__()

        self.device = device
        self.action_std = action_std

        self.perception = ConvNet(12, 250)

        self.world_model = nn.Sequential(
            nn.Linear(250 + action_dim, action_dim)
        )

        self.actor = nn.Sequential(
            nn.Tanh(),
            nn.Linear(500, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Tanh(),
            nn.Linear(500, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.action_var = torch.full((action_dim,), self.action_std).to(self.device)

    def forward(self):
        raise NotImplementedError

    def next_state(self, state_vision, action):

        x = self.perception(state_vision)
        pred_state = self.world_model(torch.cat((x, action), dim=1))

        return pred_state

    def act(self, state_vision):

        x = self.perception(state_vision)
        action_mean = self.actor(x)

        cov_mat = torch.diag(self.action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat, validate_args=False)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state_vision, action):

        x = self.perception(state_vision)
        state_pred = self.world_model(torch.cat((x, action), dim=1))
        state_value = self.critic(x.detach())
        action_mean = self.actor(x.detach())

        action_var = self.action_var.expand_as(action_mean).to(self.device)
        cov_mat = torch.diag_embed(action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat, validate_args=False)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy, state_pred


class ConvNet(nn.Module):
    def __init__(self, num_channels, num_output):
        super(ConvNet, self).__init__()

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=12, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(in_features=10092, out_features=num_output)

    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x
