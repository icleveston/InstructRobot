import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from Main.Models.brims import Brims


class ActorCritic(nn.Module):
    def __init__(self, action_dim, action_std, n_rollout, n_trajectory, device):
        super(ActorCritic, self).__init__()

        self.device = device
        self.action_std = action_std
        self.n_rollout = n_rollout
        self.n_trajectory = n_trajectory
        self.action_dim = action_dim
        self.n_steps = n_rollout*n_trajectory

        self.actor = Actor(action_dim)
        self.critic = Critic(action_dim)

        self.action_var = torch.full((action_dim,), self.action_std).to(self.device)

    def forward(self):
        raise NotImplementedError

    def act(self, state_vision, state_proprioception, hidden_actor):
        action_mean, hidden_actor = self.actor(state_vision, state_proprioception, hidden_actor)
        cov_mat = torch.diag(self.action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat, validate_args=False)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob, hidden_actor

    def evaluate(self, state_vision, state_proprioception, action):
        hidden_actor = self.actor.actor_brims.init_hidden(bsz=self.n_rollout)
        action_mean = torch.zeros(self.n_steps, self.action_dim).to(self.device)
        # Actor
        for step in range(self.n_trajectory):
            state_vision_b = [state_vision[id] for id in range(step, self.n_steps, self.n_trajectory)]
            state_vision_b = torch.stack(state_vision_b)
            state_proprioception_b = [state_proprioception[id] for id in range(step, self.n_steps, self.n_trajectory)]
            state_proprioception_b = torch.stack(state_proprioception_b)
            action_mean_b, hidden_actor = self.actor(state_vision_b, state_proprioception_b, hidden_actor)

            for id in range(self.n_rollout):
                if step == 0:
                    action_mean[id * self.n_trajectory] = action_mean_b[id]
                else:
                    action_mean[(id * self.n_trajectory) + step] = action_mean_b[id]


        action_var = self.action_var.expand_as(action_mean).to(self.device)
        cov_mat = torch.diag_embed(action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat, validate_args=False)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        # Critic
        state_value = self.critic(state_vision, state_proprioception)

        return action_logprobs, torch.squeeze(state_value), dist_entropy



class Actor(nn.Module):

    def __init__(self, action_dim):
        super().__init__()

        self.actor_vision = ConvNet(9, 128)
        self.actor_proprioception = nn.Linear(3 * 26, 128)


        self.actor_brims = Brims()

        self.actor_action = nn.Sequential(
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )

    def forward(self, state_vision, state_proprioception, hidden_actor):

        x_vision = self.actor_vision(state_vision)
        x_proprioception = self.actor_proprioception(state_proprioception)

        x = torch.cat((x_vision, x_proprioception), dim=1)

        #print(f'input shape: {x.shape}')
        x, hidden_actor = self.actor_brims(x, hidden_actor)
        x = self.actor_action(x)
        return x, hidden_actor


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

        self.fc1 = nn.Linear(in_features=10092, out_features=256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=256, out_features=num_output)

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
        x = self.relu3(x)
        x = self.fc2(x)

        return x


