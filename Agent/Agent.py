import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


#     lr_actor = 0.00003       # learning rate for actor network
#     lr_critic = 0.0001       # learning rate for critic network


class Agent:
    def __init__(self, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip, device):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.policy = ActorCritic(action_dim, action_std, self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(action_dim, action_std, self.device).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        return self.policy_old.act(state)

    def update(self, memory):
        memory.rewards = [torch.tensor(r, dtype=torch.float) for r in memory.rewards]

        rewards = torch.stack(memory.rewards).to(self.device)

        memory.states = [torch.stack(s) for s in memory.states]
        memory.actions = [torch.stack(a) for a in memory.actions]
        memory.logprobs = [torch.stack(l) for l in memory.logprobs]

        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()

        rewards = rewards.transpose(0, 1)
        old_states = old_states.transpose(0, 1)
        old_actions = old_actions.transpose(0, 1)
        old_logprobs = old_logprobs.transpose(0, 1)

        loss = 0

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss_rl = (-torch.min(surr1, surr2) - 0.01 * dist_entropy).sum(dim=0).mean()

            loss = loss_rl + 0.5 * self.MseLoss(state_values, rewards)

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        return loss


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]


class ActorCritic(nn.Module):
    def __init__(self, action_dim, action_std, device):
        super(ActorCritic, self).__init__()

        self.device = device

        self.actor = nn.Sequential(
            ConvNet(12, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
        )

        self.critic = nn.Sequential(
            ConvNet(12, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        self.action_var = torch.full((action_dim,), action_std * action_std).to(self.device)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.rsample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean).to(self.device)
        cov_mat = torch.diag_embed(action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class ConvNet(nn.Module):
    def __init__(self, num_channels, num_output):
        super(ConvNet, self).__init__()

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=20, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(in_features=3721, out_features=500)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=500, out_features=num_output)

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

        output = self.fc2(x)

        return output
