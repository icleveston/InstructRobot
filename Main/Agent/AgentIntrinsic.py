import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class Agent:
    def __init__(self, action_dim, action_std, lr, betas, gamma, k_epochs, eps_clip, total_iters, device):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.device = device

        self.policy = ActorCritic(action_dim, action_std, self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1,
            end_factor=0,
            total_iters=total_iters
        )

        self.policy_old = ActorCritic(action_dim, action_std, self.device).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.critic_loss = nn.MSELoss()
        self.intrinsic_loss = nn.MSELoss()

    def select_action(self, state):

        state_vision = state[0]

        state_vision = state_vision.unsqueeze(dim=0)

        return self.policy_old.act(state_vision)

    def predict_next_state(self, state, action):

        return self.policy_old.next_state(state, action).detach()

    def update(self, memory):

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(self.device)
        old_intrinsic_states = torch.squeeze(torch.stack(memory.intrinsic_states, dim=0)).detach().to(self.device)

        # Separate states
        state_instruction = []
        state_vision = []
        for s in memory.states:
            state_instruction.append(s[0])
            state_vision.append(s[1])

        old_vision_states = torch.squeeze(torch.stack(state_vision, dim=0)).detach().to(self.device)

        loss_actor = None
        loss_entropy = None
        loss_critic = None
        loss_intrinsic = None

        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy, pred_state = self.policy.evaluate(old_vision_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs)

            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss_actor = -torch.min(surr1, surr2)
            loss_entropy = -0.01 * dist_entropy
            loss_critic = 0.5 * self.critic_loss(state_values, rewards)
            loss_intrinsic = self.intrinsic_loss(old_intrinsic_states, pred_state)

            loss = loss_actor + loss_entropy + loss_critic + loss_intrinsic

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        return loss_actor.mean(), loss_entropy.mean(), loss_critic.mean(), loss_intrinsic.mean()


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.intrinsic_states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.intrinsic_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


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

