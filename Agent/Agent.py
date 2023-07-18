import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class Agent:
    def __init__(self, action_dim, action_std, lr, betas, gamma, k_epochs, eps_clip, total_iters, device):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = k_epochs
        self.device = device

        self.anneal_factor = 1 / total_iters

        self.policy = ActorCritic(action_dim, action_std, self.anneal_factor, self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1,
            end_factor=0,
            total_iters=total_iters
        )

        self.policy_old = ActorCritic(action_dim, action_std, self.anneal_factor, self.device).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, states):

        state_instruction = []
        state_joint_position = []
        state_vision = []
        for s in states:
            state_instruction.append(s[0])
            state_joint_position.append(s[1])
            state_vision.append(s[2])

        state_instruction = torch.stack(state_instruction)
        state_joint_position = torch.stack(state_joint_position).flatten(1, 2)
        state_vision = torch.stack(state_vision)

        return self.policy_old.act(state_instruction, state_joint_position, state_vision)

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

        # Separate states
        state_instruction = []
        state_joint_position = []
        state_vision = []
        for s in memory.states:
            state_instruction.append(s[0])
            state_joint_position.append(s[1])
            state_vision.append(s[2])

        old_instruction_states = torch.squeeze(torch.stack(state_instruction, dim=0)).detach().to(self.device)
        old_joint_position_states = torch.squeeze(torch.stack(state_joint_position, dim=0)).detach().to(self.device) \
            .flatten(1, 2)
        old_vision_states = torch.squeeze(torch.stack(state_vision, dim=0)).detach().to(self.device)

        loss = 0

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_instruction_states,
                                                                        old_joint_position_states,
                                                                        old_vision_states,
                                                                        old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs)

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss_actor = -torch.min(surr1, surr2)
            loss_entropy = - 0.01 * dist_entropy
            loss_critic = 0.5 * self.MseLoss(state_values, rewards)

            loss = loss_actor + loss_entropy + loss_critic

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Anneal parameters
        self.policy.anneal_std()
        self.scheduler.step()
        self.eps_clip -= self.eps_clip*self.anneal_factor

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        return loss_actor.mean(), loss_entropy.mean(), loss_critic.mean()


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
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, action_dim, action_std, anneal_factor, device):
        super(ActorCritic, self).__init__()

        self.action_dim = action_dim
        self.action_std = action_std
        self.anneal_factor = anneal_factor
        self.device = device
        self.action_var = torch.full((self.action_dim,), self.action_std).to(self.device)

        self.actor_instruction = Transformer()
        self.actor_joint_position = nn.Linear(3 * 26, 150)
        self.actor_vision = ConvNet(9, 250)

        self.actor = nn.Sequential(
            nn.Tanh(),
            nn.Linear(650, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim)
        )

        self.critic_vision = ConvNet(9, 250)
        self.critic_joint_position = nn.Linear(3 * 26, 150)
        self.critic_instruction = Transformer()

        self.critic = nn.Sequential(
            nn.Tanh(),
            nn.Linear(650, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state_instruction, state_joint_position, state_vision):
        x_instruction = self.actor_instruction(state_instruction)
        x_joint_position = self.actor_joint_position(state_joint_position)
        x_vision = self.actor_vision(state_vision)

        x = torch.cat((x_instruction, x_joint_position, x_vision), dim=1)

        action_mean = self.actor(x)
        cov_mat = torch.diag(self.action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat, validate_args=False)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob

    def evaluate(self, state_instruction, state_joint_position, state_vision, action):
        x_instruction_actor = self.actor_instruction(state_instruction)
        x_joint_position = self.actor_joint_position(state_joint_position)
        x_vision_actor = self.actor_vision(state_vision)

        x_actor = torch.cat((x_instruction_actor, x_joint_position, x_vision_actor), dim=1)

        action_mean = self.actor(x_actor)

        action_var = self.action_var.expand_as(action_mean).to(self.device)
        cov_mat = torch.diag_embed(action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat, validate_args=False)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        x_instruction_critic = self.critic_instruction(state_instruction)
        x_joint_position_critic = self.critic_joint_position(state_joint_position)
        x_vision_critic = self.critic_vision(state_vision)

        x_critic = torch.cat((x_instruction_critic, x_joint_position_critic, x_vision_critic), dim=1)

        state_value = self.critic(x_critic)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def anneal_std(self):
        self.action_std -= self.action_std*self.anneal_factor
        self.action_var = torch.full((self.action_dim,), self.action_std).to(self.device)


class ConvNet(nn.Module):
    def __init__(self, num_channels, num_output):
        super(ConvNet, self).__init__()

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=12, kernel_size=(5, 5))
        self.relu1 = nn.LeakyReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(5, 5))
        self.relu2 = nn.LeakyReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(in_features=10092, out_features=256)
        self.relu3 = nn.LeakyReLU()
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


class Transformer(nn.Module):
    def __init__(self, vocab_size=10, d_model=50, nhead=2, num_layers=3):
        super(Transformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)

        x = x.view(x.shape[0], -1)

        return x
