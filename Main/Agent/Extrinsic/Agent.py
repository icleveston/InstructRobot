import torch
import torch.nn as nn
from .ActorCritic import ActorCritic
from .Memory import Memory


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

    def select_action(self, state):

        state_instruction = state[0]
        state_vision = state[1]
        state_proprioception = state[2]

        state_instruction = state_instruction.unsqueeze(dim=0)
        state_vision = state_vision.unsqueeze(dim=0)
        state_proprioception = state_proprioception.unsqueeze(dim=0)

        return self.policy_old.act(state_instruction, state_vision, state_proprioception)

    def update(self, memory: Memory):

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
        state_vision = []
        state_proprioception = []
        for s in memory.states:
            state_instruction.append(s[0])
            state_vision.append(s[1])
            state_proprioception.append(s[2])

        old_instruction_states = torch.squeeze(torch.stack(state_instruction, dim=0)).detach().to(self.device)
        old_vision_states = torch.squeeze(torch.stack(state_vision, dim=0)).detach().to(self.device)
        old_proprioception_states = torch.squeeze(torch.stack(state_proprioception, dim=0)).detach().to(self.device)

        loss_actor = None
        loss_entropy = None
        loss_critic = None

        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_instruction_states,
                                                                        old_vision_states,
                                                                        old_proprioception_states,
                                                                        old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs)

            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss_actor = -torch.min(surr1, surr2)
            loss_entropy = -0.01 * dist_entropy
            loss_critic = 0.5 * self.critic_loss(state_values, rewards)

            loss = loss_actor + loss_entropy + loss_critic

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        return loss_actor.mean(), loss_entropy.mean(), loss_critic.mean()

