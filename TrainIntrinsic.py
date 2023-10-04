import argparse
import torch
from tqdm import tqdm
import multiprocessing

from Main import Main
from Main.Agent.Intrinsic import Memory
from Main.Agent.Intrinsic import Agent
from Main.Environment import Environment
from Main.Environment.CubeSimpleIntEnv import CubeSimpleIntEnv


class TrainIntrinsic(Main):

    def __init__(self, headless: bool = False, model_name: str = None, gpu: int = 0):

        # Define the agent and memory
        agent: Agent.__class__ = Agent
        memory: Memory.__class__ = Memory
        env: Environment.__class__ = CubeSimpleIntEnv

        # Initialize parent class
        super().__init__(environment=env, agent=agent, memory=memory, headless=headless, model_name=model_name, gpu=gpu)

    def train(self) -> None:

        # Start training
        self.start_train()

        with tqdm(total=self.n_steps) as pbar:

            while self.current_step < self.n_steps:

                # Train one rollout
                self.agent.policy.train()

                observations = [[] for _ in range(self.n_rollout)]

                # For each rollout
                for r in range(self.n_rollout):

                    # Get the first observation
                    old_observation = self.env.reset()

                    for j in range(self.n_trajectory):

                        # Save observations
                        observations[r].append(old_observation.copy())

                        # Build state from observation
                        _, state_flatten = self._build_state_from_observations(old_observation)

                        # Select action from the agent
                        action, logprob = self.agent.select_action(state_flatten)

                        # Predict the next state
                        state_pred = self.agent.predict_next_state(state_flatten, action)

                        # Execute action in the simulator
                        new_observation, ext_reward = self.env.step(action.squeeze().data.cpu().numpy())

                        # Build new state from new observation
                        new_state, _ = self._build_state_from_observations(new_observation)

                        # Compute the intrinsic reward
                        int_reward, state_intrinsic = self.agent.compute_intrinsic_reward(new_state, state_pred)

                        # Save rollout to memory
                        self.memory.rewards_ext.append(ext_reward)
                        self.memory.rewards_int.append(int_reward.data.cpu().numpy())
                        self.memory.states.append(state_flatten)
                        self.memory.states_intrinsic.append(state_intrinsic)
                        self.memory.actions.append(action.squeeze())
                        self.memory.logprobs.append(logprob.squeeze())
                        self.memory.is_terminals.append(j == self.n_trajectory - 1)

                        # Update observation
                        old_observation = new_observation

                # Update the weights
                loss_actor, loss_entropy, loss_critic, loss_intrinsic = self.agent.update(self.memory)

                # Pack loss into a dictionary
                loss_info = {
                    "actor": loss_actor.cpu().data.numpy(),
                    "critic": loss_critic.cpu().data.numpy(),
                    "intrinsic": loss_intrinsic.cpu().data.numpy(),
                    "entropy": loss_entropy.cpu().data.numpy()
                }

                self.current_step += self.n_trajectory * self.n_rollout

                # Process rollout conclusion
                description = self.process_rollout(loss_info, observations)

                # Set the var description
                pbar.set_description(description)

                # Update the bar
                pbar.update(self.n_trajectory * self.n_rollout)

                # Clear the memory
                self.memory.clear_memory()

        # Kill all process
        self.process_wandb.kill()

    def _build_state_from_observations(self, old_observation):

        image_tensor = torch.empty((len(old_observation), 3, 128, 128), dtype=torch.float, device=self.device)
        proprioception_tensor = torch.empty((len(old_observation), 26), dtype=torch.float, device=self.device)

        for i, o in enumerate(old_observation):
            image_top = o["frame_top"]
            image_front = o["frame_front"]

            # Convert state to tensor
            image_top_tensor = self.trans(image_top)
            image_font_tensor = self.trans(image_front)

            # Cat all images into a single one
            images_stacked = torch.cat((image_top_tensor, image_font_tensor), dim=2)

            image_tensor[i] = images_stacked

            # Save the proprioception information
            proprioception_tensor[i] = torch.tensor(o["proprioception"], device=self.device)

        state_vision = image_tensor.flatten(0, 1)
        state_proprioception = proprioception_tensor.flatten(0, 1)

        # Build state
        return (image_tensor, proprioception_tensor),  (state_vision, state_proprioception)


def parse_arguments():
    arg = argparse.ArgumentParser()

    arg.add_argument("--resume", type=str, required=False, dest='model_name',
                     help="Resume training {model_name}.")
    arg.add_argument("--gpu", type=int, default=0, required=False, help="Select the GPU card.")

    return vars(arg.parse_args())


if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')

    args = parse_arguments()

    TrainIntrinsic(headless=True, model_name=args['model_name'], gpu=args['gpu']).train()


