import random
import argparse
import time
import os
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import multiprocessing
from multiprocessing import Process

from Main import Main, save_wandb


class Train(Main):

    def __init__(self, headless: bool = False):
        super().__init__(headless=headless)

    def train(self, model_name: str = None) -> None:
        is_new_execution: bool = False

        # If model_name is none, create a new one
        if model_name is None:
            folder_time = time.strftime("%Y_%m_%d_%H_%M_%S")

            model_name = f"{self.conf}_{folder_time}"

            is_new_execution = True

        # Set the folders
        self.output_path = os.path.join('out', model_name)
        self.checkpoint_path = os.path.join(self.output_path, 'checkpoint')
        self.info_path = os.path.join(self.output_path, 'info')

        # Create the folders
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        if not os.path.exists(self.info_path):
            os.makedirs(self.info_path)

        # If it is not a new execution, load the model
        if not is_new_execution:
            self._load_checkpoint(load_best_checkpoint=False)

        # Start wandb process
        self.process_wandb = Process(target=save_wandb, args=(self.in_queues_wandb,
                                                              is_new_execution,
                                                              self.conf,
                                                              model_name,
                                                              self.info_path,
                                                              self.checkpoint_path))
        self.process_wandb.start()

        print(f"[*] Output Folder: {model_name}")
        print(f"[*] Total Trainable Params: {self.num_parameters}")

        tic = time.time()
        with tqdm(total=self.n_steps) as pbar:

            while self.current_step < self.n_steps:
                # Train one rollout
                return_info, loss_info, obs_rollout = self._train_one_rollout()

                self.current_step += self.n_trajectory * self.n_rollout

                # Measure elapsed time
                toc = time.time()

                # Set the var description
                pbar.set_description(("{:.1f}s "
                                      "- step: {:.1f} "
                                      "- loss actor: {:.6f} "
                                      "- loss critic: {:.6f} "
                                      "- loss entropy: {:.6f} "
                                      "- mean return: {:.6f}".
                                      format((toc - tic),
                                             self.current_step,
                                             loss_info["actor"],
                                             loss_info["critic"],
                                             loss_info["entropy"],
                                             return_info["mean"])))

                # Update the bar
                pbar.update(self.current_step)

                # Get states info
                states_info = {
                    'agent_state': self.agent.policy.state_dict(),
                    'optim_state': self.agent.optimizer.state_dict(),
                    'scheduler_state': self.agent.scheduler.state_dict()
                }

                # Get params info
                params_info = {
                    'best_mean_episodic_return': self.best_mean_episodic_return,
                    'eps_clip': self.agent.eps_clip,
                    'lr': self.agent.scheduler.get_last_lr()[0],
                    'action_std': self.agent.policy.action_std,
                    'step': self.current_step
                }

                # Check if it is the best model
                self.best_mean_episodic_return = max(return_info["mean"], self.best_mean_episodic_return)

                # Send data to wandb process
                self.in_queues_wandb.put((return_info, loss_info, params_info, states_info, obs_rollout))

                self.in_queues_wandb.join()

                toc = time.time()

                self.elapsed_time = toc - tic

                # Save the configuration
                self._save_config()

        # Kill all process
        self.process_wandb.kill()

    def _train_one_rollout(self) -> tuple:
        self.agent.policy.train()

        obs_rollout = []

        # Random a rollout to sample
        random_sample_index = random.randint(0, self.n_rollout - 1)

        # For each rollout
        for r in range(self.n_rollout):

            # Get the first observation
            old_observation = self.env.reset()

            for j in range(self.n_trajectory):

                # Save observations
                if r == random_sample_index:
                    obs_rollout.append(old_observation.copy())

                # Tokenize instruction
                instruction_token = self.tokenizer(old_observation[-1][0])

                # Get instructions indexes
                instruction_index = torch.tensor(self.vocab(instruction_token), device=self.device)

                image_tensor = torch.empty((len(old_observation), 3, 128, 128), dtype=torch.float, device=self.device)

                for i, o in enumerate(old_observation):
                    image_top = o[1]
                    image_front = o[2]

                    # Convert state to tensor
                    image_top_tensor = self.trans(image_top)
                    image_font_tensor = self.trans(image_front)

                    # Cat all images into a single one
                    images_stacked = torch.cat((image_top_tensor, image_font_tensor), dim=2)

                    image_tensor[i] = images_stacked

                image = image_tensor.flatten(0, 1)

                # Build state
                state = (instruction_index, image)

                # Select action from the agent
                action, logprob = self.agent.select_action(state)

                # Execute action in the simulator
                new_observation, reward = self.env.step(action.squeeze().data.cpu().numpy())

                # Save rollout to memory
                self.memory.rewards.append(reward)
                self.memory.states.append(state)
                self.memory.actions.append(action.squeeze())
                self.memory.logprobs.append(logprob.squeeze())
                self.memory.is_terminals.append(j == self.n_trajectory - 1)

                # Update observation
                old_observation = new_observation

        # Reshape rewards to compute the std
        rewards_rollouts = np.mean(np.reshape(np.array(self.memory.rewards), (self.n_rollout, -1)), axis=1)

        # Compute the mean and std episodic return
        return_info = {
            "mean": np.mean(np.array(self.memory.rewards)),
            "std": np.std(rewards_rollouts)
        }

        # Update the weights
        loss_actor, loss_entropy, loss_critic = self.agent.update(self.memory)

        # Pack loss into a dictionary
        loss_info = {
            "actor": loss_actor.cpu().data.numpy(),
            "critic": loss_critic.cpu().data.numpy(),
            "entropy": loss_entropy.cpu().data.numpy()
        }

        # Clear the memory
        self.memory.clear_memory()

        return return_info, loss_info, obs_rollout

    def _save_config(self):

        # Save the configuration into a dataframe
        df = pd.DataFrame()
        df['n_steps'] = [self.n_steps]
        df['n_rollout'] = [self.n_rollout]
        df['n_trajectory'] = [self.n_trajectory]
        df['# Params'] = [self.num_parameters]
        df['lr'] = [self.lr]
        df['random_seed'] = [self.random_seed]
        df['elapsed_time'] = [time.strftime("%H:%M:%S", time.gmtime(self.elapsed_time))]
        df = df.astype(str)

        # Build the config path
        config_path: str = os.path.join(self.info_path, "config.csv")

        # Save the dataframe as csv
        df.to_csv(config_path)


def parse_arguments():
    arg = argparse.ArgumentParser()

    arg.add_argument("--resume", type=str, required=False, dest='model_name',
                     help="Resume training {model_name}.")

    return vars(arg.parse_args())


if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')

    args = parse_arguments()

    Train(headless=True).train(args['model_name'])


