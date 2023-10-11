import argparse
import random
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing

from Main import Main, NormalizeInverse
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

        with (tqdm(total=self.n_steps) as pbar):

            while self.current_step < self.n_steps:

                # Train one rollout
                self.agent.policy.train()

                observations = [[] for _ in range(self.n_rollout)]
                intrinsic_frames = [[] for _ in range(self.n_rollout)]

                # For each rollout
                for r in range(self.n_rollout):

                    # Get the first observation
                    old_observation = self.env.reset()

                    for j in range(self.n_trajectory):
                        # Save observations
                        observations[r].append(old_observation.copy())

                        # Build state from observation
                        state_flatten, _ = self._build_state_from_observations(old_observation)

                        # Select action from the agent
                        action, logprob = self.agent.select_action(state_flatten)

                        # Predict the next state
                        state_pred = self.agent.predict_next_state(state_flatten, action)

                        # Execute action in the simulator
                        new_observation, ext_reward = self.env.step(action.squeeze().data.cpu().numpy())

                        # Build new state from new observation
                        _, state_intrinsic = self._build_state_from_observations(new_observation)

                        # Compute the intrinsic reward
                        int_reward, state_intrinsic_flatten = self.agent.compute_intrinsic_reward(state_intrinsic,
                                                                                                  state_pred)

                        # Save rollout to memory
                        self.memory.rewards_ext.append(ext_reward)
                        self.memory.rewards_int.append(int_reward.data.cpu().numpy())
                        self.memory.states.append(state_flatten)
                        self.memory.states_intrinsic.append(state_intrinsic_flatten)
                        self.memory.actions.append(action.squeeze())
                        self.memory.logprobs.append(logprob.squeeze())
                        self.memory.is_terminals.append(j == self.n_trajectory - 1)

                        # Append intrinsic frames
                        intrinsic_frames[r].append({"groundtruth": state_intrinsic, "prediction": state_pred})

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

                # Pack video to log
                video_info = {
                    "intrinsic_pred": self._format_intrinsic_video(intrinsic_frames)
                }

                self.current_step += self.n_trajectory * self.n_rollout

                # Process rollout conclusion
                description = self.process_rollout(loss_info, video_info, observations)

                # Set the var description
                pbar.set_description(description)

                # Update the bar
                pbar.update(self.n_trajectory * self.n_rollout)

                # Clear the memory
                self.memory.clear_memory()

        # Kill all process
        self.process_wandb.kill()

    def _build_state_from_observations(self, old_observation):

        image_tensor_rgb = torch.empty((len(old_observation), 3, 128, 128), dtype=torch.float, device=self.device)
        image_tensor_gray = torch.empty((len(old_observation), 1, 32, 128), dtype=torch.float, device=self.device)
        proprioception_tensor = torch.empty((len(old_observation), 26), dtype=torch.float, device=self.device)

        for i, o in enumerate(old_observation):
            image_top = o["frame_top"]
            image_front = o["frame_front"]

            # Convert state to tensor
            image_top_tensor_rgb = self.trans_rgb(image_top)
            image_front_tensor_rgb = self.trans_rgb(image_front)

            image_top_tensor_gray = self.trans_gray(image_top)
            image_front_tensor_gray = self.trans_gray(image_front)

            # Cat all images into a single one
            images_stacked_rgb = torch.cat((image_top_tensor_rgb, image_front_tensor_rgb), dim=1)
            images_stacked_gray = torch.cat((image_top_tensor_gray, image_front_tensor_gray), dim=2)

            image_tensor_rgb[i] = images_stacked_rgb
            image_tensor_gray[i] = images_stacked_gray

            # Save the proprioception information
            proprioception_tensor[i] = torch.tensor(o["proprioception"], device=self.device)

        state_vision_rgb = image_tensor_rgb.flatten(0, 1)
        state_proprioception = proprioception_tensor.flatten(0, 1)

        # Get last observation
        state_intrinsic = image_tensor_gray[0][-1]

        # Build state
        return (state_vision_rgb, state_proprioception), state_intrinsic

    def _format_intrinsic_video(self, intrinsic_frames) -> np.array:

        trans = transforms.Compose([
            NormalizeInverse(self.env.env_mean_gray, self.env.env_std_gray),
            transforms.ToPILImage(),
        ])

        # Random a rollout to sample
        random_sample_index = random.randint(0, len(intrinsic_frames) - 1)
        frames = intrinsic_frames[random_sample_index]

        video = []

        for index, o in enumerate(frames):

            prediction = o["prediction"].view(1, o["groundtruth"].shape[0], o["groundtruth"].shape[1])
            groundtruth = o["groundtruth"].unsqueeze(0)

            image_array = np.concatenate((trans(groundtruth), trans(prediction)), axis=0)

            image = Image.fromarray(image_array, mode="L")

            image = image.convert('RGB')

            video.append(np.asarray(image))

        video = np.asarray(video)

        video = np.transpose(video, (0, 3, 1, 2))

        return video

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
