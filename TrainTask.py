import argparse
from tqdm import tqdm
import multiprocessing
import os
import shutil
import torch
import wandb
import numpy as np
import random
import time
from prettytable import PrettyTable
from torchvision import transforms
from PIL import Image, ImageFont, ImageDraw
from csv import writer
from multiprocessing import Process, JoinableQueue
from Utils import NormalizeInverse
from Main.Agent.Extrinsic_task import Memory
from Main.Agent.Extrinsic_task import Agent
from Main.Environment.CubeSimpleExtEnv import CubeSimpleExtEnv

import warnings
warnings.filterwarnings('ignore')

torch.set_printoptions(threshold=10_000)

torch.set_printoptions(profile="full", precision=10, linewidth=100, sci_mode=False)

class TrainTask():

    def __init__(self, headless: bool = False, model_name: str = None, model_intrinsic: str = None, gpu: int = 0):
        # Set the default cuda card
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        # Training params
        self.model_name = model_name
        self.model_intrinsic = model_intrinsic

        self.n_steps = 1_000_000
        self.n_rollout = 12
        self.n_trajectory = 32
        self.current_step = 0
        self.lr = 1e-4
        self.action_dim = 26
        self.action_std = 0.5
        self.betas = (0.9, 0.999)
        self.gamma = 0.99
        self.k_epochs = 20
        self.eps_clip = 0.25

        # Other params
        self.random_seed = 1
        self.best_mean_episodic_return = 0
        self.elapsed_time = 0
        self.num_parameters = 0
        self.output_path = None
        self.info_path = None
        self.checkpoint_path = None
        self.tic = None
        self.process_wandb = None
        self.in_queues_wandb = None
        self.trans_rgb = None
        self.trans_inverse_rgb = None

        # Set the seed
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)

        # Check if the gpu is available
        if torch.cuda.is_available():

            self.device = torch.device("cuda")

            torch.cuda.manual_seed(self.random_seed)

        else:
            self.device = torch.device("cpu")

        print(f"\n[*] Device: {self.device}")

        # Build the environment
        self.env = CubeSimpleExtEnv(
            headless=headless,
            random_seed=self.random_seed
        )

        self.agent = Agent(
            env=self.env,
            action_dim=self.action_dim,
            action_std=self.action_std,
            lr=self.lr,
            betas=self.betas,
            gamma=self.gamma,
            k_epochs=self.k_epochs,
            eps_clip=self.eps_clip,
            total_iters=self.n_steps // (self.n_rollout * self.n_trajectory),
            device=self.device
        )

        # Build the agent's memory
        self.memory = Memory()

    def train(self) -> None:

        # Start training
        self._start_train()

        with (tqdm(total=self.n_steps) as pbar):

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
                        state_flatten, _ = self._build_state_from_observations(old_observation)

                        # Select action from the agent
                        action, logprob = self.agent.select_action(state_flatten)

                        # Execute action in the simulator
                        new_observation, ext_reward = self.env.step(action.squeeze().data.cpu().numpy())


                        # Save rollout to memory
                        self.memory.rewards.append(ext_reward)
                        self.memory.states.append(state_flatten)
                        self.memory.actions.append(action.squeeze())
                        self.memory.logprobs.append(logprob.squeeze())
                        self.memory.is_terminals.append(j == self.n_trajectory - 1)

                        # Update observation
                        old_observation = new_observation

                # Update the weights
                loss_actor, loss_entropy, loss_critic = self.agent.update(self.memory)

                # Pack loss into a dictionary
                loss_info = {
                    "actor": loss_actor.cpu().data.numpy(),
                    "critic": loss_critic.cpu().data.numpy(),
                    "entropy": loss_entropy.cpu().data.numpy()
                }

                self.current_step += self.n_trajectory * self.n_rollout

                # Process rollout conclusion
                description = self._process_rollout(loss_info, observations)

                # Set the var description
                pbar.set_description(description)

                # Update the bar
                pbar.update(self.n_trajectory * self.n_rollout)

                # Clear the memory
                self.memory.clear_memory()

        # Kill all process
        self.process_wandb.kill()

    def _start_train(self) -> None:

        is_new_execution: bool = False
        # If model_name is none, create a new one
        if self.model_intrinsic is None:
            raise ValueError("model_intrinsic must be specified.")

        # Set the folders
        self.output_path = os.path.join('out', self.model_intrinsic)
        self.checkpoint_path = os.path.join(self.output_path, 'checkpoint')
        self.info_path = os.path.join(self.output_path, 'info')

        self._load_checkpoint_intrinsic()

        if self.model_name is None:
            is_new_execution = True
            print(f'new execution!!!')
            folder_time = time.strftime("%Y_%m_%d_%H_%M_%S")

            self.model_name = f"{self.model_intrinsic}_execution_{folder_time}_task"

            self.output_path = os.path.join('out', self.model_name)
            self.checkpoint_path = os.path.join(self.output_path, 'checkpoint')
            self.info_path = os.path.join(self.output_path, 'info')

            # Create the folders
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if not os.path.exists(self.info_path):
                os.makedirs(self.info_path)
        else:
            print(f'not new execution')
            self.output_path = os.path.join('out', self.model_name)
            self.checkpoint_path = os.path.join(self.output_path, 'checkpoint')
            self.info_path = os.path.join(self.output_path, 'info')
            self._load_checkpoint(load_best_checkpoint=False)

        actor_train = ['actor.3.weight', 'actor.3.bias']
        for name, param in self.agent.policy.actor.named_parameters():
            param.requires_grad = True if name in actor_train else False
        critic_train = ['critic.3.weight', 'critic.3.bias']
        for name, param in self.agent.policy.critic.named_parameters():
            param.requires_grad = True if name in critic_train else False


        # Compose the transformations
        self.trans_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 128)),
            transforms.Normalize(self.env.env_mean_rgb, self.env.env_std_rgb)
        ])

        self.trans_inverse_rgb = transforms.Compose([
            NormalizeInverse(self.env.env_mean_rgb, self.env.env_std_rgb),
        ])

        # Wandb queue and process
        self.in_queues_wandb = JoinableQueue()
        self.process_wandb = None

        # Start wandb process
        self.process_wandb = Process(target=_save_wandb, args=(self.in_queues_wandb,
                                                               is_new_execution,
                                                               str(self.env),
                                                               self.model_name,
                                                               self.info_path,
                                                               self.checkpoint_path))
        self.process_wandb.start()

        # Count the number of the model parameters
        self._count_parameters()

        # Build the config params
        config_info = {
            'n_steps': self.n_steps,
            'n_rollout': self.n_rollout,
            'n_trajectory': self.n_trajectory,
            'k_epochs': self.k_epochs,
            'gamma': self.gamma,
            'env_mean_rgb': self.env.env_mean_rgb.tolist(),
            'env_std_rgb': self.env.env_std_rgb.tolist(),
            "num_parameters": self.num_parameters,
            "random_seed": self.random_seed
        }

        # Save configuration params
        _save_csv(os.path.join(self.info_path, 'config.csv'), config_info)

        print(f"[*] Output Folder: {self.model_name}")
        print(f"[*] Total Trainable Params: {self.num_parameters}")

        # Start timer
        self.tic = time.time()



    def _build_state_from_observations(self, old_observation):

        image_tensor_rgb = torch.empty((len(old_observation), 3, 128, 128), dtype=torch.float, device=self.device)
        proprioception_tensor = torch.empty((len(old_observation), 26), dtype=torch.float, device=self.device)

        for i, o in enumerate(old_observation):
            image_top = o["frame_top"]
            image_front = o["frame_front"]

            # Convert state to tensor
            image_top_tensor_rgb = self.trans_rgb(image_top)
            image_front_tensor_rgb = self.trans_rgb(image_front)

            # Cat all images into a single one
            images_stacked_rgb = torch.cat((image_top_tensor_rgb, image_front_tensor_rgb), dim=1)

            image_tensor_rgb[i] = images_stacked_rgb

            # Save the proprioception information
            proprioception_tensor[i] = torch.as_tensor(o["proprioception"], device=self.device)

        state_vision_rgb = image_tensor_rgb.flatten(0, 1)
        state_proprioception = proprioception_tensor.flatten(0, 1)

        # Get last observation
        state_intrinsic = image_tensor_rgb[-1].unsqueeze(0)

        # Build state
        return (state_vision_rgb, state_proprioception), state_intrinsic

    def _process_rollout(self, loss_info: dict, observations) -> str:

        # Measure elapsed time
        self.elapsed_time = time.time() - self.tic

        # Create loss description
        loss_description = ["- loss " + loss_name + ": {:.6f} " for loss_name in [*loss_info.keys()]]

        description = [
            "{:.1f}s ",
            "- step: {:.1f} ",
            *loss_description,
            "- mean return: {:.6f}"
        ]

        # Get states info
        states_info = {
            'actor_state': self.agent.policy.actor.state_dict(),
            'critic_state': self.agent.policy.critic.state_dict(),
            'optim_state': self.agent.optimizer.state_dict(),
            'scheduler_state': self.agent.scheduler.state_dict()
        }

        # Get params info
        params_info = {
            'best_mean_episodic_return': self.best_mean_episodic_return,
            'eps_clip': self.agent.eps_clip,
            'lr': self.agent.scheduler.get_last_lr()[0],
            'action_std': self.agent.policy.action_std,
            'step': self.current_step,
            "elapsed_time": time.strftime("%H:%M:%S", time.gmtime(self.elapsed_time))
        }

        # Compute mean and std
        mean_std_info = self._compute_rollout_mean_std()

        # Check if it is the best model
        self.best_mean_episodic_return = max(mean_std_info["mean"], self.best_mean_episodic_return)

        # Send data to wandb process
        self.in_queues_wandb.put((mean_std_info, loss_info, params_info, states_info, observations))
        self.in_queues_wandb.join()

        # Return description
        return "".join(description).format(self.elapsed_time,
                                           self.current_step,
                                           *loss_info.values(),
                                           mean_std_info["mean"])

    def _compute_rollout_mean_std(self) -> dict:

        # Reshape rewards to compute the std
        rewards_rollouts = np.mean(np.reshape(np.array(self.memory.rewards), (self.n_rollout, -1)), axis=1)

        # Compute the mean and std episodic return
        return {
            "mean": np.mean(np.array(self.memory.rewards)),
            "std": np.std(rewards_rollouts)
        }

    def _count_parameters(self, print_table=False):

        table = PrettyTable(["Modules", "Parameters"])

        for name, parameter in self.agent.policy.named_parameters():

            if parameter.requires_grad:
                param = parameter.numel()
                table.add_row([name, param])

                self.num_parameters += param

        if print_table:
            print(table)

    def _load_checkpoint(self, load_best_checkpoint=True):

        print(f"[*] Loading checkpoint from {self.checkpoint_path}")

        # Define which checkpoint to load
        filename = "best_checkpoint.pth" if load_best_checkpoint else "last_checkpoint.pth"

        # Set the checkpoint path
        checkpoint_path = os.path.join(self.checkpoint_path, filename)

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)

        # Load the variables from checkpoint
        self.current_step = checkpoint["step"]
        self.best_mean_episodic_return = checkpoint["best_mean_episodic_return"]
        self.agent.policy.actor.load_state_dict(checkpoint["actor_state"])
        self.agent.policy.critic.load_state_dict(checkpoint["critic_state"])
        self.agent.optimizer.load_state_dict(checkpoint["optim_state"])
        self.agent.scheduler.load_state_dict(checkpoint["scheduler_state"])

        if load_best_checkpoint:
            print(f"[*] Loaded best checkpoint @ step {self.current_step}")
        else:
            print(f"[*] Loaded last checkpoint @ step {self.current_step}")

    def _load_checkpoint_intrinsic(self):

        print(f"[*] Loading checkpoint from {self.checkpoint_path}")

        # Define which checkpoint to load
        filename = "last_checkpoint.pth"

        # Set the checkpoint path
        checkpoint_path = os.path.join(self.checkpoint_path, filename)

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)

        # Load the variables from checkpoint
        self.agent.policy.actor.load_state_dict(checkpoint["actor_state"])
        self.agent.policy.critic.load_state_dict(checkpoint["critic_state"])

        print(f"[*] Loaded last checkpoint intrinsic model @ step {self.current_step}")


def _percentage_error_formula(x: float, amount_variation: float) -> float:
    return round(x / amount_variation * 100, 3)


def _save_wandb(in_queue,
                is_new_execution: bool,
                env_name: str,
                model_name: str,
                info_path: str,
                checkpoint_path: str) -> None:
    # Init Wandb
    wandb.init(
        project=env_name,
        name=model_name,
        id=model_name,
        resume=model_name if not is_new_execution else False
    )

    while True:
        data = in_queue.get()
        in_queue.task_done()

        # Unpack data
        mean_std_info, loss_info, params_info, states_info, observations = data

        # Random a rollout to sample
        random_sample_index = random.randint(0, len(observations) - 1)
        observation = observations[random_sample_index]

        # Create loss description
        loss_description = {f"loss_{loss_name}": loss_value for loss_name, loss_value in loss_info.items()}

        # Create the dynamic video log


        # Create the wandb log
        wandb_log = {
            "mean_episodic_return": mean_std_info["mean"],
            "lr": np.float32(params_info["lr"]),
            "action_std": np.float32(params_info["action_std"]),
            "eps": np.float32(params_info["eps_clip"]),
            "video_front_top": wandb.Video(_format_video_wandb(observation), fps=8),
            # f"actions-{current_step}": wandb.Table(columns=[f"a{i}" for i in range(26)], data=actions)
        }

        # Update log with other attributes
        wandb_log.update(loss_description)

        # Log Wandb
        wandb.log(wandb_log, step=params_info["step"])

        # Join all information
        loss_info.update(params_info)
        loss_info.update(mean_std_info)

        # Save training history
        _save_csv(os.path.join(info_path, 'history.csv'), loss_info)

        # Check if it is the best rollout
        is_best_rollout: bool = mean_std_info["mean"] > params_info["best_mean_episodic_return"]

        # Save the observations for each rollout
        _save_observation(info_path, observations, is_best_rollout)

        # Include states info into params
        params_info.update(states_info)

        # Save the checkpoint for each rollout
        _save_checkpoint(checkpoint_path, params_info, is_best_rollout)


def _save_observation(info_path: str, observations, is_best_rollout: bool):
    # Set the observation path
    obs_path = os.path.join(info_path, "last_observation.tar")

    # Save the last observation data
    torch.save(observations, obs_path)

    # Save the best observation
    if is_best_rollout:
        # Copy the last observation to the best observation
        shutil.copyfile(obs_path, os.path.join(info_path, "best_observation.tar"))


def _save_csv(file_path: str, info: dict) -> None:
    # Check if file already exists
    file_exists: bool = os.path.isfile(file_path)

    # Open file
    with open(file_path, 'a') as f_object:
        writer_object = writer(f_object)

        # Write header if file does not exist
        if not file_exists:
            writer_object.writerow([*info.keys()])

        # Write row
        writer_object.writerow([
            *info.values()
        ])

        f_object.close()


def _format_video_wandb(last_obs_rollout) -> np.array:
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 64)),
        transforms.ToPILImage(),
    ])

    video = []

    font = ImageFont.truetype("Main/Roboto/Roboto-Medium.ttf", size=10)

    for index, o in enumerate(last_obs_rollout):

        image_array = np.concatenate((trans(o[-1]["frame_top"]), trans(o[-1]["frame_front"])), axis=0)

        image = Image.fromarray(image_array, mode="RGB")

        if "instruction" in o[-1].keys():
            image_editable = ImageDraw.Draw(image)
            image_editable.text((10, 115), o[-1]["instruction"], (0, 0, 0), align="center", font=font)

        video.append(np.asarray(image))

    video = np.asarray(video)
    video = np.transpose(video, (0, 3, 1, 2))

    return video


def _save_checkpoint(checkpoint_path: str, checkpoint: dict, is_best_checkpoint: bool) -> None:
    # Set the checkpoint path
    ckpt_path = os.path.join(checkpoint_path, "last_checkpoint.pth")

    # Save the checkpoint
    torch.save(checkpoint, ckpt_path)

    # Save the best checkpoint
    if is_best_checkpoint:
        # Copy the last checkpoint to the best checkpoint
        shutil.copyfile(ckpt_path, os.path.join(checkpoint_path, "best_checkpoint.pth"))


def parse_arguments():
    arg = argparse.ArgumentParser()

    arg.add_argument("--resume", type=str, required=False, dest='model_name',
                     help="Resume training {model_name}.")
    arg.add_argument("--resume_intrinsic", type=str, required=False, dest='model_intrinsic',
                     help="Resume training {model_intrinsic}.")
    arg.add_argument("--gpu", type=int, default=0, required=False, help="Select the GPU card.")

    return vars(arg.parse_args())


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    args = parse_arguments()

    TrainTask(headless=True, model_name=args['model_name'], model_intrinsic=args['model_intrinsic'], gpu=args['gpu']).train()
