import os
import shutil
import torch
import wandb
import numpy as np
import random
import time
from prettytable import PrettyTable
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from multiprocessing import JoinableQueue
from torchvision import transforms
from PIL import Image, ImageFont, ImageDraw
from csv import writer
from multiprocessing import Process

from .Agent import Agent, Memory
from .Environment import Environment
from .Environment.CubeSimpleConf import CubeSimpleConf

torch.set_printoptions(threshold=10_000)

torch.set_printoptions(profile="full", precision=10, linewidth=100, sci_mode=False)


class Main:

    def __init__(self, headless: bool = True, model_name: str = None):

        # Training params
        self.model_name = model_name
        self.conf = CubeSimpleConf()
        self.n_steps = 3E6
        self.n_rollout = 24
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
        self.env_mean = None
        self.env_std = None
        self.trans = None

        # Create tokenizer and vocab
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = self._build_vocab(self.conf.instruction_set)

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

        # Build the agent
        self.agent = Agent(
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

        # Build the environment
        self.env = Environment(
            conf=self.conf,
            headless=headless,
            random_seed=self.random_seed
        )

    def start_train(self) -> None:

        is_new_execution: bool = False

        # If model_name is none, create a new one
        if self.model_name is None:
            folder_time = time.strftime("%Y_%m_%d_%H_%M_%S")

            self.model_name = f"{self.conf}_{folder_time}"

            is_new_execution = True

        # Set the folders
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

        # If it is not a new execution, load the model
        if not is_new_execution:
            self._load_checkpoint(load_best_checkpoint=False)

        # Compute image mean and std
        self.env_mean, self.env_std = self._compute_env_mean_std()

        # Compose the transformations
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 64)),
            transforms.Normalize(self.env_mean, self.env_std)
        ])

        # Wandb queue and process
        self.in_queues_wandb = JoinableQueue()
        self.process_wandb = None

        # Start wandb process
        self.process_wandb = Process(target=_save_wandb, args=(self.in_queues_wandb,
                                                               is_new_execution,
                                                               self.conf,
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
            'env_mean': self.env_mean.tolist(),
            'env_std': self.env_std.tolist(),
            "num_parameters": self.num_parameters,
            "random_seed": self.random_seed
        }

        # Save configuration params
        _save_csv(os.path.join(self.info_path, 'config.csv'), config_info)

        print(f"[*] Output Folder: {self.model_name}")
        print(f"[*] Total Trainable Params: {self.num_parameters}")

        # Start timer
        self.tic = time.time()

    def process_rollout(self, loss_info: dict, observations) -> str:

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

    def _build_vocab(self, instruction_set):

        def build_vocab(dataset: []):
            for instruction, _ in dataset:
                yield self.tokenizer(instruction)

        vocab = build_vocab_from_iterator(build_vocab(instruction_set), specials=["<UNK>"])
        vocab.set_default_index(vocab["<UNK>"])

        return vocab

    def _count_parameters(self, print_table=False):

        table = PrettyTable(["Modules", "Parameters"])

        for name, parameter in self.agent.policy.named_parameters():

            if parameter.requires_grad:
                param = parameter.numel()
                table.add_row([name, param])

                self.num_parameters += param

        if print_table:
            print(table)

    def _load_checkpoint(self, load_best_checkpoint=False):

        print(f"[*] Loading checkpoint from {self.checkpoint_path}")

        # Define which checkpoint to load
        filename = "best_checkpoint.tar" if load_best_checkpoint else "last_checkpoint.tar"

        # Set the checkpoint path
        checkpoint_path = os.path.join(self.checkpoint_path, filename)

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)

        # Load the variables from checkpoint
        self.current_step = checkpoint["step"]
        self.best_mean_episodic_return = checkpoint["best_mean_episodic_return"]
        self.agent.policy.load_state_dict(checkpoint["agent_state"])
        self.agent.optimizer.load_state_dict(checkpoint["optim_state"])
        self.agent.scheduler.load_state_dict(checkpoint["scheduler_state"])

        if load_best_checkpoint:
            print(f"[*] Loaded best checkpoint @ step {self.current_step}")
        else:
            print(f"[*] Loaded last checkpoint @ step {self.current_step}")

    def _compute_env_mean_std(self, n_observations_computation=5):

        obs = self.env.reset()

        # Compose the transformations
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 64))
        ])

        image_tensor = torch.empty((len(obs) * n_observations_computation, 3, 256, 256), dtype=torch.float)

        index = 0

        for _ in range(n_observations_computation):

            action = [random.randrange(-1, 1) for _ in range(26)]

            obs, _ = self.env.step(action)

            for o in obs:
                image_top = o[1]
                image_front = o[2]

                # Convert state to tensor
                image_top_tensor = trans(image_top)
                image_font_tensor = trans(image_front)

                # Cat all images into a single one
                images_stacked = torch.cat((image_top_tensor, image_font_tensor), dim=2)

                image_tensor[index] = images_stacked

                index += 1

        return _online_mean_and_sd(image_tensor)


def percentage_error_formula(x: float, amount_variation: float) -> float:
    return round(x / amount_variation * 100, 3)


def _save_wandb(in_queue,
                is_new_execution: bool,
                conf: str,
                model_name: str,
                info_path: str,
                checkpoint_path: str) -> None:
    # Init Wandb
    wandb.init(
        project=str(conf),
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
        loss_description = {f"loss/{loss_name}": loss_value for loss_name, loss_value in loss_info.items()}

        # Create the wandb log
        wandb_log = {
            "mean_episodic_return": mean_std_info["mean"],
            "lr": np.float32(params_info["lr"]),
            "action_std": np.float32(params_info["action_std"]),
            "eps": np.float32(params_info["eps_clip"]),
            "video": wandb.Video(_format_video_wandb(observation), fps=8),
            # f"actions-{current_step}": wandb.Table(columns=[f"a{i}" for i in range(26)], data=actions)
        }

        # Update log with loss description
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
        transforms.Resize((128, 256)),
        transforms.ToPILImage(),
    ])

    video = []

    font = ImageFont.truetype("Main/Roboto/Roboto-Medium.ttf", size=10)

    for index, o in enumerate(last_obs_rollout):
        image_array = np.concatenate((trans(o[-1][1]), trans(o[-1][2])), axis=0)

        image = Image.fromarray(image_array, mode="RGB")

        image_editable = ImageDraw.Draw(image)

        image_editable.text((10, 115), o[-1][0], (0, 0, 0), align="center", font=font)

        video.append(np.asarray(image))

    video = np.asarray(video)
    video = np.transpose(video, (0, 3, 1, 2))

    return video


def _save_checkpoint(checkpoint_path: str, checkpoint: dict, is_best_checkpoint: bool) -> None:
    # Set the checkpoint path
    ckpt_path = os.path.join(checkpoint_path, "last_checkpoint.tar")

    # Save the checkpoint
    torch.save(checkpoint, ckpt_path)

    # Save the best checkpoint
    if is_best_checkpoint:
        # Copy the last checkpoint to the best checkpoint
        shutil.copyfile(ckpt_path, os.path.join(checkpoint_path, "best_checkpoint.tar"))


def _online_mean_and_sd(images: np.array) -> tuple:
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    b, c, h, w = images.shape
    nb_pixels = b * h * w
    sum_ = torch.sum(images, dim=[0, 2, 3])
    sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
    fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
    snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

    cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
