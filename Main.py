import argparse
import os
import pickle
import random
import shutil
import time
from csv import writer
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFont, ImageDraw
from prettytable import PrettyTable
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchvision import transforms
from tqdm import tqdm
import wandb
from Agent import Agent, Memory
from Environment import Environment
from Environment.CubeSimpleConf import CubeSimpleConf

torch.set_printoptions(threshold=10_000)

torch.set_printoptions(profile="full", precision=10, linewidth=100, sci_mode=False)


def percentage_error_formula(x, amount_variation): round(x / amount_variation * 100, 3)


def online_mean_and_sd(images):
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


def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument("--test", type=str, required=False, help="should train or test")
    arg.add_argument("--resume", type=str, required=False, help="should resume the train")

    args = vars(arg.parse_args())

    return args


def _format_video_wandb(last_obs_rollout) -> np.array:
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 256)),
        transforms.ToPILImage(),
    ])

    video = []

    font = ImageFont.truetype("Roboto/Roboto-Medium.ttf", size=10)

    for index, o in enumerate(last_obs_rollout):
        image_array = np.concatenate((trans(o[-1][1]), trans(o[-1][2])), axis=0)

        image = Image.fromarray(image_array, mode="RGB")

        image_editable = ImageDraw.Draw(image)

        image_editable.text((10, 115), o[-1][0], (0, 0, 0), align="center", font=font)

        video.append(np.asarray(image))

    video = np.asarray(video)
    video = np.transpose(video, (0, 3, 1, 2))

    return video


class Main:

    def __init__(self):

        # Training params
        self.conf = CubeSimpleConf()
        self.n_steps = 3E6
        self.n_rollout = 1
        self.n_trajectory = 3
        self.current_step = 0
        self.lr = 1e-5
        self.action_dim = 26
        self.action_std = 0.6
        self.betas = (0.9, 0.999)
        self.gamma = 0.99
        self.k_epochs = 60
        self.eps_clip = 0.2

        # Other params
        self.random_seed = 1
        self.best_mean_episodic_return = 0
        self.elapsed_time = 0
        self.num_parameters = 0
        self.output_path = None
        self.loss_path = None
        self.checkpoint_path = None
        self.images_path = None

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

            self.num_workers = 1
            self.pin_memory = True
            self.preload = True

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
            K_epochs=self.k_epochs,
            eps_clip=self.eps_clip,
            device=self.device
        )

        # Build the agent's memory
        self.memory = Memory()

        # Build the environment
        self.env = Environment(
            conf=self.conf,
            headless=True,
            random_seed=self.random_seed
        )

        # Count the number of the model parameters
        self._count_parameters()

        # Compute image mean and std
        mean, std = self._compute_mean_std()

        # Compose the transformations
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 64)),
            transforms.Normalize(mean, std)
        ])

    def train(self, resume=None):

        # Should resume the train
        if resume is None:

            # Set the folder time for each execution
            folder_time = time.strftime("%Y_%m_%d_%H_%M_%S")

            # Set the model name
            model_name = f"{self.conf}_{folder_time}"

            # Set the folders
            self.output_path = os.path.join('out', model_name)
            self.checkpoint_path = os.path.join(self.output_path, 'checkpoint')
            self.loss_path = os.path.join(self.output_path, 'loss')
            self.images_path = os.path.join(self.output_path, 'images')

            # Create the folders
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if not os.path.exists(self.loss_path):
                os.makedirs(self.loss_path)
            if not os.path.exists(self.images_path):
                os.makedirs(self.images_path)

        else:

            # Set the model to be loaded
            model_name = resume

            # Set the folders
            self.output_path = os.path.join('out', model_name)
            self.checkpoint_path = os.path.join(self.output_path, 'checkpoint')
            self.loss_path = os.path.join(self.output_path, 'loss')
            self.images_path = os.path.join(self.output_path, 'images')

            # Load the model
            self._load_checkpoint(best=False)

        # Init Wandb
        wandb.init(
            project=str(self.conf),
            name=model_name,
            id=model_name
        )

        print(f"[*] Output Folder: {model_name}")
        print(f"[*] Total Trainable Params: {self.num_parameters}")

        tic = time.time()
        with tqdm(total=self.n_steps) as pbar:

            while self.current_step < self.n_steps:
                # Train one rollout
                mean_episodic_return, loss, last_obs_rollout = self._train_one_rollout()

                self.current_step += self.n_trajectory * self.n_rollout

                # Measure elapsed time
                toc = time.time()

                # Set the var description
                pbar.set_description(("{:.1f}s - step: {:.1f} - loss: {:.6f} - return: {:.6f}".
                                      format((toc - tic),
                                             self.current_step,
                                             loss,
                                             mean_episodic_return)))

                # Update the bar
                pbar.update(self.current_step)

                # Log Wandb
                wandb.log(
                    {
                        "charts/mean_episodic_return": mean_episodic_return,
                        "charts/loss": loss,
                        "video": wandb.Video(_format_video_wandb(last_obs_rollout), fps=8)
                    }, step=self.current_step)

                # Check if it is the best model
                is_best = mean_episodic_return > self.best_mean_episodic_return

                self.best_mean_episodic_return = max(mean_episodic_return, self.best_mean_episodic_return)

                # Save the checkpoint for each rollout
                self._save_checkpoint({
                    "current_step": self.current_step,
                    "best_mean_episodic_return": self.best_mean_episodic_return,
                    "model_state": self.agent.policy.state_dict(),
                    "optim_state": self.agent.optimizer.state_dict(),
                }, is_best)

                # Dump the last observation data
                with open(os.path.join(self.images_path, f"last_observation.p"), "wb") as f:
                    pickle.dump(last_obs_rollout, f)

                row = [loss, mean_episodic_return]

                # Save training history
                with open(os.path.join(self.loss_path, 'history.csv'), 'a') as f_object:
                    writer_object = writer(f_object)
                    writer_object.writerow(row)
                    f_object.close()

        toc = time.time()

        self.elapsed_time = toc - tic

        # Save the configuration as image
        self._save_config()

    def _train_one_rollout(self) -> ():

        self.agent.policy.train()

        last_obs_rollout = []

        # For each rollout
        for r, _ in enumerate(range(self.n_rollout)):

            # Get the first observation
            old_observation = self.env.reset()

            for j in range(self.n_trajectory):

                # Save observations for the last rollout
                if r == self.n_rollout - 1:
                    last_obs_rollout.append(old_observation.copy())

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
                new_observation, reward = self.env.step(action.squeeze())

                # Save rollout to memory
                self.memory.rewards.append(reward)
                self.memory.states.append(state)
                self.memory.actions.append(action.squeeze())
                self.memory.logprobs.append(logprob.squeeze())
                self.memory.is_terminals.append(j == self.n_trajectory - 1)

                # Update observation
                old_observation = new_observation

        # Compute the mean episodic return
        mean_episodic_return = sum(self.memory.rewards) / len(self.memory.rewards)

        # Update the weights
        loss = self.agent.update(self.memory)

        # Clear the memory
        self.memory.clear_memory()

        return mean_episodic_return, loss.cpu().data.numpy(), last_obs_rollout

    def _build_vocab(self, instruction_set):

        def build_vocab(dataset: []):
            for instruction, _ in dataset:
                yield self.tokenizer(instruction)

        vocab = build_vocab_from_iterator(build_vocab(instruction_set), specials=["<UNK>"])
        vocab.set_default_index(vocab["<UNK>"])

        return vocab

    @torch.no_grad()
    def test(self, model_name):

        # Set the folders
        self.output_path = os.path.join('out', model_name)
        self.checkpoint_path = os.path.join(self.output_path, 'checkpoint')
        self.loss_path = os.path.join(self.output_path, 'loss')
        self.images_path = os.path.join(self.output_path, 'images')

        # Load the model
        self._load_checkpoint(best=False)

    def _visualize_observations(self, obs: [] = None):

        obs = self.env.reset()

        for index in range(128):
            action = [random.randrange(-3, 3) for i in range(26)]

            obs, _ = self.env.step(action)

            print(obs[-1][0])

            im = Image.fromarray(obs[-1][1], mode="RGB")
            im.save(f"/home/ic-unicamp/phd/out/{index}.png")

    def _count_parameters(self, print_table=False):

        table = PrettyTable(["Modules", "Parameters"])

        for name, parameter in self.agent.policy.named_parameters():

            if parameter.requires_grad:
                param = parameter.numel()
                table.add_row([name, param])

                self.num_parameters += param

        if print_table:
            print(table)

    def _save_checkpoint(self, state, is_best):

        # Set the checkpoint path
        ckpt_path = os.path.join(self.checkpoint_path, "checkpoint.tar")

        # Save the checkpoint
        torch.save(state, ckpt_path)

        # Save the best model
        if is_best:
            # Copy the checkpoint to the best model
            shutil.copyfile(ckpt_path, os.path.join(self.checkpoint_path, "best_model.tar"))

    def _load_checkpoint(self, best=False):

        print(f"[*] Loading model from {self.checkpoint_path}")

        # Define which model to load
        filename = "best_model.tar" if best else "checkpoint.tar"

        # Set the checkpoint path
        checkpoint_path = os.path.join(self.checkpoint_path, filename)

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)

        # Load the variables from checkpoint
        self.current_step = checkpoint["current_step"]
        self.best_mean_episodic_return = checkpoint["best_mean_episodic_return"]
        self.agent.policy.load_state_dict(checkpoint["model_state"])
        self.agent.optimizer.load_state_dict(checkpoint["optim_state"])

        if best:
            print(
                f"[*] Loaded checkpoint @ step {self.current_step} with best mean episodic return of "
                f"{self.best_mean_episodic_return}")
        else:
            print(f"[*] Loaded Best Model @ step {self.current_step}")

    def _save_config(self):

        df = pd.DataFrame()
        df['n_steps'] = [self.n_steps]
        df['n_rollout'] = [self.n_rollout]
        df['n_trajectory'] = [self.n_trajectory]
        df['# Params'] = [self.num_parameters]
        df['lr'] = [self.lr]
        df['random_seed'] = [self.random_seed]
        df['elapsed_time'] = [time.strftime("%H:%M:%S", time.gmtime(self.elapsed_time))]
        df = df.astype(str)

        # Save
        df.to_csv(f"{self.output_path}/config.csv")

    def _compute_mean_std(self, n_observations_computation=5):

        obs = self.env.reset()

        images = []

        # Compose the transformations
        trans = transforms.Compose([
            transforms.ToTensor(),
        ])

        image_tensor = torch.empty((len(obs) * n_observations_computation, 3, 512, 2048), dtype=torch.float)

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

        return online_mean_and_sd(image_tensor)


if __name__ == "__main__":

    args = parse_arguments()

    main = Main()

    if args['test'] is not None:
        main.test(args['test'])
    else:
        main.train(args['resume'])
        # main._visualize_observations()
