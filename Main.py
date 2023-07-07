import argparse
import os
import pickle
import random
import shutil
import time

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
from Environment.CubeSimpleSet import CubeSimpleSet

torch.set_printoptions(threshold=10_000)


torch.set_printoptions(profile="full", precision=10, linewidth=100, sci_mode=False)


def percentage_error_formula(x, amount_variation): round(x / amount_variation * 100, 3)


class Main:

    def __init__(self):

        # Training params
        self.scene_file = 'Scenes/Cubes_Simple.ttt'
        self.instruction_set = CubeSimpleSet()
        self.n_steps = 3E6
        self.n_rollout = 32
        self.n_trajectory = 128
        self.current_step = 0
        self.lr = 1e-5
        self.action_dim = 26
        self.action_std = 0.6
        self.betas = (0.9, 0.999)
        self.gamma = 0.99
        self.k_epochs = 20
        self.eps_clip = 0.2

        # Other params
        self.random_seed = 1
        self.best_mean_episodic_return = 0
        self.elapsed_time = 0
        self.num_parameters = 0
        self.loss_path = None
        self.checkpoint_path = None
        self.output_path = None

        # Create tokenizer and vocab
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = self._build_vocab(self.instruction_set)

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
            scene_file=self.scene_file,
            headless=True,
            instruction_set=self.instruction_set,
            random_seed=self.random_seed
        )

        # Count the number of the model parameters
        self._count_parameters()

        # Compute image mean and std
        self.mean, self.std = self._compute_mean_std()

    def train(self, resume=None):

        # Should resume the train
        if resume is None:

            # Set the folder time for each execution
            folder_time = time.strftime("%Y_%m_%d_%H_%M_%S")

            # Set the model name
            model_name = f"exec_{self.instruction_set}_{folder_time}"

            # Set the folders
            self.output_path = os.path.join('out', model_name)
            self.checkpoint_path = os.path.join(self.output_path, 'checkpoint')
            self.loss_path = os.path.join(self.output_path, 'loss')

            # Create the folders
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if not os.path.exists(self.loss_path):
                os.makedirs(self.loss_path)

        else:

            # Set the model to be loaded
            model_name = resume

            # Set the folders
            self.output_path = os.path.join('out', model_name)
            self.checkpoint_path = os.path.join(self.output_path, 'checkpoint')
            self.loss_path = os.path.join(self.output_path, 'loss')

            # Load the model
            self._load_checkpoint(best=False)

        # Init Wandb
        wandb.init(
            project=str(self.instruction_set),
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
                        "charts/loss":  loss,
                        "video": wandb.Video(self._format_video_wandb(last_obs_rollout), fps=8)
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

                # Dump the training data
                with open(os.path.join(self.loss_path, f"loss_step_{self.current_step}.p"), "wb") as f:
                    pickle.dump((mean_episodic_return, loss, last_obs_rollout), f)

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
                if r == self.n_rollout-1:
                    last_obs_rollout.append(old_observation.copy())

                # Tokenize instruction
                instruction_token = self.tokenizer(old_observation[-1][0])

                # Get instructions indexes
                instruction_index = torch.tensor(self.vocab(instruction_token), device=self.device)

                # Compose the transformations
                trans = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((256, 128)),
                    transforms.Normalize(self.mean, self.std)
                ])

                image_tensor = torch.empty((len(old_observation), 3, 256, 256), dtype=torch.float, device=self.device)

                for i, o in enumerate(old_observation):
                    image_top = o[1]
                    image_front = o[2]

                    # Convert state to tensor
                    image_top_tensor = trans(image_top)
                    image_font_tensor = trans(image_front)

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
                self.memory.is_terminals.append(j == self.n_trajectory-1)

                # Update observation
                old_observation = new_observation

        # Compute the mean episodic return
        mean_episodic_return = sum(self.memory.rewards)/len(self.memory.rewards)

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
        pass

    #
    #     # Set the model to load
    #     self.model_name = model_name
    #
    #     # Set the folders
    #     self.output_path = os.path.join('out', self.model_name, 'results', str(test_seq))
    #     self.checkpoint_path = os.path.join('out', self.model_name, 'checkpoint')
    #
    #     if not os.path.exists(self.output_path):
    #         os.makedirs(self.output_path)
    #
    #     # Load the model
    #     self._load_checkpoint(best=False)
    #
    #     # Print the model info
    #     print(f"[*] Total Trainable Params: {self.num_parameters}")
    #
    #     mse_all = []
    #     samples = []
    #
    #     predictions_array = torch.tensor([]).to(self.device)
    #     y_array = torch.tensor([]).to(self.device)
    #     l_t_array_all = torch.tensor([]).to(self.device)
    #
    #     print(f"[*] Test on {self.num_test} samples")
    #
    #     loss_mse = torch.nn.MSELoss()
    #
    #     starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    #     repetitions = len(self.test_loader)
    #     timings = np.zeros((repetitions, 1))
    #
    #     for i, (x, y) in enumerate(self.test_loader):
    #
    #         glimpse_location = []
    #
    #         # Set data to the respected device
    #         x, y = x.to(self.device), y.to(self.device)
    #
    #         # Generate the context for the first image
    #         h_t_0 = torch.zeros(self.batch_size, 1024).to(self.device)
    #         h_t_1 = torch.zeros(self.batch_size, 1024).to(self.device)
    #
    #         # Initialize the latent space for each new mini batch
    #         self.model.core.hidden_cell = (
    #             torch.stack((h_t_0, h_t_1)), torch.zeros(2, self.batch_size, 1024).to(self.device))
    #
    #         predicted = None
    #
    #         starter.record()
    #
    #         for t in range(self.num_glimpses):
    #             # Running policy_old:
    #             l_t = self.ppo.select_action(h_t_1.detach(), self.memory)
    #
    #             # Store the glimpse location for both frames
    #             glimpse_location.append(l_t)
    #
    #             # Call the model and pass the minibatch
    #             h_t_0, h_t_1, predicted = self.model(x, l_t)
    #
    #         ender.record()
    #         torch.cuda.synchronize()
    #         curr_time = starter.elapsed_time(ender)
    #         timings[i] = curr_time
    #
    #         predictions_array = torch.cat((predictions_array, predicted))
    #         y_array = torch.cat((y_array, y))
    #         l_t_array_all = torch.cat((l_t_array_all, torch.stack(glimpse_location)), axis=1)
    #
    #         self.memory.clear_memory()
    #
    #         # For the first minibatch
    #         if i == 0:
    #             trans = transforms.Compose([
    #                 NormalizeInverse([0.4209265411], [0.2889825404]),
    #                 transforms.ToPILImage()
    #             ])
    #
    #             # Build the glimpses array
    #             glimpses = [trans(x[0, 0].cpu()), trans(x[0, 1].cpu()),
    #                         torch.stack(glimpse_location)[:, 0].cpu().data.numpy()]
    #
    #             # Dump the glimpses
    #             with open(os.path.join(self.output_path, f"glimpses_epoch_test.p"), "wb") as f:
    #                 pickle.dump(glimpses, f)
    #
    #     mean_syn = np.sum(timings) / repetitions
    #     std_syn = np.std(timings)
    #     print(mean_syn)
    #
    #     # Dump the glimpses for heatmap
    #     with open(os.path.join(self.output_path, f"glimpses_heatmap.p"), "wb") as f:
    #         pickle.dump(l_t_array_all, f)
    #
    #     # Get samples every 20 frames
    #     skip = len(y_array) // 20
    #
    #     # Save the first prediction
    #     samples = [[h.cpu().numpy(), p.cpu().numpy()] for h, p in zip(y_array[::skip], predictions_array[::skip])]
    #
    #     # Compute the metrics
    #     y_rot = y_array[:, :3]
    #     y_tran = y_array[:, 3:]
    #     pred_rot = predictions_array[:, :3]
    #     pred_tran = predictions_array[:, 3:]
    #
    #     # Compute losses for differentiable modules
    #     rot_loss = loss_mse(pred_rot, y_rot)
    #     trans_loss = loss_mse(pred_tran, y_tran)
    #
    #     regressor_loss = rot_loss + trans_loss
    #
    #     # Save the results as image
    #     self._save_results(regressor_loss.item(), rot_loss.item(), trans_loss.item(), samples, glimpses)
    #
    #     mean = torch.tensor(
    #         [-7.6397992e-05, 2.6872402e-04, 4.7161593e-06, -9.7197731e-04, -1.7675826e-02, 9.2309231e-01]).to(
    #         self.device)
    #     std = torch.tensor([0.00305257, 0.01770405, 0.00267268, 0.02503707, 0.01716818, 0.30884704]).to(self.device)
    #
    #     # Denormalize gt
    #     std_inv = 1 / (std + 1e-8)
    #     mean_inv = -mean * std_inv
    #
    #     y_array = (y_array - mean_inv) / std_inv
    #     predictions_array = (predictions_array - mean_inv) / std_inv
    #
    #     predictions_array = predictions_array.cpu().data.numpy()
    #     y_array = y_array.cpu().data.numpy()
    #
    #     # Generate the trajectory and metrics
    #     self._save_evaluation(predictions_array, dataset)

    def _visualize_observations(self, obs: [] = None):

        obs = self.env.reset()

        # transinv = transforms.Compose([
        #     NormalizeInverse([0.4561], [0.3082]),
        #     transforms.ToPILImage()
        # ])

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

    def _format_video_wandb(self, last_obs_rollout) -> np.array:

        video = []

        font = ImageFont.truetype("Roboto/Roboto-Medium.ttf", size=25)

        for index, o in enumerate(last_obs_rollout):

            image_array = np.concatenate((o[-1][1], o[-1][2]), axis=0)

            image = Image.fromarray(image_array, mode="RGB")

            image_editable = ImageDraw.Draw(image)

            image_editable.text((15, 480), o[-1][0], (0, 0, 0), align="center", font=font)

            video.append(np.asarray(image))

        video = np.asarray(video)
        video = np.transpose(video, (0, 3, 1, 2))

        return video

    def _compute_mean_std(self, n_observations_computation=5):

        obs = self.env.reset()

        images = []

        # Compose the transformations
        trans = transforms.Compose([
            transforms.ToTensor(),
        ])

        image_tensor = torch.empty((len(obs)*n_observations_computation, 3, 512, 2048), dtype=torch.float)

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


if __name__ == "__main__":

    args = parse_arguments()

    main = Main()

    if args['test'] is not None:
        main.test(args['test'])
    else:
        main.train(args['resume'])
        #main._visualize_observations()

