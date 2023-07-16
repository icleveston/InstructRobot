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
from torchvision import transforms
from tqdm import tqdm
import wandb
from Agent import Agent, Memory
from Environment import Environment
from multiprocessing import Process, JoinableQueue
import multiprocessing
from Environment.CubeSimpleConf import CubeSimpleConf

torch.set_printoptions(threshold=10_000)

torch.set_printoptions(profile="full", precision=10, linewidth=100, sci_mode=False)


def percentage_error_formula(x, amount_variation): round(x / amount_variation * 100, 3)


def _create_env(queues: (), n_steps, n_rollout, n_trajectory, conf, trans_mean_std, random_seed):
    in_queue, out_queue = queues

    # Create the environment
    env = Environment(
        conf=conf,
        trans_mean_std=trans_mean_std,
        random_seed=random_seed
    )

    current_step = 0

    while current_step < n_steps:

        for _ in range(n_rollout):

            current_step += n_trajectory * n_rollout

            # Reset environment
            data = env.reset()

            # Return first observation
            out_queue.put(data)
            out_queue.join()

            for _ in range(n_trajectory):
                # Wait for action from main process
                action = in_queue.get()
                in_queue.task_done()

                # Execute step
                data = env.step(action)

                # Return obs and reward to main process
                out_queue.put(data)
                out_queue.join()


def _save_wandb(in_queue, conf, model_name, images_path, loss_path, checkpoint_path):
    # Init Wandb
    wandb.init(
        project=str(conf),
        name=model_name,
        id=model_name
    )

    while True:
        data = in_queue.get()
        in_queue.task_done()

        # Unpack data
        mean_episodic_return, loss, lr, eps, action_std, obs_rollout, current_step, agent_state, optim_state, \
            best_mean_episodic_return = data

        # Unpack observations
        actions = []
        observations = []

        for r in obs_rollout:
            actions.append(r[0].cpu().data.numpy())
            observations.append(r[1])

        # Log Wandb
        wandb.log(
            {
                "mean_episodic_return": mean_episodic_return,
                "loss": loss,
                "lr": np.float32(lr),
                "action_std": np.float32(action_std),
                "eps": np.float32(eps),
                "video": wandb.Video(_format_video_wandb(observations), fps=8),
                f"actions-{current_step}": wandb.Table(columns=[f"a{i}" for i in range(26)], data=actions)
            }, step=current_step)

        # Dump observation data
        with open(os.path.join(images_path, f"observation.p"), "wb") as f:
            pickle.dump(observations, f)

        row = [loss, mean_episodic_return]

        # Save training history
        with open(os.path.join(loss_path, 'history.csv'), 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(row)
            f_object.close()

        is_best = mean_episodic_return > best_mean_episodic_return

        # Save the checkpoint for each rollout
        _save_checkpoint({
            "current_step": current_step,
            "best_mean_episodic_return": best_mean_episodic_return,
            "model_state": agent_state,
            "optim_state": optim_state,
        }, checkpoint_path, is_best)


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
        image_array = np.concatenate((trans(o[-1][2]), trans(o[-1][3])), axis=0)

        image = Image.fromarray(image_array, mode="RGB")

        image_editable = ImageDraw.Draw(image)

        image_editable.text((10, 115), o[-1][0], (0, 0, 0), align="center", font=font)

        video.append(np.asarray(image))

    video = np.asarray(video)
    video = np.transpose(video, (0, 3, 1, 2))

    return video


def _save_checkpoint(state, checkpoint_path, is_best):
    # Set the checkpoint path
    ckpt_path = os.path.join(checkpoint_path, "checkpoint.tar")

    # Save the checkpoint
    torch.save(state, ckpt_path)

    # Save the best model
    if is_best:
        # Copy the checkpoint to the best model
        shutil.copyfile(ckpt_path, os.path.join(checkpoint_path, "best_model.tar"))


class Main:

    def __init__(self):

        # Training params
        self.conf = CubeSimpleConf()
        self.n_steps = 3E6
        self.n_rollout = 16
        self.n_trajectory = 32
        self.current_step = 0
        self.lr = 3e-4
        self.action_dim = 26
        self.action_std = 0.6
        self.betas = (0.9, 0.999)
        self.gamma = 0.99
        self.k_epochs = 15
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
            k_epochs=self.k_epochs,
            total_iters=self.n_steps // (self.n_rollout * self.n_trajectory),
            eps_clip=self.eps_clip,
            device=self.device
        )

        # Build the agent's memory
        self.memory = Memory()

        # Count the number of the model parameters
        self._count_parameters()

        # Create queues
        self.in_queues = [JoinableQueue() for _ in range(self.n_rollout)]
        self.out_queues = [JoinableQueue() for _ in range(self.n_rollout)]

        # Compute image mean and std
        # self._compute_mean_std()
        trans_mean_std = ([0.8517414331, 0.8405256271, 0.8349498510], [0.1922473758, 0.2080573738, 0.2201343030])

        # Create processes
        self.processes = [Process(target=_create_env, args=(q,
                                                            self.n_steps,
                                                            self.n_rollout,
                                                            self.n_trajectory,
                                                            self.conf,
                                                            trans_mean_std,
                                                            self.random_seed + i)) for i, q in
                          enumerate(zip(self.in_queues,
                                        self.out_queues))]

        # Start processes
        [p.start() for p in self.processes]

        # Wandb queue and process
        self.in_queues_wandb = JoinableQueue()
        self.process_wandb = None

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

        # Start wandb process
        self.process_wandb = Process(target=_save_wandb, args=(self.in_queues_wandb,
                                                               self.conf,
                                                               model_name,
                                                               self.images_path,
                                                               self.loss_path,
                                                               self.checkpoint_path))
        self.process_wandb.start()

        print(f"[*] Output Folder: {model_name}")
        print(f"[*] Total Trainable Params: {self.num_parameters}")

        tic = time.time()
        with tqdm(total=self.n_steps) as pbar:

            while self.current_step < self.n_steps:
                # Train one rollout
                mean_episodic_return, loss, obs_rollout = self._train_one_rollout()

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

                # Check if it is the best model
                self.best_mean_episodic_return = max(mean_episodic_return, self.best_mean_episodic_return)

                # Get states
                agent_state = self.agent.policy.state_dict()
                optim_state = self.agent.optimizer.state_dict()

                # Send data to wandb process
                self.in_queues_wandb.put((mean_episodic_return,
                                          loss,
                                          self.agent.scheduler.get_last_lr()[0],
                                          self.agent.eps_clip,
                                          self.agent.policy.action_std,
                                          obs_rollout,
                                          self.current_step,
                                          agent_state,
                                          optim_state,
                                          self.best_mean_episodic_return
                                          ))
                self.in_queues_wandb.join()

        # Wait all process to finish
        [p.join() for p in self.processes]
        self.process_wandb.join()

        toc = time.time()

        self.elapsed_time = toc - tic

        # Save the configuration as image
        self._save_config()

    def _train_one_rollout(self) -> ():

        self.agent.policy.train()

        obs_rollout = []

        # Random a rollout to sample
        random_sample_index = random.randint(0, self.n_rollout - 1)

        states_array = [[] for _ in range(self.n_rollout)]
        reward_array = [[] for _ in range(self.n_rollout)]
        action_array = [[] for _ in range(self.n_rollout)]
        logprob_array = [[] for _ in range(self.n_rollout)]
        is_terminals_array = [[] for _ in range(self.n_rollout)]
        training_data_array = [[] for _ in range(self.n_rollout)]

        # Get first observations
        for r, o in enumerate(self.out_queues):
            state, training_data = o.get()
            o.task_done()

            states_array[r].append(state)
            training_data_array[r].append((torch.tensor([0 for _ in range(self.action_dim)]), training_data))

        # For each trajectory
        for j, t in enumerate(range(self.n_trajectory)):

            # Save observations for the last rollout
            obs_rollout.append(training_data_array[random_sample_index][j])

            # Select action from the agent
            actions, logprobs = self.agent.select_action([s[-1] for s in states_array])

            # Send actions to environments
            for in_index, in_q in enumerate(self.in_queues):
                in_q.put(actions[in_index].cpu().data.numpy().tolist())
                in_q.join()

            # Get states from environments
            for r, out_q in enumerate(self.out_queues):
                new_state, reward, training_data = out_q.get()
                out_q.task_done()

                reward_array[r].append(reward)
                action_array[r].append(actions[r])
                logprob_array[r].append(logprobs[r])
                is_terminals_array[r].append(0 if j != self.n_trajectory - 1 else 1)
                if j != self.n_trajectory-1:
                    states_array[r].append(new_state)

                training_data_array[r].append((actions[r], training_data))

        # Save rollout to memory
        for r in range(self.n_rollout):
            self.memory.rewards += reward_array[r]
            self.memory.states += states_array[r]
            self.memory.actions += action_array[r]
            self.memory.logprobs += logprob_array[r]
            self.memory.is_terminals += is_terminals_array[r]

        # Compute the mean episodic return
        mean_episodic_return = sum(self.memory.rewards) / len(self.memory.rewards)

        # Update the weights
        loss = self.agent.update(self.memory)

        # Clear the memory
        self.memory.clear_memory()

        return mean_episodic_return, loss.cpu().data.numpy(), obs_rollout

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

            im = Image.fromarray(obs[-1][2], mode="RGB")
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

        # Create the environment
        env = Environment(
            conf=self.conf,
            random_seed=self.random_seed
        )

        _, obs = env.reset()

        # Compose the transformations
        trans = transforms.Compose([
            transforms.ToTensor(),
        ])

        image_tensor = torch.empty((len(obs) * n_observations_computation, 3, 128, 512), dtype=torch.float)

        index = 0

        for _ in range(n_observations_computation):

            action = [random.randrange(-1, 1) for _ in range(26)]

            _, _, obs = env.step(action)

            for o in obs:
                image_top = o[2]
                image_front = o[3]

                # Convert state to tensor
                image_top_tensor = trans(image_top)
                image_font_tensor = trans(image_front)

                # Cat all images into a single one
                images_stacked = torch.cat((image_top_tensor, image_font_tensor), dim=2)

                image_tensor[index] = images_stacked

                index += 1

        env.close()

        print(online_mean_and_sd(image_tensor))


if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')

    args = parse_arguments()

    main = Main()

    if args['test'] is not None:
        main.test(args['test'])
    else:
        main.train(args['resume'])
        # main._visualize_observations()
