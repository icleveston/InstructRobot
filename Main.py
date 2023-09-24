import random
import time
import pandas as pd
from prettytable import PrettyTable
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm
from Agent import Agent, Memory
from Environment import Environment
from Environment.CubeSimpleConf import CubeSimpleConf
from multiprocessing import Process, JoinableQueue
import multiprocessing
from Utils import *

torch.set_printoptions(threshold=10_000)

torch.set_printoptions(profile="full", precision=10, linewidth=100, sci_mode=False)


class Main:

    def __init__(self, headless=True):

        # Training params
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
            self.info_path = os.path.join(self.output_path, 'info')
            self.images_path = os.path.join(self.output_path, 'images')

            # Create the folders
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if not os.path.exists(self.info_path):
                os.makedirs(self.info_path)
            if not os.path.exists(self.images_path):
                os.makedirs(self.images_path)

        else:

            # Set the model to be loaded
            model_name = resume

            # Set the folders
            self.output_path = os.path.join('out', model_name)
            self.checkpoint_path = os.path.join(self.output_path, 'checkpoint')
            self.info_path = os.path.join(self.output_path, 'info')
            self.images_path = os.path.join(self.output_path, 'images')

            # Load the model
            self._load_checkpoint(best=False)

        # Start wandb process
        self.process_wandb = Process(target=save_wandb, args=(self.in_queues_wandb,
                                                               resume is not None,
                                                               self.conf,
                                                               model_name,
                                                               self.images_path,
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

                # Check if it is the best model
                self.best_mean_episodic_return = max(return_info["mean"], self.best_mean_episodic_return)

                # Get states
                agent_state = self.agent.policy.state_dict()
                optim_state = self.agent.optimizer.state_dict()
                scheduler_state = self.agent.scheduler.state_dict()

                # Send data to wandb process
                self.in_queues_wandb.put((return_info,
                                          loss_info,
                                          self.agent.scheduler.get_last_lr()[0],
                                          self.agent.eps_clip,
                                          self.agent.policy.action_std,
                                          obs_rollout,
                                          self.current_step,
                                          agent_state,
                                          optim_state,
                                          scheduler_state,
                                          self.best_mean_episodic_return
                                          ))

                self.in_queues_wandb.join()

        # Kill all process
        [p.kill() for p in self.processes]
        self.process_wandb.kill()

        toc = time.time()

        self.elapsed_time = toc - tic

        # Save the configuration as image
        self._save_config()

    def _train_one_rollout(self) -> ():

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

        # Compute the mean and std episodic return
        return_info = {
            "mean": np.mean(np.array(self.memory.rewards)),
            "std": np.std(np.mean(np.reshape(np.array(self.memory.rewards)), (self.n_rollout, -1)), axis=1)
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
        self.info_path = os.path.join(self.output_path, 'info')
        self.images_path = os.path.join(self.output_path, 'images')

        # Load the model
        self._load_checkpoint(best=False)

    def validate_observations(self):

        self.env.reset()

        for index in range(128):
            action = [random.randrange(-3, 3) for i in range(26)]

            obs, _ = self.env.step(action)

            print(obs[-1][0])

            im = Image.fromarray(obs[-1][1], mode="RGB")
            im.save(f"./out/{index}.png")

    def validate_joints_nao(self):
        self.env.validate_joints_nao()

    def validate_collisions_nao(self):
        self.env.validate_collisions_nao()

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


def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument("--test", type=str, required=False, help="Test trained agent {name}.")
    arg.add_argument("--resume", type=str, required=False, help="Resume training {name}.")
    arg.add_argument("--val-obs", type=bool, default=False, required=False, help="Validate observations.")
    arg.add_argument("--val-joint-nao", type=bool, default=False, required=False, help="Validate NAO's joints.")
    arg.add_argument("--val-collisions-nao", type=bool, default=False, required=False, help="Validate NAO's collisions.")

    return vars(arg.parse_args())


if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')

    args = parse_arguments()

    if args['test'] is not None:
        Main().test(args['test'])
    elif args['val_obs']:
        Main().validate_observations()
    elif args['val_joint_nao']:
        Main(headless=False).validate_joints_nao()
    elif args['val_collisions_nao']:
        Main(headless=False).validate_collisions_nao()
    else:
        Main(headless=True).train(args['resume'])

