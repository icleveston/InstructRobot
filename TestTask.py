import os
import torch
import argparse
import time
import wandb
import random
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from csv import writer
from multiprocessing import Process, JoinableQueue
from torchvision import transforms
from Main.Agent.Extrinsic_task.ActorCritic import Actor
from Main.Environment.CubeChangeTableExtEnvTest import CubeChangeTableExtEnvTest



class TestTask():

    def __init__(self, headless: bool = False, model_name: str = None, device: str = 'cuda:0'):

        self.device = device
        self.n_tests = 17
        self.n_trajectory = 32
        self.n_exec_test = 10
        self.random_seed = 1
        self.env = CubeChangeTableExtEnvTest(
            headless=headless,
            num_styles=self.n_tests,
            random_seed=self.random_seed
        )

        self.actor = Actor(action_dim=26).to(device)

        self.current_step = None
        self.best_mean_episodic_return = None

        self.model_name = model_name
        self.output_path = None
        self.checkpoint_path = None
        self.trans_rgb = None
        self.env_mean = [0.5868541598320007, 0.4782417416572571, 0.46996355056762695]
        self.env_std = [0.3157169818878174, 0.3070945739746094, 0.3131164312362671]


    @torch.no_grad()
    def test(self) -> None:
        self._start_test()
        cont = 0
        out = dict()
        table_wb = wandb.Table(columns=['env', 'score_mean'])
        for r in range(self.n_tests):
            score_test = 0
            # Get the first observation
            for _ in range(self.n_exec_test):
                observations = []
                old_observation = self.env.reset(r)

                for j in range(self.n_trajectory):
                    # Save observations
                    observations.append(old_observation.copy())

                    # Build state from observation
                    state_flatten = self._build_state_from_observations(old_observation)
                    state_vision = state_flatten[0]
                    state_proprioception = state_flatten[1]

                    state_vision = state_vision.unsqueeze(dim=0)
                    state_proprioception = state_proprioception.unsqueeze(dim=0)

                    # Select action from the agent
                    action = self.actor(state_vision, state_proprioception)

                    # Execute action in the simulator
                    new_observation, ext_reward = self.env.step(action.squeeze().data.cpu().numpy())
                    score_test += ext_reward
                    # Update observation
                    old_observation = new_observation

                wandb.log({
                    "video_test": wandb.Video(_format_video_wandb(observations), fps=8),
                    #f"score_mean-{current_step}": wandb.Table(columns=[f"a{i}" for i in range(26)], data=actions)
                }, step=cont)
                cont += 1

            score_mean = score_test / self.n_exec_test
            table_wb.add_data(r, score_mean)
            print(f'Score test: {r}  -- mean: {score_mean}')
            out[r] = score_mean

        _save_csv(os.path.join(self.output_path, 'results.csv'), out)
        wandb.log({f"score_table": table_wb})


    def _start_test(self) -> None:
        is_new_execution = True
        # Set the folders
        self.output_path = os.path.join('out', self.model_name)
        self.checkpoint_path = os.path.join(self.output_path, 'checkpoint')

        # Load the model
        self._load_checkpoint(load_best_checkpoint=True)

        self.trans_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 128)),
            transforms.Normalize(self.env_mean, self.env_std)
        ])

        # Set the folders
        self.output_path = os.path.join(self.output_path, 'test')

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        folder_time = time.strftime("%Y_%m_%d_%H_%M_%S")

        self.model_name = f"{self.model_name}_{folder_time}_test"

        # Compose the transformations
        self.trans_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 128)),
            transforms.Normalize(self.env.env_mean_rgb, self.env.env_std_rgb)
        ])

        wandb.init(
            project=self.env._name,
            name=self.model_name,
            id=self.model_name,
            resume=False
        )

        # Build the config params
        config_info = {
            'n_tests': self.n_tests,
            'n_exec_test': self.n_exec_test,
            'n_trajectory': self.n_trajectory,
            'env_mean_rgb': self.env.env_mean_rgb.tolist(),
            'env_std_rgb': self.env.env_std_rgb.tolist(),
            "random_seed": self.random_seed
        }

        # Save configuration params
        _save_csv(os.path.join(self.output_path, 'test_config.csv'), config_info)



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

        # Build state
        return (state_vision_rgb, state_proprioception)


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
        self.actor.load_state_dict(checkpoint["actor_state"])



        if load_best_checkpoint:
            print(f"[*] Loaded best checkpoint @ step {self.current_step} with mean_return: {self.best_mean_episodic_return}")
        else:
            print(f"[*] Loaded last checkpoint @ step {self.current_step} with mean_return: {self.best_mean_episodic_return}")

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

    for index, o in enumerate(last_obs_rollout):

        image_array = np.concatenate((trans(o[-1]["frame_top"]), trans(o[-1]["frame_front"])), axis=0)

        image = Image.fromarray(image_array, mode="RGB")

        video.append(np.asarray(image))

    video = np.asarray(video)
    video = np.transpose(video, (0, 3, 1, 2))

    return video


def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument("--resume", type=str, required=False, dest='model_name',
                     help="Resume training {model_name}.")

    return vars(arg.parse_args())


if __name__ == "__main__":

    args = parse_arguments()

    TestTask(headless=True, model_name=args['model_name']).test()

