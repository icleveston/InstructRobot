import random
from typing import List, Any
import torch
from torchvision import transforms
import numpy as np
from collections import deque
from abc import ABC, abstractmethod
from PIL import Image
from procgen import ProcgenGym3Env


class Environment(ABC):

    def __init__(
            self,
            name,
            headless: bool = True,
            stack_obs: int = 3,
            random_seed: int = 1):

        self._name = name
        self._stack_obs = stack_obs*2
        self.random_seed = random_seed
        self.env_mean = None
        self.env_std = None
        self.env_min_values = None
        self.env_max_values = None

        # Set seed
        random.seed(self.random_seed)

        self.env = ProcgenGym3Env(num=1, env_name=name, num_levels=0, start_level=0, distribution_mode="easy",
                                  render_mode="rgb_array")

        self._obs = deque(maxlen=self._stack_obs)
        self._observe()

        # Compute image mean and std
        self.env_mean_rgb, self.env_std_rgb, self.env_min_values, self.env_max_values = self._compute_env_stat_metrics()

    def __str__(self) -> str:
        return self._name

    def reset(self) -> list[Any]:

        return [o for i, o in enumerate(self._obs) if i % 2 == 1]

    def _observe(self):

        obs = self.env.get_info()

        img = Image.fromarray(obs[0]['rgb'])
        img = img.resize((256, 256), Image.Resampling.BOX)

        null_action = 0
        for i in range(self._stack_obs):
            observation = (img, null_action)
            self._obs.append(observation)

    def step(self, action: int):

        self.env.act({"action": np.array(action)})

        ext_reward, _, done = self.env.observe()

        obs = self.env.get_info()

        img = Image.fromarray(obs[0]['rgb'])
        img = img.resize((256, 256), Image.Resampling.BOX)

        self._obs.append((img, action))

        return [o for i, o in enumerate(self._obs) if i % 2 == 1], ext_reward, done

    def close(self):
        self.env.close()

    def _compute_env_stat_metrics(self, n_observations_computation=1000, is_gray: bool = False):

        # Add basic transformations
        transformations_array = [
            transforms.ToTensor()
        ]

        if is_gray:
            transformations_array.append(transforms.Grayscale())

        # Compose the transformations
        trans = transforms.Compose(transformations_array)

        height, width = (256, 256)
        image_tensor = torch.empty((self._stack_obs*n_observations_computation, 1 if is_gray else 3, height, width),
                                   dtype=torch.float)
        index = 0
        for _ in range(n_observations_computation):

            print(self.env.ac_space)

            self.env.act(self.env.ac_space)

            obs = self.env.get_info()

            for o in obs:
                image_tensor[index] = trans(o[0])
                index += 1

        mean_images, std_images = _online_mean_and_sd(image_tensor, is_gray)

        transformations_norm = [
            transforms.Normalize(mean=mean_images, std=std_images)
        ]
        trans = transforms.Compose(transformations_norm)
        image_tensor = trans(image_tensor)
        if is_gray:
            min_values = torch.min(image_tensor)
            max_values = torch.max(image_tensor)
        else:
            min_values = torch.min(image_tensor[:,0,:,:]), torch.min(image_tensor[:,1,:,:]), torch.min(image_tensor[:,2,:,:])
            max_values = torch.max(image_tensor[:,0,:,:]), torch.max(image_tensor[:,1,:,:]), torch.max(image_tensor[:,2,:,:])

        return mean_images, std_images, list(min_values), list(max_values)


def _online_mean_and_sd(images: np.array, is_gray: bool = False) -> tuple:
    cnt = 0
    fst_moment = torch.empty(1 if is_gray else 3)
    snd_moment = torch.empty(1 if is_gray else 3)

    b, c, h, w = images.shape
    nb_pixels = b * h * w
    sum_ = torch.sum(images, dim=[0, 2, 3])
    sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
    fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
    snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

    cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
