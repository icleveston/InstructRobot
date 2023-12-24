import random
from typing import List, Any

import torch
from torchvision import transforms
from pyrep import PyRep
import numpy as np
from .Nao import Nao
from pyrep.objects.vision_sensor import VisionSensor
from collections import deque
from abc import ABC, abstractmethod
from PIL import Image


class Environment(ABC):

    def __init__(
            self,
            name,
            scene_file,
            num_styles,
            headless: bool = True,
            stack_obs: int = 3,
            random_seed: int = 1):

        self._name = name
        self._scene_file = scene_file
        self._stack_obs = stack_obs * 2
        self.random_seed = random_seed
        self.env_mean = None
        self.env_std = None
        self.env_min_values = None
        self.env_max_values = None
        self.num_styles = num_styles

        self.pr = PyRep()
        self.pr.launch(self._scene_file, headless=headless)

        self.NAO = None

        self._obs = deque(maxlen=self._stack_obs)

        # Instantiate visual sensors
        self.cam_top = VisionSensor('Vision_Top')
        self.cam_front = VisionSensor('Vision_Front')

        # Set seed
        random.seed(self.random_seed)

        # Compute image mean and std
        self.env_mean_rgb, self.env_std_rgb, self.env_min_values, self.env_max_values = self._compute_env_stat_metrics(height=64, width=128)

    def __str__(self) -> str:
        return self._name

    @property
    def scene_file(self):
        return self._scene_file

    @abstractmethod
    def configure(self, id_rollout: int):
        pass

    @abstractmethod
    def reward(self):
        pass

    @abstractmethod
    def observe(self):
        pass

    def start(self, id_rollout: int, warm_up_steps=10):

        # Start simulation
        self.pr.start()

        # Instantiate Nao
        self.NAO = Nao()

        # Clear observation array
        self._obs.clear()

        # Configure init scene
        self.configure(id_rollout)

        # Warm up simulator
        for i in range(warm_up_steps):
            self.pr.step()

        # Populate initial observations
        for i in range(self._stack_obs):
            self._obs.append(self.observe())

    def reset(self, id_rollout: int) -> list[Any]:

        if self.pr.running:
            self.pr.stop()

        self.start(id_rollout)

        return [o for i, o in enumerate(self._obs) if i % 2 == 1]

    def step(self, action: []) -> ():

        assert self.pr.running, "You must start the environment before any step."

        # Execute action in the simulator
        self.NAO.make_action(action)

        # Execute one step in the simulator
        self.pr.step()

        # Observe the environment
        self._obs.append(self.observe())

        # Compute the reward
        reward = self.reward()

        return [o for i, o in enumerate(self._obs) if i % 2 == 1], reward

    def get_camera_frames(self):

        # Get frame from top and front
        frame_top = (self.cam_top.capture_rgb() * 255).astype(np.uint8)
        frame_front = (self.cam_front.capture_rgb() * 255).astype(np.uint8)

        return frame_top, frame_front

    def validate_joints_nao(self):

        self.reset()

        for is_new_joint, action in self.NAO.validate_joints():

            if is_new_joint:
                self.reset()

            self.step(action)

    def validate_collisions_nao(self):

        self.reset()

        for action in self.NAO.validate_collisions():
            self.step(action)

    def close(self):
        self.pr.stop()
        self.pr.shutdown()

    def _compute_env_stat_metrics(self, width=128, height=64, n_observations_computation=100, is_gray: bool = False):

        # Add basic transformations
        transformations_array = [
            transforms.ToTensor(),
            transforms.Resize((height, width))
        ]

        if is_gray:
            transformations_array.append(transforms.Grayscale())

        # Compose the transformations
        trans = transforms.Compose(transformations_array)

        obs = self.reset(0)

        image_tensor = torch.empty(
            (len(obs) * n_observations_computation * self.num_styles, 1 if is_gray else 3, height * 2, width),
            dtype=torch.float)
        index = 0

        for id_rollout in range(self.num_styles):
            obs = self.reset(id_rollout)

            for _ in range(n_observations_computation):

                action = [random.randrange(-1, 1) for _ in range(26)]

                obs, _ = self.step(action)

                for o in obs:
                    image_top = o["frame_top"]
                    image_front = o["frame_front"]

                    # Convert state to tensor
                    image_top_tensor = trans(image_top)
                    image_font_tensor = trans(image_front)

                    # Cat all images into a single one
                    images_stacked = torch.cat((image_top_tensor, image_font_tensor), dim=1)

                    image_tensor[index] = images_stacked

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
