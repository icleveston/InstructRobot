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


class Environment(ABC):

    def __init__(
            self,
            name,
            scene_file,
            headless: bool = True,
            stack_obs: int = 3,
            random_seed: int = 1):

        self._name = name
        self._scene_file = scene_file
        self._stack_obs = stack_obs * 2
        self.random_seed = random_seed
        self.env_mean = None
        self.env_std = None

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
        self.env_mean, self.env_std = self._compute_env_mean_std()

    def __str__(self) -> str:
        return self._name

    @property
    def scene_file(self):
        return self._scene_file

    @abstractmethod
    def configure(self):
        pass

    @abstractmethod
    def reward(self):
        pass

    @abstractmethod
    def observe(self):
        pass

    def start(self, warm_up_steps=10):

        # Start simulation
        self.pr.start()

        # Instantiate Nao
        self.NAO = Nao()

        # Clear observation array
        self._obs.clear()

        # Configure init scene
        self.configure()

        # Warm up simulator
        for i in range(warm_up_steps):
            self.pr.step()

        # Populate initial observations
        for i in range(self._stack_obs):
            self._obs.append(self.observe())

    def reset(self) -> list[Any]:

        if self.pr.running:
            self.pr.stop()

        self.start()

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

    def _compute_env_mean_std(self, height=64, width=128, n_observations_computation=5):

        obs = self.reset()

        # Compose the transformations
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((width, height))
        ])

        image_tensor = torch.empty((len(obs) * n_observations_computation, 3, width, height*2), dtype=torch.float)

        index = 0

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
                images_stacked = torch.cat((image_top_tensor, image_font_tensor), dim=2)

                image_tensor[index] = images_stacked

                index += 1

        return _online_mean_and_sd(image_tensor)


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

