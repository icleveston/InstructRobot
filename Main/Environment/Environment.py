import random
from pyrep import PyRep
import numpy as np
from .Nao import Nao
from pyrep.objects.vision_sensor import VisionSensor
from collections import deque
from . import Conf


class Environment:

    def __init__(
        self,
        conf: Conf,
        headless: bool = True,
        stack_obs: int = 4,
        random_seed: int = 1
    ):

        self._conf = conf
        self._stack_obs = stack_obs
        self.random_seed = random_seed

        self.pr = PyRep()
        self.pr.launch(conf.scene_file, headless=headless)

        self.NAO = None
        self.instruction = None
        self.reward_function = None
        self._obs = deque(maxlen=self._stack_obs)

        # Instantiate visual sensors
        self.cam_top = VisionSensor('Vision_Top')
        self.cam_front = VisionSensor('Vision_Front')

        # Set seed
        random.seed(self.random_seed)

    def start(self, warm_up_steps=10):

        # Start simulation
        self.pr.start()

        # Instantiate Nao
        self.NAO = Nao()

        # Clear observation array
        self._obs.clear()

        # Configure init scene
        self._conf.configure()

        # Warm up simulator
        for i in range(warm_up_steps):
            self.pr.step()

        # Get instruction
        self.instruction, self.reward_function = random.choice(self._conf.instruction_set)

        # Populate initial observations
        for i in range(self._stack_obs):
            self._observe()

    def _observe(self):

        # Get frame from top and front
        frame_top = (self.cam_top.capture_rgb() * 255).astype(np.uint8)
        frame_front = (self.cam_front.capture_rgb() * 255).astype(np.uint8)

        # Build observation state
        observation = (self.instruction, frame_top, frame_front)

        # Append the new observation
        self._obs.append(observation)

    def reset(self) -> deque:

        if self.pr.running:
            self.pr.stop()

        self.start()

        return self._obs

    def step(self, action: []) -> ():

        assert self.pr.running, "You must start the environment before any step."

        # Execute action in the simulator
        self.NAO.make_action(action)

        # Execute one step in the simulator
        self.pr.step()

        # Observe the environment
        self._observe()

        # Compute the reward
        reward = self.reward_function(self.NAO)

        return self._obs, reward

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

