import random
from pyrep import PyRep
import numpy as np
from pyrep.robots.arms.nao_arm import NAOLeftArm, NAORightArm
from pyrep.robots.end_effectors.nao_hand import NAOHand
from .Nao import Nao
from pyrep.objects.vision_sensor import VisionSensor
from collections import deque
from Environment import InstructionSet


class Environment:

    def __init__(
        self,
        scene_file: str,
        instruction_set: InstructionSet,
        headless: bool = True,
        _change_inst_step: int = 1000,
        stack_obs: int = 4,
        random_seed: int = 1
    ):

        self._instruction_set = instruction_set
        self._change_inst_step = _change_inst_step
        self._stack_obs = stack_obs

        random.seed(random_seed)

        self.pr = PyRep()
        self.pr.launch(scene_file, headless=headless)

        self._global_step = None
        self.NAO = None
        self._active_instruction = None
        self._obs = deque(maxlen=self._stack_obs)

        # Instantiate visual sensors
        self.cam_top = VisionSensor('Vision_Top')
        self.cam_front = VisionSensor('Vision_Front')

        # Start environment
        self.start()

    def start(self):

        self._global_step = 0

        # Start simulation
        self.pr.start()
        self.pr.step()

        # Instantiate Nao
        self.NAO = Nao(NAOLeftArm(), NAORightArm(), NAOHand(), NAOHand())

        # Populate initial observations
        for i in range(self._stack_obs):
            self._observe()

    def _observe(self) -> ():

        # Get frame from top and front
        frame_top = (self.cam_top.capture_rgb() * 255).astype(np.uint8)
        frame_front = (self.cam_front.capture_rgb() * 255).astype(np.uint8)

        # Get instruction
        instruction, reward_function = self._get_instruction()

        # Build observation state
        observation = (instruction, frame_top, frame_front)

        # Append the new observation
        self._obs.append(observation)

        return reward_function

    def reset(self) -> deque:

        if self.pr.running:
            self.pr.stop()

        self.start()

        return self._obs

    def step(self, action: []) -> ():

        assert self.pr.running, "You must start the environment before any step."

        # Increment the global counter
        self._global_step += 1

        # Execute action in the simulator
        self.NAO.make_action(action)

        # Execute one step in the simulator
        self.pr.step()

        # Observe and get the reward function
        reward_function = self._observe()

        # Compute the reward
        reward = reward_function(self.NAO)

        return self._obs, reward

    def close(self):
        self.pr.stop()
        self.pr.shutdown()

    def _get_instruction(self):

        if self._global_step % self._change_inst_step == 0:
            self._active_instruction = random.choice(self._instruction_set)

        return self._active_instruction

