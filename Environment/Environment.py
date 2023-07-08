import random
import torch
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchvision import transforms
from pyrep import PyRep
import numpy as np
from .Nao import Nao
from pyrep.objects.vision_sensor import VisionSensor
from collections import deque
from Environment import InstructionSet


class Environment:

    def __init__(
        self,
        scene_file: str,
        instruction_set: InstructionSet,
        trans_mean_std=None,
        headless: bool = True,
        _change_inst_step: int = 1000,
        stack_obs: int = 4,
        random_seed: int = 1
    ):

        self._instruction_set = instruction_set
        self._change_inst_step = _change_inst_step
        self._stack_obs = stack_obs
        self.random_seed = random_seed

        self.pr = PyRep()
        self.pr.launch(scene_file, headless=headless)

        self.NAO = None
        self.instruction = None
        self.reward_function = None
        self._obs = deque(maxlen=self._stack_obs)

        # Instantiate visual sensors
        self.cam_top = VisionSensor('Vision_Top')
        self.cam_front = VisionSensor('Vision_Front')

        # Create tokenizer and vocab
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = self._build_vocab(self._instruction_set)

        # Compose the transformations
        if trans_mean_std is not None:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((128, 64)),
                transforms.Normalize(trans_mean_std[0], trans_mean_std[1])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((128, 64))
            ])

    def start(self):

        # Set seed
        random.seed(self.random_seed)

        # Start simulation
        self.pr.start()
        self.pr.step()

        # Instantiate Nao
        self.NAO = Nao()

        self._obs.clear()

        # Get instruction
        self.instruction, self.reward_function = random.choice(self._instruction_set)

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

    def reset(self):

        if self.pr.running:
            self.pr.stop()

        self.start()

        return self._format_obs(), self._obs

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

        return self._format_obs(), reward, self._obs

    def _format_obs(self):

        # Tokenize instruction
        instruction_token = self.tokenizer(self._obs[-1][0])

        # Get instructions indexes
        instruction_index = torch.tensor(self.vocab(instruction_token))

        image_tensor = torch.empty((len(self._obs), 3, 128, 128), dtype=torch.float)

        for i, o in enumerate(self._obs):
            image_top = o[1]
            image_front = o[2]

            # Convert state to tensor
            image_top_tensor = self.trans(image_top)
            image_font_tensor = self.trans(image_front)

            # Cat all images into a single one
            images_stacked = torch.cat((image_top_tensor, image_font_tensor), dim=2)

            image_tensor[i] = images_stacked

        image = image_tensor.flatten(0, 1)

        state = (instruction_index, image)

        return state

    def _build_vocab(self, instruction_set):

        def build_vocab(dataset: []):
            for instruction, _ in dataset:
                yield self.tokenizer(instruction)

        vocab = build_vocab_from_iterator(build_vocab(instruction_set), specials=["<UNK>"])
        vocab.set_default_index(vocab["<UNK>"])

        return vocab

    def close(self):
        self.pr.stop()
        self.pr.shutdown()

