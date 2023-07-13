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
from Environment import Conf


class Environment:

    def __init__(
        self,
        conf: Conf,
        trans_mean_std=None,
        headless: bool = True,
        _change_inst_step: int = 1000,
        stack_obs: int = 4,
        random_seed: int = 1
    ):

        self._conf = conf
        self._change_inst_step = _change_inst_step
        self._stack_obs = stack_obs
        self._random_seed = random_seed

        self.pr = PyRep()
        self.pr.launch(conf.scene_file, headless=headless)

        self.NAO = None
        self.instruction = None
        self.reward_function = None
        self._obs = deque(maxlen=self._stack_obs)

        # Instantiate visual sensors
        self.cam_top = VisionSensor('Vision_Top')
        self.cam_front = VisionSensor('Vision_Front')

        # Create tokenizer and vocab
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = self._build_vocab(self._conf.instruction_set)

        # Compose the transformations
        if trans_mean_std is None:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((128, 64))
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((128, 64)),
                transforms.Normalize(trans_mean_std[0], trans_mean_std[1])
            ])

        # Set seed
        random.seed(self._random_seed)

        # Configure init scene
        self._conf.configure()

    def start(self):

        # Start simulation
        self.pr.start()
        self.pr.step()

        # Instantiate Nao
        self.NAO = Nao()

        self._obs.clear()

        # Get instruction
        self.instruction, self.reward_function = random.choice(self._conf.instruction_set)

        # Populate initial observations
        for i in range(self._stack_obs):
            self._observe()

    def _observe(self):

        # Get frame from top and front
        frame_top = (self.cam_top.capture_rgb() * 255).astype(np.uint8)
        frame_front = (self.cam_front.capture_rgb() * 255).astype(np.uint8)

        # Get joint positions
        joint_position_flatten = []
        for j in self.NAO.get_joint_positions():
            joint_position_flatten += j

        # Build observation state
        observation = (self.instruction, joint_position_flatten, frame_top, frame_front)

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
        instruction_index = torch.tensor(self.vocab(instruction_token), device="cuda")

        joint_position_tensor = torch.empty((len(self._obs), 26), dtype=torch.float, device="cuda")

        image_tensor = torch.empty((len(self._obs), 3, 128, 128), dtype=torch.float, device="cuda")

        for i, o in enumerate(self._obs):

            # Convert joint position to tensor
            joint_position_tensor[i] = torch.tensor(o[1], device="cuda")

            image_top = o[2]
            image_front = o[3]

            # Convert state to tensor
            image_top_tensor = self.trans(image_top)
            image_font_tensor = self.trans(image_front)

            # Cat all images into a single one
            images_stacked = torch.cat((image_top_tensor, image_font_tensor), dim=2)

            image_tensor[i] = images_stacked

        image = image_tensor.flatten(0, 1)

        state = (instruction_index, joint_position_tensor, image)

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

