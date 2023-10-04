import random
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data import get_tokenizer

from .Environment import Environment
from pyrep.backend import sim
from pyrep.objects.shape import Shape


class CubeSimpleExtEnv(Environment):

    def __init__(self, **kwargs):

        scene_file = 'Main/Scenes/Cubes_Simple.ttt'

        self._instructional_set = (
            ("Touch the blue cube.", self._touch_blue_cube),
            ("Touch the red cube.", self._touch_red_cube),
            ("Touch the green cube.", self._touch_green_cube),
        )

        self.active_instruction = None
        self.reward_function = None

        self.cube_green: Shape | None = None
        self.cube_red: Shape | None = None
        self.cube_blue: Shape | None = None

        # Create tokenizer and vocab
        self._tokenizer = get_tokenizer("basic_english")
        self._vocab = self._build_vocab(self._instructional_set)

        # Initialize parent class
        super().__init__("CubeSimpleExt", scene_file, **kwargs)

    def __len__(self) -> int:
        return len(self._instructional_set)

    def __getitem__(self, position) -> ():
        return self._instructional_set[position]

    @property
    def instructional_set(self):
        return self._instructional_set

    def configure(self) -> None:

        self._load_objects()

        # Get object's current position
        positions = [
            self.cube_green.get_position(),
            self.cube_red.get_position(),
            self.cube_blue.get_position()
        ]

        # Shuffle array of positions
        random.shuffle(positions)

        # Set objects position
        self.cube_green.set_position(positions[0])
        self.cube_red.set_position(positions[1])
        self.cube_blue.set_position(positions[2])

        # Get instruction
        self.active_instruction, self.reward_function = random.choice(self._instructional_set)

    def reward(self):
        return self.reward_function()

    def observe(self):

        # Get frame from top and front
        frame_top, frame_front = self.get_camera_frames()

        # Get proprioception
        proprioception = self.NAO.get_joint_positions()

        # Build observation array
        observation = {
            "frame_top": frame_top,
            "frame_front": frame_front,
            "proprioception": proprioception,
            "instruction": self.active_instruction
        }

        return observation

    def tokenize(self, instruction_token: str):

        # Tokenize instruction
        return self._vocab(self._tokenizer(instruction_token))

    def _load_objects(self) -> None:

        # Load objects shapes from handles
        if self.cube_green is None:
            self.cube_green = Shape(name_or_handle=sim.simGetObjectHandle("Cube_Green"))
        if self.cube_red is None:
            self.cube_red = Shape(name_or_handle=sim.simGetObjectHandle("Cube_Red"))
        if self.cube_blue is None:
            self.cube_blue = Shape(name_or_handle=sim.simGetObjectHandle("Cube_Blue"))

    def _get_collisions(self):

        self._load_objects()

        n_collision_blue, _ = self.NAO.check_collisions(self.cube_blue)
        n_collision_red, _ = self.NAO.check_collisions(self.cube_red)
        n_collision_green, _ = self.NAO.check_collisions(self.cube_green)

        return n_collision_blue, n_collision_red, n_collision_green

    def _touch_green_cube(self):

        n_collision_blue, n_collision_red, n_collision_green = self._get_collisions()

        return n_collision_green

    def _touch_red_cube(self):

        n_collision_blue, n_collision_red, n_collision_green = self._get_collisions()

        return n_collision_red

    def _touch_blue_cube(self):

        n_collision_blue, n_collision_red, n_collision_green = self._get_collisions()

        return n_collision_blue

    def _build_vocab(self, instruction_set):

        def build_vocab(dataset: []):
            for instruction, _ in dataset:
                yield self._tokenizer(instruction)

        vocab = build_vocab_from_iterator(build_vocab(instruction_set), specials=["<UNK>"])
        vocab.set_default_index(vocab["<UNK>"])

        return vocab

