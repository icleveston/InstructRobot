import random

from .Nao import Nao
from .Conf import Conf
from pyrep.backend import sim
from pyrep.objects.shape import Shape


class CubeSimpleConf(Conf):

    def __init__(self):

        scene_file = 'Scenes/Cubes_Simple.ttt'

        instructions_set = (
            ("Touch the blue cube.", self._touch_blue_cube),
            ("Touch the red cube.", self._touch_red_cube),
            ("Touch the green cube.", self._touch_green_cube),
        )

        super().__init__("CubeSimpleSet", scene_file, instructions_set)

        self.cube_green: Shape = None
        self.cube_red: Shape = None
        self.cube_blue: Shape = None

    def configure(self) -> None:

        if False:

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

    def _load_objects(self) -> None:

        # Load objects shapes from handles
        if self.cube_green is None:
            self.cube_green = Shape(name_or_handle=sim.simGetObjectHandle("Cube_Green"))
        if self.cube_red is None:
            self.cube_red = Shape(name_or_handle=sim.simGetObjectHandle("Cube_Red"))
        if self.cube_blue is None:
            self.cube_blue = Shape(name_or_handle=sim.simGetObjectHandle("Cube_Blue"))

    def _get_collisions(self, nao: Nao):

        self._load_objects()

        n_collision_blue, _ = nao.check_collisions(self.cube_blue)
        n_collision_red, _ = nao.check_collisions(self.cube_red)
        n_collision_green, _ = nao.check_collisions(self.cube_green)

        return n_collision_blue, n_collision_red, n_collision_green

    def _touch_green_cube(self, nao: Nao):

        n_collision_blue, n_collision_red, n_collision_green = self._get_collisions(nao)

        if n_collision_green:
            return 10 #n_collision_green
        elif n_collision_red or n_collision_blue:
            return -1 #-0.1 * n_collision_red - 0.1 * n_collision_blue
        else:
            return 0

    def _touch_red_cube(self, nao: Nao):

        n_collision_blue, n_collision_red, n_collision_green = self._get_collisions(nao)

        if n_collision_red:
            return 10 #n_collision_red
        elif n_collision_blue or n_collision_green:
            return -1 # -0.1 * n_collision_blue - 0.1 * n_collision_green
        else:
            return 0

    def _touch_blue_cube(self, nao: Nao):

        n_collision_blue, n_collision_red, n_collision_green = self._get_collisions(nao)

        if n_collision_blue:
            return 10 #n_collision_blue
        elif n_collision_red or n_collision_green:
            return -1 #-0.1 * n_collision_red - 0.1 * n_collision_green
        else:
            return 0

