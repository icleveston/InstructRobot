from .Nao import Nao
from .InstructionSet import InstructionSet
from pyrep.backend import sim


class CubeSimpleSet(InstructionSet):

    def __init__(self):

        instructions_set = (
            ("Touch the blue cube.", self._touch_blue_cube),
            ("Touch the red cube.", self._touch_red_cube),
            ("Touch the green cube.", self._touch_green_cube),
        )

        super().__init__("CubeSimpleSet", instructions_set)

        self.cube_green_handle = None
        self.cube_red_handle = None
        self.cube_blue_handle = None

    def _get_handles(self):
        self.cube_green_handle = sim.simGetObjectHandle("Cube_Green")
        self.cube_red_handle = sim.simGetObjectHandle("Cube_Red")
        self.cube_blue_handle = sim.simGetObjectHandle("Cube_Blue")

    def _touch_green_cube(self, nao: Nao):

        self._get_handles()

        if nao.check_collisions(self.cube_green_handle):
            return 10
        elif nao.check_collisions(self.cube_red_handle) or nao.check_collisions(self.cube_blue_handle):
            return -1
        else:
            return 0

    def _touch_red_cube(self, nao: Nao):

        self._get_handles()

        if nao.check_collisions(self.cube_red_handle):
            return 10
        elif nao.check_collisions(self.cube_green_handle) or nao.check_collisions(self.cube_blue_handle):
            return -1
        else:
            return 0

    def _touch_blue_cube(self, nao: Nao):

        self._get_handles()

        if nao.check_collisions(self.cube_blue_handle):
            return 10
        elif nao.check_collisions(self.cube_red_handle) or nao.check_collisions(self.cube_green_handle):
            return -1
        else:
            return 0

