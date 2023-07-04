from .Nao import Nao
from pyrep.backend import sim


def _touch_green_cube(nao: Nao):
    cube_handle = sim.simGetObjectHandle("Cube_Green")

    if nao.check_collisions(cube_handle):
        return 1

    return 0


def _touch_red_cube(nao: Nao):
    cube_handle = sim.simGetObjectHandle("Cube_Red")

    if nao.check_collisions(cube_handle):
        return 1

    return 0


def _touch_blue_cube(nao: Nao):
    cube_handle = sim.simGetObjectHandle("Cube_Blue")

    if nao.check_collisions(cube_handle):
        return 1

    return 0


class InstructionSet:

    def __init__(self):
        self._instructions_set = (
            ("Touch the blue cube.", _touch_blue_cube),
            ("Touch the red cube.", _touch_red_cube),
            ("Touch the green cube.", _touch_green_cube),
        )

    def __len__(self):
        return len(self._instructions_set)

    def __getitem__(self, position):
        return self._instructions_set[position]
