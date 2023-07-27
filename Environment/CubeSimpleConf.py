from .Nao import Nao
from .Conf import Conf
from pyrep.backend import sim


class CubeSimpleConf(Conf):

    def __init__(self):

        scene_file = 'Scenes/Cubes_Simple_Touch_Sensor.ttt'

        instructions_set = (
            ("Touch the blue cube.", self._touch_blue_cube),
            ("Touch the red cube.", self._touch_red_cube),
            ("Touch the green cube.", self._touch_green_cube),
        )

        super().__init__("CubeSimpleSet", scene_file, instructions_set)

        self.cube_green_handle = None
        self.cube_red_handle = None
        self.cube_blue_handle = None

    def configure(self):
        pass

    def _get_collisions(self, nao: Nao):

        if self.cube_green_handle is None:
            self.cube_green_handle = sim.simGetObjectHandle("Cube_Green")
        if self.cube_red_handle is None:
            self.cube_red_handle = sim.simGetObjectHandle("Cube_Red")
        if self.cube_blue_handle is None:
            self.cube_blue_handle = sim.simGetObjectHandle("Cube_Blue")

        n_collision_blue = nao.check_collisions(self.cube_blue_handle)
        n_collision_red = nao.check_collisions(self.cube_red_handle)
        n_collision_green = nao.check_collisions(self.cube_green_handle)

        return n_collision_blue, n_collision_red, n_collision_green

    def _touch_green_cube(self, nao: Nao):

        n_collision_blue, n_collision_red, n_collision_green = self._get_collisions(nao)

        if n_collision_green:
            return n_collision_green/6
        elif n_collision_red:
            return -0.1 * n_collision_red/6
        elif n_collision_blue:
            return -0.1 * n_collision_blue/6
        else:
            return 0

    def _touch_red_cube(self, nao: Nao):

        n_collision_blue, n_collision_red, n_collision_green = self._get_collisions(nao)

        if n_collision_red:
            return n_collision_red/6
        elif n_collision_blue:
            return -0.1 * n_collision_blue/6
        elif n_collision_green:
            return -0.1 * n_collision_green/6
        else:
            return 0

    def _touch_blue_cube(self, nao: Nao):

        n_collision_blue, n_collision_red, n_collision_green = self._get_collisions(nao)

        if n_collision_blue:
            return n_collision_blue/6
        elif n_collision_red:
            return -0.1 * n_collision_red/6
        elif n_collision_green:
            return -0.1 * n_collision_green/6
        else:
            return 0

