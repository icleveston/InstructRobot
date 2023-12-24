import random

from .Nao import Nao
from .Environment import Environment
from pyrep.backend import sim
from pyrep.objects.shape import Shape


class CubeChangeTableIntEnv(Environment):

    def __init__(self, **kwargs):

        scene_file = 'Main/Scenes/Cubes_Simple_Int.ttt'

        self.cube_green: Shape | None = None
        self.cube_red: Shape | None = None
        self.cube_blue: Shape | None = None
        self.highTable: Shape | None = None

        super().__init__("CubeChangeTableInt", scene_file, **kwargs)

    def configure(self, id_rollout: int) -> None:
        self._load_objects()
        #self.cube_green.set_color(color=[255.0, 255.0, 255.0])
        if id_rollout <= 2:
            self.highTable.set_color(color=[255.0/255.0, 255.0/255.0, 255.0/255.0])
        elif id_rollout > 2 and  id_rollout <= 5:
            self.highTable.set_color(color=[179.0/255.0, 171.0/255.0, 171.0/255.0])
        elif id_rollout > 5 and  id_rollout <= 8:
            self.highTable.set_color(color=[183.0/255.0, 120.0/255.0, 120.0/255.0])
        elif id_rollout > 8:
            self.highTable.set_color(color=[102.0/255.0, 56.0/255.0, 56.0/255.0])


    def reward(self):
        return self._touch_green_cube() + self._touch_red_cube() + self._touch_blue_cube()

    def observe(self):

        # Get frame from top and front
        frame_top, frame_front = self.get_camera_frames()

        # Get proprioception
        proprioception = self.NAO.get_joint_positions()

        # Build observation array
        observation = {
            "frame_top": frame_top,
            "frame_front": frame_front,
            "proprioception": proprioception
        }

        return observation

    def _load_objects(self) -> None:

        # Load objects shapes from handles
        if self.cube_green is None:
            self.cube_green = Shape(name_or_handle=sim.simGetObjectHandle("Cube_Green"))
        if self.cube_red is None:
            self.cube_red = Shape(name_or_handle=sim.simGetObjectHandle("Cube_Red"))
        if self.cube_blue is None:
            self.cube_blue = Shape(name_or_handle=sim.simGetObjectHandle("Cube_Blue"))
        if self.highTable is None:
            self.highTable = Shape(name_or_handle=sim.simGetObjectHandle("_highTable"))

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

