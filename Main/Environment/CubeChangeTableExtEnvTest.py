import random

from .Environment import Environment
from pyrep.backend import sim
from pyrep.objects.shape import Shape


class CubeChangeTableExtEnvTest(Environment):

    def __init__(self, **kwargs):

        scene_file = 'Main/Scenes/Cubes_Simple_Task.ttt'

        self.cube_green: Shape | None = None
        self.cube_red: Shape | None = None
        self.cube_blue: Shape | None = None
        self.highTable: Shape | None = None

        # Initialize parent class
        super().__init__("CubeChangeTableExtTest", scene_file, **kwargs)


    def configure(self, id_test: int) -> None:
        self._load_objects()

        if id_test == 0:
            self.highTable.set_color(color=[255.0/255.0, 255.0/255.0, 255.0/255.0])
        elif id_test == 1:
            self.highTable.set_color(color=[179.0/255.0, 171.0/255.0, 171.0/255.0])
        elif id_test == 2:
            self.highTable.set_color(color=[183.0/255.0, 120.0/255.0, 120.0/255.0])
        elif id_test == 3:
            self.highTable.set_color(color=[102.0/255.0, 56.0/255.0, 56.0/255.0])
        elif id_test == 4:
            self.highTable.set_color(color=[255.0/255.0, 0.0/255.0, 0.0/255.0])
        elif id_test == 5:
            self.highTable.set_color(color=[255.0/255.0, 128.0/255.0, 0.0/255.0])
        elif id_test == 6:
            self.highTable.set_color(color=[255.0/255.0, 255.0/255.0, 0.0/255.0])
        elif id_test == 7:
            self.highTable.set_color(color=[128.0/255.0, 255.0/255.0, 0.0/255.0])
        elif id_test == 8:
            self.highTable.set_color(color=[0.0/255.0, 255.0/255.0, 0.0/255.0])
        elif id_test == 9:
            self.highTable.set_color(color=[0.0/255.0, 255.0/255.0, 128.0/255.0])
        elif id_test == 10:
            self.highTable.set_color(color=[0.0/255.0, 255.0/255.0, 255.0/255.0])
        elif id_test == 11:
            self.highTable.set_color(color=[0.0/255.0, 128.0/255.0, 255.0/255.0])
        elif id_test == 12:
            self.highTable.set_color(color=[0.0/255.0, 0.0/255.0, 255.0/255.0])
        elif id_test == 13:
            self.highTable.set_color(color=[127.0/255.0, 0.0/255.0, 255.0/255.0])
        elif id_test == 14:
            self.highTable.set_color(color=[255.0/255.0, 0.0/255.0, 255.0/255.0])
        elif id_test == 15:
            self.highTable.set_color(color=[255.0/255.0, 0.0/255.0, 127.0/255.0])
        elif id_test == 16:
            self.highTable.set_color(color=[128.0/255.0, 128.0/255.0, 128.0/255.0])
    def reward(self):
        return self._touch_green_cube()

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

