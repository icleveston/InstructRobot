import random
import math
from .Environment import Environment
from pyrep.backend import sim
from pyrep.objects.shape import Shape
import numpy as np



class CubeSimpleExtEnv(Environment):

    def __init__(self, **kwargs):

        scene_file = 'Main/Scenes/Cubes_Simple_Task.ttt'



        self.cube_green: Shape | None = None
        self.cube_red: Shape | None = None
        self.cube_blue: Shape | None = None

        # Initialize parent class
        super().__init__("CubeExtStack", scene_file, **kwargs)


    def configure(self) -> None:
        pass

    def reward(self):
        self._load_objects()

        # Obtenha as dimensões do objeto
        dimensions = self.cube_green.get_bounding_box()

        min_x, max_x, min_y, max_y, min_z, max_z = dimensions
        width = max_x - min_x #largura
        height = max_y - min_y #altura
        depth = max_z - min_z #profundidade

        print(np.allclose(self.cube_green.get_position(), self.cube_red.get_position(), atol=width, rtol=0.0))
        print(self.cube_green.get_position()[-1])
        print(self.cube_red.get_position()[-1])

        stack_green_red = (np.allclose(self.cube_green.get_position(), self.cube_red.get_position(), atol=width, rtol=0.0)
                           and (self.cube_green.get_position()[-1] == 0.5345 or self.cube_red.get_position()[-1] == 0.5345))
        stack_green_blue = (np.allclose(self.cube_green.get_position(), self.cube_blue.get_position(), atol=width, rtol=0.0)
                            and (self.cube_green.get_position()[-1] == 0.5345 or self.cube_blue.get_position()[-1] == 0.5345))
        stack_red_blue = (np.allclose(self.cube_red.get_position(), self.cube_blue.get_position(), atol=width, rtol=0.0)
                          and (self.cube_red.get_position()[-1] == 0.5345 or self.cube_blue.get_position()[-1] == 0.5345))


        r = 0.0
        if stack_green_red:
            r += 1.0
        elif stack_green_blue:
            r += 1.0
        elif stack_red_blue:
            r += 1.0

        return r

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

