import random
import math
from .Environment import Environment
from pyrep.backend import sim
from pyrep.objects.shape import Shape


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

        dist_green_red = (abs(self.cube_green.get_position()[0] - self.cube_red.get_position()[0]) < 0.07 and
                          abs(self.cube_green.get_position()[1] - self.cube_red.get_position()[1]) < 0.07)
        dist_green_blue = (abs(self.cube_green.get_position()[0] - self.cube_blue.get_position()[0]) < 0.07 and
                           abs(self.cube_green.get_position()[1] - self.cube_blue.get_position()[1]) < 0.07)
        dist_red_blue = (abs(self.cube_red.get_position()[0] - self.cube_blue.get_position()[0]) < 0.07 and
                         abs(self.cube_red.get_position()[1] - self.cube_blue.get_position()[1]) < 0.07)
        dist_three_cubes = (abs(self.cube_green.get_position()[0] - self.cube_red.get_position()[0]) < 0.07 and
        abs(self.cube_green.get_position()[1] - self.cube_red.get_position()[1]) < 0.07) and
        (abs(self.cube_red.get_position()[0] - self.cube_blue.get_position()[0]) < 0.07 and
         abs(self.cube_red.get_position()[1] - self.cube_blue.get_position()[1]) < 0.07)

        stack_green_red = dist_green_red and ((self.cube_green.get_position()[2] == 0.5345 and self.cube_red.get_position()[2] > 0.5360) or (self.cube_green.get_position()[2] > 0.5360 and self.cube_red.get_position()[2] == 0.5345))
        stack_green_blue = dist_green_blue and ((self.cube_green.get_position()[2] == 0.5345 and self.cube_blue.get_position()[2] > 0.5360) or (self.cube_green.get_position()[2] > 0.5360 and self.cube_blue.get_position()[2] == 0.5345))
        stack_red_blue = dist_red_blue and ((self.cube_red.get_position()[2] == 0.5345 and self.cube_blue.get_position()[2] > 0.5360) or (self.cube_red.get_position()[2] > 0.5360 and self.cube_blue.get_position()[2] == 0.5345))

        stack_green_red_blue = dist_three_cubes and (self.cube_green.get_position()[2] == 0.5345 and self.cube_red.get_position()[2] > self.cube_green.get_position()[2] and self.cube_blue.get_position()[2] > self.cube_red.get_position()[2])
        stack_green_blue_red = dist_three_cubes and (self.cube_green.get_position()[2] == 0.5345 and self.cube_blue.get_position()[2] > self.cube_green.get_position()[2] and self.cube_red.get_position()[2] > self.cube_blue.get_position()[2])
        stack_blue_green_red = dist_three_cubes and (self.cube_blue.get_position()[2] == 0.5345 and self.cube_green.get_position()[2] > self.cube_blue.get_position()[2] and self.cube_red.get_position()[2] > self.cube_green.get_position()[2])
        stack_blue_red_green = dist_three_cubes and (self.cube_blue.get_position()[2] == 0.5345 and self.cube_red.get_position()[2] > self.cube_blue.get_position()[2] and self.cube_green.get_position()[2] > self.cube_red.get_position()[2])
        stack_red_green_blue = dist_three_cubes and (self.cube_red.get_position()[2] == 0.5345 and self.cube_green.get_position()[2] > self.cube_red.get_position()[2] and self.cube_blue.get_position()[2] > self.cube_green.get_position()[2])
        stack_red_blue_green = dist_three_cubes and (self.cube_red.get_position()[2] == 0.5345 and self.cube_blue.get_position()[2] > self.cube_red.get_position()[2] and self.cube_green.get_position()[2] > self.cube_blue.get_position()[2])

        r = 0.0
        if stack_green_red:
            r += 1.0
        elif stack_green_blue:
            r += 1.0
        elif stack_red_blue:
            r += 1.0
        elif stack_green_red_blue:
            r += 1.0
        elif stack_green_blue_red:
            r += 1.0
        elif stack_blue_green_red:
            r += 1.0
        elif stack_blue_red_green:
            r += 1.0
        elif stack_blue_red_green:
            r += 1.0
        elif stack_red_green_blue:
            r += 1.0
        elif stack_red_blue_green:
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

