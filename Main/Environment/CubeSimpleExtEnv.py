import random
import math
from .Environment import Environment
from pyrep.backend import sim
from pyrep.objects.shape import Shape
import numpy as np
from numpy import round



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

        width = 0.07
        table_height = 0.5345
        heigth_two_cubes = table_height + 0.09
        heigth_three_cubes = table_height + 2*0.09

        cube_green_pos = round(self.cube_green.get_position(),4)
        cube_red_pos = round(self.cube_red.get_position(),4)
        cube_blue_pos = round(self.cube_blue.get_position(),4)

        close_green_red = abs(cube_green_pos[0] - cube_red_pos[0]) <= width and abs(cube_green_pos[1] - cube_red_pos[1]) <= width
        close_green_blue = abs(cube_green_pos[0] - cube_blue_pos[0]) <= width and abs(cube_green_pos[1] - cube_blue_pos[1]) <= width
        close_red_blue = abs(cube_red_pos[0] - cube_blue_pos[0]) <= width and abs(cube_red_pos[1] - cube_blue_pos[1]) <= width

        stack_green_red = close_green_red and ((cube_green_pos[-1] == table_height and cube_red_pos[-1] >= table_height and cube_red_pos[-1] <= heigth_two_cubes) or
                                (cube_red_pos[-1] == table_height and cube_green_pos[-1] >= table_height and cube_green_pos[-1] <= heigth_two_cubes))
         
        stack_green_blue = close_green_blue and ((cube_green_pos[-1] == table_height and cube_blue_pos[-1] >= table_height and cube_blue_pos[-1] <= heigth_two_cubes) or
                                (cube_blue_pos[-1] == table_height and cube_green_pos[-1] >= table_height and cube_green_pos[-1] <= heigth_two_cubes))

        stack_red_blue = close_red_blue and ((cube_red_pos[-1] == table_height and cube_blue_pos[-1] >= table_height and cube_blue_pos[-1] <= heigth_two_cubes) or
                                (cube_blue_pos[-1] == table_height and cube_red_pos[-1] >= table_height and cube_red_pos[-1] <= heigth_two_cubes))

        sum_high_three_cubes = cube_green_pos[-1]+cube_red_pos[-1]+cube_blue_pos[-1]
        stack_all = close_green_red and close_green_blue and close_red_blue and (sum_high_three_cubes > heigth_two_cubes and sum_high_three_cubes <= heigth_three_cubes)

        r = 0.0
        if stack_green_red:
            r += 1.0
        elif stack_green_blue:
            r += 1.0
        elif stack_red_blue:
            r += 1.0
        elif stack_all:
            r += 2.0

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

