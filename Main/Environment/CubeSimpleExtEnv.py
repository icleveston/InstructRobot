import random
import math
from .Environment import Environment
from pyrep.backend import sim
from pyrep.objects.shape import Shape
import numpy as np
from numpy import round, abs


class CubeSimpleExtEnv(Environment):

    def __init__(self, **kwargs):

        scene_file = 'Main/Scenes/Cubes_Shelf.ttt'



        self.cube_green: Shape | None = None
        self.cube_red: Shape | None = None
        self.cube_blue: Shape | None = None
        self.shelf_1: Shape | None = None
        self.shelf_2: Shape | None = None

        # Initialize parent class
        super().__init__("CubeExtStack", scene_file, **kwargs)


    def configure(self) -> None:
        pass

    def reward(self):
        self._load_objects()

        min_x, max_x, min_y, max_y, min_z, max_z = self.shelf_1.get_bounding_box()

        width = max_x - min_x
        height = max_y - min_y
        depth = max_z - min_z

        print(width)
        print(height)

        pos_green_cube = round(self.cube_green.get_position(),4)
        pos_red_cube = round(self.cube_red.get_position(),4)
        pos_blue_cube = round(self.cube_blue.get_position(),4)

        pos_shelf1 = round(self.shelf_1.get_position(), 4)
        pos_shelf2 = round(self.shelf_2.get_position(), 4)

        cube_green_shelf_1 = abs(pos_green_cube[0] - pos_shelf1[0]) < width/2 and abs(pos_green_cube[1] - pos_shelf1[1]) < height/2 and pos_green_cube[-1] >= pos_shelf1[-1] and pos_green_cube[-1] <= (pos_shelf1[-1] + depth/2)
        cube_red_shelf_1 = abs(pos_red_cube[0] - pos_shelf1[0]) < width/2 and abs(pos_red_cube[1] - pos_shelf1[1]) < height/2 and pos_red_cube[-1] >= pos_shelf1[-1] and pos_red_cube[-1] <= (pos_shelf1[-1] + depth/2)
        cube_blue_shelf_1 = abs(pos_blue_cube[0] - pos_shelf1[0]) < width/2 and abs(pos_blue_cube[1] - pos_shelf1[1]) < height/2 and pos_blue_cube[-1] >= pos_shelf1[-1] and pos_blue_cube[-1] <= (pos_shelf1[-1] + depth/2)

        cube_green_shelf_2 = abs(pos_green_cube[0] - pos_shelf2[0]) < width / 2 and abs(
            pos_green_cube[1] - pos_shelf2[1]) < height / 2 and pos_green_cube[-1] >= pos_shelf2[-1] and pos_green_cube[
                                 -1] <= (pos_shelf2[-1] + depth / 2)
        cube_red_shelf_2 = abs(pos_red_cube[0] - pos_shelf2[0]) < width / 2 and abs(
            pos_red_cube[1] - pos_shelf2[1]) < height / 2 and pos_red_cube[-1] >= pos_shelf2[-1] and pos_red_cube[
                               -1] <= (pos_shelf2[-1] + depth / 2)
        cube_blue_shelf_2 = abs(pos_blue_cube[0] - pos_shelf2[0]) < width / 2 and abs(
            pos_blue_cube[1] - pos_shelf2[1]) < height / 2 and pos_blue_cube[-1] >= pos_shelf2[-1] and pos_blue_cube[
                                -1] <= (pos_shelf2[-1] + depth / 2)

        r = 0.0

        if cube_green_shelf_1:
            r += 1.0
        if cube_red_shelf_1:
            r += 1.0
        if cube_blue_shelf_1:
            r += 1.0
        if cube_green_shelf_2:
            r += 1.0
        if cube_red_shelf_2:
            r += 1.0
        if cube_blue_shelf_2:
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
        if self.shelf_1 is None:
            self.shelf_1 = Shape(name_or_handle=sim.simGetObjectHandle("shelf1"))
        if self.shelf_2 is None:
            self.shelf_2 = Shape(name_or_handle=sim.simGetObjectHandle("shelf2"))

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

