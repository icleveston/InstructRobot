import random
import math
import numpy as np
from .Environment import Environment
from pyrep.backend import sim
from pyrep.objects.shape import Shape


class CubeSimpleExtEnv(Environment):

    def __init__(self, **kwargs):

        scene_file = 'Main/Scenes/Cubes_Simple_Task_One_Cube.ttt'

        self.cube_blue: Shape | None = None

        num_positions = 50

        x_pos = np.round(np.random.uniform(low=-0.05, high=0.05, size=num_positions), 2)
        y_pos = np.round(np.random.uniform(low=-0.1, high=0.1, size=num_positions), 2)

        self.positions = np.column_stack((x_pos, y_pos))

        np.save('positions.npy', self.positions)

        # Initialize parent class
        super().__init__("CubeExtTouch(GenPosition)", scene_file, **kwargs)


    def configure(self) -> None:
        self._load_objects()
        ind = np.random.randint(len(self.positions))

        pos = np.round(self.positions[ind], 2)

        self.cube_blue.set_position([pos[0], pos[1], 0.5345])

    def reward(self):
        self._load_objects()
        return self._touch_blue_cube()


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
        if self.cube_blue is None:
            self.cube_blue = Shape(name_or_handle=sim.simGetObjectHandle("Cube_Blue"))

    def _get_collisions(self):

        self._load_objects()

        n_collision_blue, _ = self.NAO.check_collisions(self.cube_blue)


        return n_collision_blue


    def _touch_blue_cube(self):

        n_collision_blue = self._get_collisions()

        return n_collision_blue

