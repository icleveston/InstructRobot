import random
import math
import numpy as np
from .Environment import Environment
from pyrep.backend import sim
from pyrep.objects.shape import Shape
import pyrep

class CubeSimpleExtEnv(Environment):

    def __init__(self, **kwargs):

        scene_file = 'Main/Scenes/Cubes_Simple_Task_One_Cube.ttt'

        self.object: Shape | None = None
        self.object_2: Shape | None = None

        num_positions = 50

        x_pos = np.round(np.random.uniform(low=-0.2, high=0.2, size=num_positions),2)
        y_pos = np.round(np.random.uniform(low=-0.2, high=0.2, size=num_positions),2)

        self.positions = np.column_stack((x_pos, y_pos))

        np.save('positions.npy', self.positions)

        index_sequence = np.tile(np.arange(3), num_positions // 3 + 1)[:num_positions]
        np.random.shuffle(index_sequence)
        self.rgb_colors = np.zeros((num_positions, 3))
        self.rgb_colors[np.arange(num_positions), index_sequence] = 1
        self.rgb_colors = self.rgb_colors.astype(float)

        np.save('rgb_colors.npy', self.rgb_colors)

        self.masses = [0.5, 1.0, 1.5] * num_positions
        np.random.shuffle(self.masses)

        np.save('masses.npy', self.masses)
        # Initialize parent class
        super().__init__("CubeExtTouch(GenPosition_&_Color_&_Mass_&_Form)", scene_file, **kwargs)


    def configure(self) -> None:
        self._load_objects()

        ind = np.random.randint(len(self.positions))
        pos = np.round(self.positions[ind], 2)
        color = list(self.rgb_colors[ind])
        mass = self.masses[ind]
        self.object.set_position([pos[0], pos[1], 0.5345])
        self.object.set_color(color)
        self.object.set_mass(mass)

    def reward(self):
        self._load_objects()
        return self._touch_cube()


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
        if self.object is None:
            self.object = Shape(name_or_handle=sim.simGetObjectHandle("Cube_Blue"))
        self.object_2 = Shape.create(type=pyrep.const.PrimitiveShape.SPHERE, color=[1.0, 0.0, 0.0], size=[0.5, 0.5, 0.5])
        self.object_2.set_position([0.2, 0.2, 0.54])
    def _get_collisions(self):

        self._load_objects()

        n_collision_blue, _ = self.NAO.check_collisions(self.object)


        return n_collision_blue


    def _touch_cube(self):

        n_collision = self._get_collisions()

        return n_collision

