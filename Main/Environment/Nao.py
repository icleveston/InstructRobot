import math
import time
from typing import List
import numpy as np
from pyrep.objects.joint import Joint
from pyrep.objects.shape import Shape, PrimitiveShape


def vectorized_to_interval(limits: np.array, actions: np.array) -> np.array:
    """
    Converts a vector of actions in the range (-1, 1) to the range given by
    limits array

    Args:
        limits: the limits for which the actions will be scaled
        actions: the action in the range (-1, 1)
    """
    # Actions ranges in (-1, 1)
    min_, max_ = -1, 1
    a, b = limits[:, 0], limits[:, 1]

    return ((b - a) * (actions - min_) / (max_ - min_)) + a


class Nao:
    def __init__(self):

        _joints_info_deg = {
            "/NAO/LShoulderPitch": (-1.195e+02, 2.390e+02),
            "/NAO/LShoulderRoll": (-1.800e+01, 9.401e+01),
            "/NAO/LElbowYaw": (-1.195e+02, 2.390e+02),
            "/NAO/LElbowRoll": (-8.850e+01, 8.650e+01),
            "/NAO/LWristYaw": (-1.045e+02, 2.090e+02),
            "/NAO/LThumbBase": (+0.000e+00, 6.000e+01),
            "/NAO/LShoulderPitch/joint": (+0.000e+00, 6.000e+01),
            "/NAO/LRFingerBase": (+0.000e+00, 6.000e+01),
            "/NAO/LRFingerBase/joint": (+0.000e+00, 6.000e+01),
            "/NAO/LShoulderPitch/joint/Cuboid/joint": (+0.000e+00, 6.000e+01),
            "/NAO/LLFingerBase": (+0.000e+00, 6.000e+01),
            "/NAO/LLFingerBase/joint": (+0.000e+00, 6.000e+01),
            "/NAO/LLFingerBase/joint/Cuboid/joint": (+0.000e+00, 6.000e+01),
            "/NAO/RShoulderPitch": (-1.195e+02, 2.390e+02),
            "/NAO/RShoulderRoll": (-7.600e+01, 9.401e+01),
            "/NAO/RElbowYaw": (-1.195e+02, 2.390e+02),
            "/NAO/RElbowRoll": (+2.000e+00, 8.650e+01),
            "/NAO/RWristYaw": (-1.045e+02, 2.090e+02),
            "/NAO/RThumbBase": (+0.000e+00, 6.000e+01),
            "/NAO/joint": (+0.000e+00, 6.000e+01),
            "/NAO/RRFingerBase": (+0.000e+00, 6.000e+01),
            "/NAO/RRFingerBase/joint": (+0.000e+00, 6.000e+01),
            "/NAO/joint/Cuboid/joint": (+0.000e+00, 6.000e+01),
            "/NAO/RLFingerBase": (+0.000e+00, 6.000e+01),
            "/NAO/RLFingerBase/joint": (+0.000e+00, 6.000e+01),
            "/NAO/RLFingerBase/joint/Cuboid/joint": (+0.000e+00, 6.000e+01)
        }

        _joints_info_rad = {
            name: (np.radians(limits[0]), np.radians(limits[1])) for name, limits in _joints_info_deg.items()
        }

        _fingers_names = [
            "Cuboid0",
            "Cuboid15",
            "Cuboid14",
            "Cuboid4",
            "Cuboid2",
            "Cuboid12",
            "Cuboid10",
            "Cuboid11",
            "Cuboid6",
            "Cuboid13",
            "Cuboid9",
            "Cuboid7",
            "Cuboid8",
            "Cuboid",
            "Cuboid3",
            "Cuboid1",
        ]

        self._all_joints: List[Joint] = []
        self._all_fingers: List[Shape] = []
        self._joint_limits: np.array = []
        self._head = Shape(name_or_handle='HeadPitch_link_respondable')
        self._chest = Shape(name_or_handle='imported_part_20_sub0')

        joint_limits = []
        for name, limits in _joints_info_rad.items():
            joint = Joint(name_or_handle=name)
            joint.set_control_loop_enabled(True)
            joint.set_motor_locked_at_zero_velocity(True)
            self._all_joints.append(joint)

            joint_limits.append((limits[0], limits[1] + limits[0]))

        self._joint_limits = np.array(joint_limits)

        for finger_name in _fingers_names:
            shape = Shape(name_or_handle=finger_name)
            self._all_fingers.append(shape)

    def get_joint_positions(self) -> np.array:

        joint_positions = [
            joint.get_joint_position() for joint in self._all_joints
        ]

        return np.array(joint_positions)

    def make_action(self, action: [], show_denormalization: bool = False) -> None:

        action = np.array(action)
        action_tanh = np.tanh(action)
        action_denormalized = vectorized_to_interval(self._joint_limits, action_tanh)

        if show_denormalization:
            print(f"Actions from Agent: {action}")
            print(f"Actions tanh: {action_tanh}")
            print(f"Actions action_denormalized: {action_denormalized}")

        for target_position, joint in zip(action_denormalized, self._all_joints):
            joint.set_joint_target_position(target_position)

    def check_collisions(self, object_shape: Shape) -> ():

        collision_array = {}

        for finger in self._all_fingers:
            collision_array[finger.get_name()] = int(finger.check_collision(object_shape))

        return sum(collision_array.values()), collision_array

    def validate_joints(self, joint_position_step=0.005):

        # Test each joint independently
        for i, joint_limits in enumerate(self._joint_limits):

            is_new_joint = True

            # Increase the joint position
            joint_values = np.arange(joint_limits[0], joint_limits[1], joint_position_step)

            for joint_value in joint_values:
                action = np.zeros(26)
                action[i] = joint_value

                yield is_new_joint, action

                is_new_joint = False

    def validate_collisions(self,
                            joint_shoulder_left_id=0,
                            joint_elbow_left_id=2,
                            joint_wrist_left_id=4,
                            joint_shoulder_right_id=13,
                            joint_elbow_right_id=15,
                            joint_wrist_right_id=17,
                            hand_left_id=0,
                            joint_position_step=0.05):

        # Get wrist and elbow joints
        joints = [
            (joint_shoulder_left_id, self._joint_limits[joint_shoulder_left_id][0]/10),
            (joint_elbow_left_id, self._joint_limits[joint_elbow_left_id][0]),
            (joint_wrist_left_id, self._joint_limits[joint_wrist_left_id][0]/1.8),
            (joint_shoulder_right_id, self._joint_limits[joint_shoulder_right_id][0]/10),
            (joint_elbow_right_id, self._joint_limits[joint_elbow_right_id][1]),
            (joint_wrist_right_id, self._joint_limits[joint_wrist_right_id][1]/1.8)
        ]

        action = np.zeros(26)

        # Move joints to desired position
        for i, limit in joints:

            for _ in range(20):
                action[i] = limit
                yield action

        # Get hands position
        hands = [
            (hand_left_id, self._joint_limits[hand_left_id][0], [-0.05, 0, 0.1], [math.pi, math.pi, 0]),
        ]

        for i, hand, position_shift, orientation_shift in hands:

            position = self._all_fingers[i].get_position()
            position[0] += position_shift[0]
            position[1] = 0
            position[2] += position_shift[2]

            # Create test cube
            cube: Shape = Shape.create(
                PrimitiveShape.CUBOID,
                mass=0.3,
                size=[0.1, 1, 0.003],
                position=position,
                orientation=orientation_shift
            )
            
            # Set cube properties
            cube.set_collidable(True)
            cube.set_measurable(True)
            cube.set_detectable(True)
            cube.set_respondable(True)

            print(f"IsDynamic: {cube.is_dynamic()} -"
                  f" IsRespondable: {cube.is_respondable()} -"
                  f" IsCollidable: {cube.is_collidable()} -"
                  f" IsMesurable: {cube.is_measurable()} -"
                  f" IsDetectable: {cube.is_detectable()}")

        fingers = zip(reversed(range(5,13)), reversed(self._joint_limits[5:13]))

        # Close hands
        for i, limits in fingers:

            # Increase the joint position
            joint_values = np.arange(limits[0], limits[1], joint_position_step)

            for joint_value in joint_values:
                action[i] = joint_value
                print(f"Detected Collisions: {self.check_collisions(cube)[0]}")
                yield action

        fingers = zip(reversed(range(18, 26)), reversed(self._joint_limits[18:26]))

        # Close hands
        for i, limits in fingers:

            # Increase the joint position
            joint_values = np.arange(limits[0], limits[1], joint_position_step)

            for joint_value in joint_values:
                action[i] = joint_value
                print(f"Detected Collisions: {self.check_collisions(cube)[0]}")
                yield action

        for _ in range(100):
            yield action
            print(f"Detected Collisions: {self.check_collisions(cube)[0]}")

        time.sleep(120)

