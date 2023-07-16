from pyrep.backend import sim
from pyrep.robots.arms.nao_arm import NAOLeftArm, NAORightArm
from pyrep.robots.end_effectors.nao_hand import NAOHand
import numpy as np
import math


def normalize(x, old_range, new_range):

    norm = (x - old_range[0]) / (old_range[1] - old_range[0])

    return norm * (new_range[1] - new_range[0]) + new_range[0]


class Nao:
    def __init__(self):

        self.leftArm = NAOLeftArm()
        self.rightArm = NAORightArm()
        self.leftHand = NAOHand()
        self.rightHand = NAOHand()

        self.leftArm.set_motor_locked_at_zero_velocity(True)
        self.rightArm.set_motor_locked_at_zero_velocity(True)
        self.leftHand.set_motor_locked_at_zero_velocity(True)
        self.rightHand.set_motor_locked_at_zero_velocity(True)

        self.left_tips, self.right_tips = self.get_hand_tip()
        (
            self.leftArm.initial_joints_positions,
            self.leftHand.initial_joints_positions,
            self.rightArm.initial_joints_positions,
            self.rightHand.initial_joints_positions
        ) = self.get_joint_positions()

        joint_limits = {
            "NAO_rightArm_joint1": [-119.5, 119.5],
            "NAO_rightArm_joint2": [-76, 60],
            "NAO_rightArm_joint3": [-119.5, 119.5],
            "NAO_rightArm_joint4": [2, 120],
            "NAO_rightArm_joint5": [-104.5, 104.5],
            "NAOHand_thumb1#0": [0, 60],
            "NAOHand_thumb2#0": [0, 60],
            "NAOHand_rightJoint1#0": [0, 60],
            "NAOHand_rightJoint2#0": [0, 60],
            "NAOHand_rightJoint3#0": [0, 60],
            "NAOHand_leftJoint1#0": [0, 60],
            "NAOHand_leftJoint2#0": [0, 60],
            "NAOHand_leftJoint3#0": [0, 60],
            "NAO_leftArm_joint1": [-119.5, 119.5],
            "NAO_leftArm_joint2": [-60, 76],
            "NAO_leftArm_joint3": [-119.5, 119.5],
            "NAO_leftArm_joint4": [-120, -2],
            "NAO_leftArm_joint5": [-104.5, 104.5],
            "NAOHand_thumb1": [0, 60],
            "NAOHand_thumb2": [0, 60],
            "NAOHand_rightJoint1": [0, 60],
            "NAOHand_rightJoint2": [0, 60],
            "NAOHand_rightJoint3": [0, 60],
            "NAOHand_leftJoint1": [0, 60],
            "NAOHand_leftJoint2": [0, 60],
            "NAOHand_leftJoint3": [0, 60]
        }

        joint_names = joint_limits.keys()

        self._joint_handles = list(map(sim.simGetObjectHandle, joint_names))

        low_act = []
        high_act = []
        for joint in joint_names:
            low_act.append(joint_limits[joint][0])
            high_act.append(joint_limits[joint][1])

        low_desired_act = np.array(low_act) * math.pi / 180
        high_desired_act = np.array(high_act) * math.pi / 180

        low_actual_act = np.full((26,), -1)
        high_atual_act = np.full((26,), 1)

        self.old_range = (low_actual_act, high_atual_act)
        self.new_range = (low_desired_act, high_desired_act)

        fingers_names = [
            "/NAOHandRight_Finger1",
            "/NAOHandRight_Finger2",
            "/NAOHandRight_Finger3",
            "/NAOHandLeft_Finger1",
            "/NAOHandLeft_Finger2",
            "/NAOHandLeft_Finger3"
        ]

        self._fingers_handles = [sim.simGetObjectHandle(i) for i in fingers_names]

        touch_names = [
            "NAOHandRight_Touch1",
            "NAOHandRight_Touch2",
            "NAOHandRight_Touch3",
            "NAOHandLeft_Touch1",
            "NAOHandLeft_Touch2",
            "NAOHandLeft_Touch3"
        ]

        self._touch_handles = [sim.simGetObjectHandle(i) for i in touch_names]

    def get_joint_positions(self):
        return self.leftArm.get_joint_positions(), self.leftHand.get_joint_positions(), \
            self.rightArm.get_joint_positions(), self.rightHand.get_joint_positions()

    def get_hand_tip(self):
        return self.leftArm.get_tip(), self.rightArm.get_tip()

    def get_tip_position(self):
        return self.left_tips.get_position(), self.right_tips.get_position()

    def make_action(self, actions):

        for joint_handle, action in zip(self._joint_handles, actions):
            sim.simSetJointTargetPosition(joint_handle, action)

    def set_joint_positions(self, left_positions, left_hand_positions, right_positions, right_hand_positions):
        self.leftArm.set_joint_positions(left_positions)
        self.leftHand.set_joint_positions(left_hand_positions)
        self.rightArm.set_joint_positions(right_positions)
        self.rightHand.set_joint_positions(right_hand_positions)

    def set_initial_joint_positions(self):
        self.set_joint_positions(self.leftArm.initial_joints_positions,
                                 self.leftHand.initial_joints_positions,
                                 self.rightArm.initial_joints_positions,
                                 self.rightHand.initial_joints_positions)

    def check_touch_sensors(self):

        forceVector = []

        for touch_handle in self._touch_handles:
            _, force, _ = sim.simReadForceSensor(touch_handle)

            forceVector += force

        print(sum(forceVector))

    def check_collisions(self, object_handle) -> bool:

        for joint in self._fingers_handles:
            if sim.simCheckCollision(joint, object_handle):
                return True

        return False
