from pyrep.backend import sim
import numpy as np
import math


class Nao:
    def __init__(self, left_arm, right_arm, hand_left, hand_right):
        self.leftArm = left_arm
        self.rightArm = right_arm
        self.leftHand = hand_left
        self.rightHand = hand_right
        self.leftArm.set_control_loop_enabled(False)
        self.rightArm.set_control_loop_enabled(False)
        self.leftHand.set_control_loop_enabled(False)
        self.rightHand.set_control_loop_enabled(False)
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
            "NAO_head_joint1": [-119.5, 119.5],
            "NAO_head_joint2": [-38.5, 29.5],
            "NAO_rightArm_joint1": [-119.5, 119.5],
            "NAO_rightArm_joint2": [-76, 18],
            "NAO_rightArm_joint3": [-119.5, 119.5],
            "NAO_rightArm_joint4": [2, 88.5],
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
            "NAO_leftArm_joint2": [-18, 76],
            "NAO_leftArm_joint3": [-119.5, 119.5],
            "NAO_leftArm_joint4": [-88.5, -2],
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

        low_act = []
        high_act = []
        for joint in joint_names:
            low_act.append(joint_limits[joint][0])
            high_act.append(joint_limits[joint][1])

        self.low_act = np.array(low_act) * math.pi / 180
        self.high_act = np.array(high_act) * math.pi / 180

        fingers_names = [
            "NAOHand_thumb2_visible",
            "NAOHand_rightJoint3_visible",
            "NAOHand_leftJoint3_visible",
            "NAOHand_thumb2_visible#0",
            "NAOHand_rightJoint3_visible#0",
            "NAOHand_leftJoint3_visible#0"
        ]

        self._fingers_handles = [sim.simGetObjectHandle(i) for i in fingers_names]

        self._joint_handles = list(map(sim.simGetObjectHandle, joint_names))

    def get_low_act(self):
        return self.low_act

    def get_high_act(self):
        return self.high_act

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

    def check_collisions(self, object_handle) -> bool:

        for joint in self._fingers_handles:
            if sim.simCheckCollision(joint, object_handle):
                return True

        return False
