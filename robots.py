from pyrep.backend import sim
import numpy as np
import math



class NAOAgent():
    def __init__(self, NAOleftArm, NAOrightArm, NAOHandLeft, NAOHandRight):
        self.leftArm = NAOleftArm
        self.rightArm = NAOrightArm
        self.leftHand = NAOHandLeft
        self.rightHand = NAOHandRight
        self.leftArm.set_control_loop_enabled(False)
        self.rightArm.set_control_loop_enabled(False)
        self.leftHand.set_control_loop_enabled(False)
        self.rightHand.set_control_loop_enabled(False)
        self.leftArm.set_motor_locked_at_zero_velocity(True)
        self.rightArm.set_motor_locked_at_zero_velocity(True)
        self.leftHand.set_motor_locked_at_zero_velocity(True)
        self.rightHand.set_motor_locked_at_zero_velocity(True)
        self.left_tips, self.right_tips = self.get_handTip()
        (self.leftArm.initial_joints_positions,
        self.leftHand.initial_joints_positions,
        self.rightArm.initial_joints_positions,
        self.rightHand.initial_joints_positions
        ) = self.get_joint_positions()


        joint_limits = {}

        #joint_limits["NAO_head_joint1"] = [-119.5, 119.5]
        #joint_limits["NAO_head_joint2"] = [-38.5, 29.5]
        #arm right

        joint_limits["NAO_rightArm_joint1"] = [-119.5, 119.5]
        joint_limits["NAO_rightArm_joint2"] = [-76, 18]
        joint_limits["NAO_rightArm_joint3"] = [-119.5, 119.5]
        joint_limits["NAO_rightArm_joint4"] = [2, 88.5]
        joint_limits["NAO_rightArm_joint5"] = [-104.5, 104.5]
        #hand right
        joint_limits["NAOHand_thumb1#0"] = [0, 60]
        joint_limits["NAOHand_thumb2#0"] = [0, 60]
        joint_limits["NAOHand_rightJoint1#0"] = [0, 60]
        joint_limits["NAOHand_rightJoint2#0"] = [0, 60]
        joint_limits["NAOHand_rightJoint3#0"] = [0, 60]
        joint_limits["NAOHand_leftJoint1#0"] = [0, 60]
        joint_limits["NAOHand_leftJoint2#0"] = [0, 60]
        joint_limits["NAOHand_leftJoint3#0"] = [0, 60]
        #arm left
        joint_limits["NAO_leftArm_joint1"] = [-119.5, 119.5]
        joint_limits["NAO_leftArm_joint2"] = [-18, 76]
        joint_limits["NAO_leftArm_joint3"] = [-119.5, 119.5]
        joint_limits["NAO_leftArm_joint4"] = [-88.5, -2]
        joint_limits["NAO_leftArm_joint5"] = [-104.5, 104.5]
        #hand left
        joint_limits["NAOHand_thumb1"] = [0, 60]
        joint_limits["NAOHand_thumb2"] = [0, 60]
        joint_limits["NAOHand_rightJoint1"] = [0, 60]
        joint_limits["NAOHand_rightJoint2"] = [0, 60]
        joint_limits["NAOHand_rightJoint3"] = [0, 60]
        joint_limits["NAOHand_leftJoint1"] = [0, 60]
        joint_limits["NAOHand_leftJoint2"] = [0, 60]
        joint_limits["NAOHand_leftJoint3"] = [0, 60]

        joint_names = joint_limits.keys()
        print(joint_names)

        low_act = []
        high_act = []
        for joint in joint_names:
            low_act.append(joint_limits[joint][0])
            high_act.append(joint_limits[joint][1])
        #converte para radianos
        self.low_act = np.array(low_act)* math.pi/180
        self.high_act = np.array(high_act)* math.pi/180


        ''' 
        col_names = [
                "CollisionLThumbTip", "CollisionLThumbBase",
                "CollisionLLFingerTip", "CollisionLLFingerMid", "CollisionLLFingerBase",
                "CollisionLRFingerTip", "CollisionLRFingerMid", "CollisionLRFingerBase",
                "CollisionLHand",
                "CollisionRHand",
                "CollisionRThumbTip", "CollisionRThumbBase",
                "CollisionRLFingerTip", "CollisionRLFingerMid", "CollisionRLFingerBase",
                "CollisionRRFingerTip", "CollisionRRFingerMid", "CollisionRRFingerBase"
                ]

        self.col_fingers = ["CollisionLThumbTip",
        "CollisionLLFingerTip",
        "CollisionLRFingerTip",
        "CollisionRThumbTip",
        "CollisionRLFingerTip",
        "CollisionRRFingerTip"]'''

        col_names = [
            "CollisionLThumbTip", "CollisionLThumbBase",
            "CollisionLLFingerTip", "CollisionLLFingerMid", "CollisionLLFingerBase",
            "CollisionLRFingerTip", "CollisionLRFingerMid", "CollisionLRFingerBase",
            "CollisionLHand"]

        self.col_fingers = ["CollisionLThumbTip",
                            "CollisionLLFingerTip",
                            "CollisionLRFingerTip"]



        self.joint_handles = list(map(sim.simGetObjectHandle, joint_names))
        self.col_handles = list(map(sim.simGetCollisionHandle, col_names))

    def get_low_act(self):
        return self.low_act

    def get_high_act(self):
        return self.high_act

    def get_joint_positions(self):
        return (self.leftArm.get_joint_positions(),
                self.leftHand.get_joint_positions(),
                self.rightArm.get_joint_positions(),
                self.rightHand.get_joint_positions())

    def get_handTip(self):
        return (self.leftArm.get_tip(),
                self.rightArm.get_tip())

    def get_tip_position(self):
    	return (self.left_tips.get_position(),
                self.right_tips.get_position())

    def make_action(self, actions):
        for joint_handle, action in zip(self.joint_handles, actions):
            sim.simSetJointTargetPosition(joint_handle, action)

    def set_joint_positions(self, leftPositions, leftHandPositions,
                            rightPositions, rightHandPositions):
        self.leftArm.set_joint_positions(leftPositions)
        self.leftHand.set_joint_positions(leftHandPositions)
        self.rightArm.set_joint_positions(rightPositions)
        self.rightHand.set_joint_positions(rightHandPositions)

    def set_initial_joint_positions(self):
        self.set_joint_positions(self.leftArm.initial_joints_positions,
                                 self.leftHand.initial_joints_positions,
                                 self.rightArm.initial_joints_positions,
                                 self.rightHand.initial_joints_positions)
