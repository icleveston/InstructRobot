"""
A baxter picks up a cup with its left arm and then passes it to its right arm.
This script contains examples of:
    - Path planning (linear and non-linear).
    - Using multiple arms.
    - Using a gripper.
"""
from rlkit.core import logger
from pyrep.backend import vrep
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.nao import NAOLeft, NAORight
# from pyrep.robots.end_effectors.nao_hand import NAOHand_leftArm, NAOHand_rightArm
from pyrep.objects.shape import Shape
import gym
import numpy as np
import math
import csv

POS_MIN, POS_MAX = [-0.587, 0.39, 0.40874], [-0.437, 0.565, 0.40874]

# SCENE_FILE = join('/home/nanbaima/PyRep/examples', 'scene_NAO_SAC_env.ttt')
SCENE_FILE = join(dirname(abspath(__file__)), 'Scenes/scene_NAO_SAC_env.ttt')

class NAOBallEnv():
    def __init__(self):
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=False)
        self.pr.start()
        self.NAO = NAOAgent(NAOLeft(), NAORight())
        self.target = Shape('Sphere')
        self.kinect = Shape('kinect')

        self.ball_z = 0
        self.steped = False

        dim_obs = len(self.make_observation())

        high_obs = np.inf*np.ones([dim_obs])

        self.total_reward = 0
        self.total_episodes = 0
        self.last_episode_reward = []

        self.action_space = gym.spaces.Box(self.NAO.low_act, self.NAO.high_act)
        self.observation_space = gym.spaces.Box(-high_obs, high_obs)

    def make_observation(self):
        """
        Observing every element that we want to take in consideration during 
        the observation traning phase.
        """
        observation  = []

        if self.steped:
            observation += list(self.action_last)
        else:
            observation += self.NAO.left.initial_joints_positions
            observation += self.NAO.right.initial_joints_positions

        # image processing from arquivo import function

        """
        reading the balls position to take in considaration for the observation, it will represent's
        if the objective has been reached, moving the ball, or restart if the ball has fallen
        """

        self.ball_z = self.target.get_position()[2]
        ball_position = self.target.get_position(relative_to=self.kinect)
        observation += ball_position

        ball_angle = self.target.get_orientation(relative_to=self.kinect)
        observation += ball_angle

        """
        Read every joint handle in orders to have the joints position angle, and train the robot
        """

        if self.steped:
            observation += self.joint_position_last

        self.joint_position_last = []

        for joint_handle in self.NAO.joint_handles:
            observation.append(vrep.simGetJointPosition(joint_handle))
            self.joint_position_last.append(vrep.simGetJointPosition(joint_handle))
            if not self.steped:
                observation.append(vrep.simGetJointPosition(joint_handle))

        self.NAO.LfingerSensors3 = 0
        self.NAO.RfingerSensors3 = 0
        n_handle = 0
        for col_handle in self.NAO.col_handles:
            observation.append(vrep.simReadCollision(col_handle))
            #print(self.read_collision(col_handle))
            """
            Taking in consideration that the vector col_handle are organazed in order of the first
            half colisions handle regards the left manipulator and the rest are right, and that the
            colision hanle only has those two possibilities.
            It is trying to read if eather side of the robot's manipulator has reached and grabed 
            the object.
            """
            if n_handle < len(self.NAO.col_handles)//2:
                if vrep.simReadCollision(col_handle):
                    self.NAO.LfingerSensors3 += 1
            else:
                if vrep.simReadCollision(col_handle):
                    self.NAO.RfingerSensors3 += 1
            n_handle += 1

        self.NAO.Ltouch = False
        self.NAO.Rtouch = False
        if self.NAO.LfingerSensors3 >= 3:
            self.NAO.Ltouch = True
        if self.NAO.RfingerSensors3 >= 3:
            self.NAO.Rtouch = True

        observation.append(self.NAO.Ltouch)
        observation.append(self.NAO.Rtouch)

        return np.asarray(observation).astype('float32')

    def reset(self):
        # Get a random position within the sphere and set the target position

        print("Last episode n " + str(self.total_episodes) + 
              " reward: " + str(self.total_reward))

        with open(join(logger.get_snapshot_dir(),"reward.csv"), "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.last_episode_reward)

        self.last_episode_reward = []

        self.pr.stop()

        pos = list(np.random.uniform(POS_MIN, POS_MAX))
        self.target.set_position(pos)
        self.NAO.set_initial_joint_positions()
        self.steped = False

        self.pr.start()

        self.total_reward = 0
        self.total_episodes += 1

        self.NAO.make_action(np.random.uniform(self.NAO.low_act, self.NAO.high_act))

        return self.make_observation()

    def step(self, action):
        #function to take actions and step the physics simulation

        self.NAO.make_action(action)

        self.pr.step()

        self.action_last = action
        o = self.make_observation()
        reward = 0
        lastPreward1 = 0
        lastPreward2 = 0
        lastPreward3 = 0

        if self.NAO.Ltouch or self.NAO.Rtouch:
#        if self.NAO.Rtouch:
            reward += 1000
            lastPreward1 = 1000
        elif self.NAO.RfingerSensors3 > 0 or self.NAO.LfingerSensors3 > 0:
#        elif self.NAO.RfingerSensors3 > 0:
            reward += 10
            lastPreward2 = 10
        else:
            reward -= 1
            lastPreward3 = -1

        # Reward get the negative distance to target
        ((l_ax, l_ay, l_az), (r_ax, r_ay, r_az)) = self.NAO.get_tip_position()
        tx, ty, tz = self.target.get_position()
        reward += -np.sqrt((l_ax - tx) ** 2 + (l_ay - ty) ** 2 + (l_az - tz) ** 2)
        reward += -np.sqrt((r_ax - tx) ** 2 + (r_ay - ty) ** 2 + (r_az - tz) ** 2)

        # force = self.obj_read_force_sensor(handleDoSensor) # retorna vetor de
        #torque e vetor de forca linear
        # se force for none, n ta pronto o dado
        # se force 0

        done = (self.ball_z < +2.8608e-1) or self.NAO.Rtouch or self.NAO.Ltouch

        self.total_reward += reward
        self.steped = True

        self.last_episode_reward.append([lastPreward1, lastPreward2, lastPreward3, 
                                  -np.sqrt((l_ax - tx) ** 2 + (l_ay - ty) ** 2 + (l_az - tz) ** 2),
                                  -np.sqrt((r_ax - tx) ** 2 + (r_ay - ty) ** 2 + (r_az - tz) ** 2)]
            )#[3ftouch, touch, notouch, Lhanddistance, Rhanddistance ]

        return o, reward, done, ''

    def render(self):
        pass

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

class NAOAgent():
    
    def __init__(self, NAOleft, NAOright):
        self.left = NAOleft
        self.right = NAOright
        self.left.set_control_loop_enabled(False)
        self.right.set_control_loop_enabled(False)
        self.left.set_motor_locked_at_zero_velocity(True)
        self.right.set_motor_locked_at_zero_velocity(True)
        self.left_tips, self.right_tips = self.get_handTip()
        self.left.initial_joints_positions, self.right.initial_joints_positions = self.get_joint_positions()
        
        joint_limits = {}
#        joint_limits["HeadYaw"] = [-119.5, 119.5]
#        joint_limits["HeadPitch"] = [-38.5, 29.5]
        joint_limits["NAO_leftArm_joint1"] = [-119.5, 119.5]
        joint_limits["NAO_leftArm_joint2"] = [-18, 76]
        joint_limits["NAO_leftArm_joint3"] = [-119.5, 119.5]
        joint_limits["NAO_leftArm_joint4"] = [-88.5, -2]
        joint_limits["NAO_leftArm_joint5"] = [-104.5, 104.5]
#        joint_limits["NAO_LThumbBase"] = [0, 60]
#        joint_limits["Revolute_joint8"] = [0, 60]
#        joint_limits["NAO_LLFingerBase"] = [0, 60]
#        joint_limits["Revolute_joint12"] = [0, 60]
#        joint_limits["Revolute_joint14"] = [0, 60]
#        joint_limits["NAO_LRFingerBase"] = [0, 60]
#        joint_limits["Revolute_joint11"] = [0, 60]
#        joint_limits["Revolute_joint13"] = [0, 60]
        joint_limits["NAO_rightArm_joint1"] = [-119.5, 119.5]
        joint_limits["NAO_rightArm_joint2"] = [-76, 18]
        joint_limits["NAO_rightArm_joint3"] = [-119.5, 119.5]
        joint_limits["NAO_rightArm_joint4"] = [2, 88.5]
        joint_limits["NAO_rightArm_joint5"] = [-104.5, 104.5]
#        joint_limits["NAO_RThumbBase"] = [0, 60]
#        joint_limits["Revolute_joint0"] = [0, 60]
#        joint_limits["NAO_RLFingerBase"] = [0, 60]
#        joint_limits["Revolute_joint5"] = [0, 60]
#        joint_limits["Revolute_joint6"] = [0, 60]
#        joint_limits["NAO_RRFingerBase"] = [0, 60]
#        joint_limits["Revolute_joint2"] = [0, 60]
#        joint_limits["Revolute_joint3"] = [0, 60]

        joint_names = joint_limits.keys()

        low_act = []
        high_act = []
        for joint in joint_names:
            low_act.append(joint_limits[joint][0])
            high_act.append(joint_limits[joint][1])

        self.low_act = np.array(low_act)* math.pi/180
        self.high_act = np.array(high_act)* math.pi/180

        col_names = [
                "CollisionLThumbTip", "CollisionLThumbBase",
                "CollisionLLFingerTip", "CollisionLLFingerMid", "CollisionLLFingerBase",
                "CollisionLRFingerTip", "CollisionLRFingerMid", "CollisionLRFingerBase",
                "CollisionRThumbTip", "CollisionRThumbBase",
                "CollisionRLFingerTip", "CollisionRLFingerMid", "CollisionRLFingerBase",
                "CollisionRRFingerTip", "CollisionRRFingerMid", "CollisionRRFingerBase"
                ]

        self.joint_handles = list(map(vrep.simGetObjectHandle, joint_names))
        self.col_handles = list(map(vrep.simGetCollisionHandle, col_names))

    def get_joint_positions(self):
        return (self.left.get_joint_positions(), 
                self.right.get_joint_positions())

    def get_handTip(self):
        return (self.left.get_tip(), 
                self.right.get_tip())

    def get_tip_position(self):
    	return (self.left_tips.get_position(), 
                self.right_tips.get_position())

    def make_action(self, actions):
        for joint_handle, action in zip(self.joint_handles, actions):
            vrep.simSetJointTargetPosition(joint_handle, action)

    def set_joint_positions(self, leftPositions, rightPositions):
        self.left.set_joint_positions(leftPositions)
        self.right.set_joint_positions(rightPositions)

    def set_initial_joint_positions(self):
        self.left.set_joint_positions(self.left.initial_joints_positions)
        self.right.set_joint_positions(self.right.initial_joints_positions)

if __name__ == "__main__":
    env = NAOBallEnv()
    nao = NAOAgent(NAOLeft(), NAORight())
    on = True

    while on:
        env.reset()
        for _ in range(1000):
            o = env.step(np.random.uniform(nao.low_act, nao.high_act))
            if o[3]:
            	break
        if _ == 1000:
            on = False

    print('Done!')
    env.shutdown()