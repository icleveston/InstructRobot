"""
A NAO tries to touch a ball, if it catches with more than 3 fingers, it.
restarts the scene.
This script contains examples of:
    - SAC Environment Setting.
    - Using multiple arms and fingers.
    - Using arms and hands.
"""
from rlkit.core import logger
from pyrep.backend import sim
from pyrep.backend._sim_cffi import lib
from os.path import dirname, join, abspath
from shutil import copy
from pyrep import PyRep
from pyrep.robots.arms.nao_arm import NAOLeftArm, NAORightArm
from pyrep.robots.end_effectors.nao_hand import NAOHand
from pyrep.objects.shape import Shape
import gym
import numpy as np
import math
import csv

POS_MIN, POS_MAX = [-0.9, 0.88, 0.815], [-1.05, 0.99, 0.815]
SCENE_FILE = join(dirname(abspath(__file__)), 'Scenes/scene_NAO_SAC_envV1.9.ttt')

class NAOBallEnv(gym.Env):

    def __init__(self):
        print("Initiating Environment")
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=True,
                        write_coppeliasim_stdout_to_file=False)
        self.init = True
        self.reset()

        # Creating the agent and objects to be controlled and interacted with
        self.NAO = NAOAgent(NAOLeftArm(), NAORightArm(), NAOHand(0), NAOHand(1))
        self.target = Shape('Sphere')
        self.target.scalefactor = 1
        self.head = Shape('NAO_head_link2_visible')

        # Grabbing the logger path directory, to place the observed data fille
        # copying the NAO_SAC.py to this path
        self.log_dir = logger.get_snapshot_dir()
        copy("NAO_SAC.py", self.log_dir)

        # Setting the action and observation space on the gym space box
        dim_obs = len(self.make_observation())
        high_obs = np.inf*np.ones([dim_obs])
        self.action_space = gym.spaces.Box(np.float32(self.NAO.low_act), np.float32(self.NAO.high_act))
        self.observation_space = gym.spaces.Box(np.float32(-high_obs), np.float32(high_obs))

    def make_observation(self):
        """
        Observing every element that we want to take in consideration during 
        the observation training phase.
        """
        observation  = []

        if self.steped:
            observation += list(self.action_last)
        else:
            observation += self.NAO.left.initial_joints_positions
            observation += self.NAO.leftHand.initial_joints_positions
            observation += self.NAO.right.initial_joints_positions
            observation += self.NAO.rightHand.initial_joints_positions

        # image processing from archive import function

        """
        reading the object position to take in consideration for the observation;
        it represents if the objective has been reached, moved,
        or if it need to restart its position due it has fallen.
        """

        self.ball_z = self.target.get_position()[2]
        observation.append(lib.simGetObjectSizeFactor(self.target.get_handle()))
        ball_position = self.target.get_position(relative_to=self.head)
        observation += ball_position.tolist()

        self.NAO.BallFell=  False
        if  self.ball_z < +0.69:
            self.NAO.BallFell =  True

        observation.append(self.NAO.BallFell)

        ball_angle = self.target.get_orientation(relative_to=self.head)
        observation += ball_angle.tolist()

        """
        Read every joint handle in orders to have the joints position angle,
        and train the robot
        """

        if self.steped:
            observation += self.joint_position_last

        self.joint_position_last = []

        for joint_handle in self.NAO.joint_handles:
            observation.append(sim.simGetJointPosition(joint_handle))
            self.joint_position_last.append(sim.simGetJointPosition(joint_handle))

        if not self.steped:
                observation += self.joint_position_last

        self.NAO.LHandSensors3 = 0
        self.NAO.RHandSensors3 = 0

        n_handle = 0
        for col_handle in self.NAO.col_handles:
            observation.append(sim.simReadCollision(col_handle))
            """
            Taking in consideration that the vector col_handle are organized in
            order of the first half collisions handle regards the left manipulator
            and the rest are right, and that the collision handle only has 
            those two possibilities (True or False).
            It is trying to read if either side of the robot's manipulator has
            reached and grabbed the object.
            """
            if n_handle < (len(self.NAO.col_handles)-2)//2:
                if sim.simReadCollision(col_handle):
                    self.NAO.LHandSensors3 += 1
            elif n_handle > len(self.NAO.col_handles)//2:
                if sim.simReadCollision(col_handle):
                    self.NAO.RHandSensors3 += 1
            n_handle += 1

        self.NAO.Ltouch = False
        self.NAO.Rtouch = False
        self.NAO.ObjectCaught = False
        if self.NAO.LHandSensors3 >= 3:
            self.NAO.Ltouch = True
            if sim.simReadCollision(sim.simGetCollisionHandle("CollisionLHand")):
                self.NAO.ObjectCaught = True
        if self.NAO.RHandSensors3 >= 3:
            self.NAO.Rtouch = True
            if sim.simReadCollision(sim.simGetCollisionHandle("CollisionRHand")):
                self.NAO.ObjectCaught = True

        observation.append(self.NAO.Ltouch)
        observation.append(self.NAO.Rtouch)
        observation.append(self.NAO.ObjectCaught)

        self.ball_let_go = False
        self.ball_grabbed = False
        if self.NAO.ObjectCaught:
            self.ball_time_caught += self.pr.get_simulation_timestep()
            if self.ball_time_caught >= 0.5:
                self.ball_grabbed = True
        elif not self.NAO.ObjectCaught and self.ball_time_caught > 0:
            self.ball_time_caught = 0
            self.ball_let_go = True

        observation.append(self.ball_time_caught)
        observation.append(self.ball_let_go)
        observation.append(self.ball_grabbed)

        #self.ball_dropped = False
        #self.NAO.ObjectRaised =  False
        #if  self.ball_z > +0.82 and self.ball_z < +1.4 and self.ball_time_caught > 0:
        #    self.ball_time_raised += self.pr.get_simulation_timestep()
        #    if self.ball_time_raised > 1 and self.ball_grabbed:
        #        self.NAO.ObjectRaised =  True
        #elif self.ball_time_raised > 0 and self.ball_let_go:
        #    self.ball_time_raised = 0
        #    self.ball_dropped = True

        #observation.append(self.ball_time_raised)
        #observation.append(self.NAO.ObjectRaised)
        #observation.append(self.ball_dropped)

        return np.asarray(observation).astype('float32')

    def reset(self):
        #Check if the env is running if it is, stop
        if self.pr.running:
            with open(join(self.log_dir,"reward.csv"), "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(self.last_episode_reward)
            self.pr.stop()

        #Create a detailed contribution's list of the total reward to be filled
        self.last_episode_reward = []

        #Set object control variables
        self.ball_z = 0
        self.ball_time_caught = 0
        self.ball_time_raised = 0

        #Create step and hand prior state control status
        self.steped = False

        #Start the environment for the observation
        self.pr.start()

        if self.init:
            print("Successfully Initiated")
            self.init = False
            return
        else:
            # Get a random position within the object and set the target position
            pos = list(np.random.uniform(POS_MIN, POS_MAX))
            self.target.set_position(pos)

            # Scale objects size to multi-sizes
            lib.simScaleObjects([self.target.get_handle()], 1, self.target.scalefactor, False)
            self.target.scalefactor = np.random.uniform(0.7,  1.1)
            lib.simScaleObjects([self.target.get_handle()], 1, self.target.scalefactor, False)
            self.target.scalefactor = 1/self.target.scalefactor

            # Set NAO's joint to the initial position
            self.NAO.set_initial_joint_positions()

            #Set an initial random position for the joints every reset,
            #in order to increase the different initial states
            self.NAO.make_action(np.random.uniform(self.NAO.low_act, 
                                                   self.NAO.high_act)
                                )

            #Do a step to make all actions in place
            self.pr.step()

            return self.make_observation()

    def step(self, action):
        #function to take actions and step the physics simulation

        self.NAO.make_action(action)

        self.pr.step()

        self.action_last = action
        o = self.make_observation()
        reward = 0
        lastPreward = [0]*12

        #Giving reward for either hand touching the ball with more than 3 fingers
        # 3ftouch, touch, notouch,
        # Reward0, Reward1, Reward2,
        if self.NAO.Ltouch or self.NAO.Rtouch:
            lastPreward[0] = 80
            reward += lastPreward[0]
        #Giving reward for less then 3 fingers from either hand touching the ball
        elif self.NAO.RHandSensors3 > 0 or self.NAO.LHandSensors3 > 0:
            lastPreward[1] = 10
            reward += lastPreward[1]
        #Giving negative reward for not touching the ball
        else:
            lastPreward[2] = - 5
            reward += lastPreward[2]

        # If the ball is caught by the NAO manipulator it gets a reward
        # to be caught is to get with full hand and the palm of either hand
        # ObjectCaught
        # reward3, reward4
        if self.ball_grabbed:
            lastPreward[3] = 4800
            reward += lastPreward[3]
        elif self.NAO.ObjectCaught:
            lastPreward[4] = 800
            reward += lastPreward[4]

        # Gives a negative reward if the ball is considered let go
        # the ball was caught but NAO let it go
        if self.ball_let_go:
            lastPreward[5] = - 400
            reward += lastPreward[5]

        # If the ball is suspended and raised by the NAO manipulator it get an reward
        # ObjectRaised, if not, it will get an reward for try to lift the ball
        # reward7,
        #if self.NAO.ObjectRaised:
        #    lastPreward[6] = 64000
        #    reward += lastPreward[6]
        #elif self.ball_time_raised > 0:
        #    lastPreward[7] = 8000
        #    reward += lastPreward[7]

        # Gives a negative reward if the ball is considered dropped
        # the ball was raised but NAO dropped it 
        #if self.ball_dropped:
        #    lastPreward[8] = - 4000
        #    reward += lastPreward[8]

        # force = self.obj_read_force_sensor(handleDoSensor) # retorna vetor de
        # torque e vetor de forca linear
        # se force for none, n ta pronto o dado
        # se force 0

        # Reward get the negative distance to target
        # Lhanddistance, Rhanddistance,
        # reward8, reward9,
        ((l_ax, l_ay, l_az), (r_ax, r_ay, r_az)) = self.NAO.get_tip_position()
        tx, ty, tz = self.target.get_position()
        lastPreward[9] = -np.sqrt((l_ax - tx)**2 + (l_ay - ty)**2 + (l_az - tz)**2)*50
        reward += lastPreward[9]
        lastPreward[10] = -np.sqrt((r_ax - tx)**2 + (r_ay - ty)**2 + (r_az - tz)**2)*50
        reward += lastPreward[10]

        lastPreward[11] = reward

        #Store individual rewards, to
        self.last_episode_reward.append(lastPreward)
        # [3ftouch, touch, notouch,
        # ObjectCaught,
        # ball_grabbed,
        # ball_let_go,
        # ObjectRaised,
        # Lhanddistance, Rhanddistance,
        # Total Reward ]

        done = self.NAO.BallFell

        self.steped = True

        return o, reward, done, {}

    def render(self):
        pass

    def close(self):
        self.pr.stop()
        self.pr.shutdown()
        print('Success!')

class NAOAgent():
    def __init__(self, NAOleft, NAOright, NAOHandLeft, NAOHandRight):
        self.left = NAOleft
        self.right = NAOright
        self.leftHand = NAOHandLeft
        self.rightHand = NAOHandRight
        self.left.set_control_loop_enabled(False)
        self.right.set_control_loop_enabled(False)
        self.leftHand.set_control_loop_enabled(False)
        self.rightHand.set_control_loop_enabled(False)
        self.left.set_motor_locked_at_zero_velocity(True)
        self.right.set_motor_locked_at_zero_velocity(True)
        self.leftHand.set_motor_locked_at_zero_velocity(True)
        self.rightHand.set_motor_locked_at_zero_velocity(True)
        self.left_tips, self.right_tips = self.get_handTip()
        (self.left.initial_joints_positions,
        self.leftHand.initial_joints_positions,
        self.right.initial_joints_positions,
        self.rightHand.initial_joints_positions
        ) = self.get_joint_positions()

        joint_limits = {}
        # joint_limits["HeadYaw"] = [-119.5, 119.5]
        # joint_limits["HeadPitch"] = [-38.5, 29.5]
        joint_limits["NAO_leftArm_joint1"] = [-119.5, 119.5]
        joint_limits["NAO_leftArm_joint2"] = [-18, 76]
        joint_limits["NAO_leftArm_joint3"] = [-119.5, 119.5]
        joint_limits["NAO_leftArm_joint4"] = [-88.5, -2]
        joint_limits["NAO_leftArm_joint5"] = [-104.5, 104.5]
        joint_limits["NAOHand_thumb1"] = [0, 60]
        joint_limits["NAOHand_thumb2"] = [0, 60]
        joint_limits["NAOHand_leftJoint1"] = [0, 60]
        joint_limits["NAOHand_leftJoint2"] = [0, 60]
        joint_limits["NAOHand_leftJoint3"] = [0, 60]
        joint_limits["NAOHand_rightJoint1"] = [0, 60]
        joint_limits["NAOHand_rightJoint2"] = [0, 60]
        joint_limits["NAOHand_rightJoint3"] = [0, 60]
        joint_limits["NAO_rightArm_joint1"] = [-119.5, 119.5]
        joint_limits["NAO_rightArm_joint2"] = [-76, 18]
        joint_limits["NAO_rightArm_joint3"] = [-119.5, 119.5]
        joint_limits["NAO_rightArm_joint4"] = [2, 88.5]
        joint_limits["NAO_rightArm_joint5"] = [-104.5, 104.5]
        joint_limits["NAOHand_thumb1#0"] = [0, 60]
        joint_limits["NAOHand_thumb2#0"] = [0, 60]
        joint_limits["NAOHand_leftJoint1#0"] = [0, 60]
        joint_limits["NAOHand_leftJoint2#0"] = [0, 60]
        joint_limits["NAOHand_leftJoint3#0"] = [0, 60]
        joint_limits["NAOHand_rightJoint1#0"] = [0, 60]
        joint_limits["NAOHand_rightJoint2#0"] = [0, 60]
        joint_limits["NAOHand_rightJoint3#0"] = [0, 60]

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
                "CollisionLHand",
                "CollisionRHand",
                "CollisionRThumbTip", "CollisionRThumbBase",
                "CollisionRLFingerTip", "CollisionRLFingerMid", "CollisionRLFingerBase",
                "CollisionRRFingerTip", "CollisionRRFingerMid", "CollisionRRFingerBase"
                ]

        self.joint_handles = list(map(sim.simGetObjectHandle, joint_names))
        self.col_handles = list(map(sim.simGetCollisionHandle, col_names))

    def get_joint_positions(self):
        return (self.left.get_joint_positions(),
                self.leftHand.get_joint_positions(),
                self.right.get_joint_positions(),
                self.rightHand.get_joint_positions())

    def get_handTip(self):
        return (self.left.get_tip(), 
                self.right.get_tip())

    def get_tip_position(self):
    	return (self.left_tips.get_position(), 
                self.right_tips.get_position())

    def make_action(self, actions):
        for joint_handle, action in zip(self.joint_handles, actions):
            sim.simSetJointTargetPosition(joint_handle, action)

    def set_joint_positions(self, leftPositions, leftHandPositions,
                            rightPositions, rightHandPositions):
        self.left.set_joint_positions(leftPositions)
        self.leftHand.set_joint_positions(leftHandPositions)
        self.right.set_joint_positions(rightPositions)
        self.rightHand.set_joint_positions(rightHandPositions)

    def set_initial_joint_positions(self):
        self.set_joint_positions(self.left.initial_joints_positions,
                                 self.leftHand.initial_joints_positions,
                                 self.right.initial_joints_positions,
                                 self.rightHand.initial_joints_positions)

if __name__ == "__main__":
    env = NAOBallEnv()
    nao = NAOAgent(NAOLeftArm(), NAORightArm(), NAOHand(0), NAOHand(1))
    on = True

    while on:
        env.reset()
        for _ in range(1000):
            o = env.step(np.random.uniform(nao.low_act, nao.high_act))
            if o[2]:
                break
        if _ == 1000:
            on = False

    print('Done!')
    env.close()
