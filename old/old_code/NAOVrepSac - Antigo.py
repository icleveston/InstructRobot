from vrep_env import vrep_env
import numpy as np
import gym
import math, random

# -*- coding: utf-8 -*-

scene_path = r"C:\Users\renan-note\Documents\VrepNAOSac\Scenes\CenarioV1.3.6.5.ttt"

class NAOBallEnv(vrep_env.VrepEnv):
    def __init__(self):
        vrep_env.VrepEnv.__init__(self, "127.0.0.1", -1, scene_path)
        self.sim_running = False
        

        joint_limits = {}
        joint_limits["HeadYaw"] = [-119.5, 119.5]
        joint_limits["HeadPitch"] = [-38.5, 29.5]
        joint_limits["LShoulderPitch3"] = [-119.5, 119.5]
        joint_limits["LShoulderRoll3"] = [-18, 76]
        joint_limits["LElbowYaw3"] = [-119.5, 119.5]
        joint_limits["LElbowRoll3"] = [-88.5, -2]
        joint_limits["LWristYaw3"] = [-104.5, 104.5]
        joint_limits["NAO_LThumbBase"] = [0, 60]
        joint_limits["Revolute_joint8"] = [0, 60]
        joint_limits["NAO_LLFingerBase"] = [0, 60]
        joint_limits["Revolute_joint12"] = [0, 60]
        joint_limits["Revolute_joint14"] = [0, 60]
        joint_limits["NAO_LRFingerBase"] = [0, 60]
        joint_limits["Revolute_joint11"] = [0, 60]
        joint_limits["Revolute_joint13"] = [0, 60]
        joint_limits["RShoulderPitch3"] = [-119.5, 119.5]
        joint_limits["RShoulderRoll3"] = [-76, 18]
        joint_limits["RElbowYaw3"] = [-119.5, 119.5]
        joint_limits["RElbowRoll3"] = [2, 88.5]
        joint_limits["RWristYaw3"] = [-104.5, 104.5]
        joint_limits["NAO_RThumbBase"] = [0, 60]
        joint_limits["Revolute_joint0"] = [0, 60]
        joint_limits["NAO_RLFingerBase"] = [0, 60]
        joint_limits["Revolute_joint5"] = [0, 60]
        joint_limits["Revolute_joint6"] = [0, 60]
        joint_limits["NAO_RRFingerBase"] = [0, 60]
        joint_limits["Revolute_joint2"] = [0, 60]
        joint_limits["Revolute_joint3"] = [0, 60]

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

        self.joint_handles = list(map(self.get_object_handle, joint_names))
        self.col_handles = list(map(self.get_collision_handle, col_names))
        self.ball_handle = self.get_object_handle("Sphere3")
        self.kinect_handle = self.get_object_handle("kinect")
        self.ball_z = 0

        self.makeobservation = []
        #dim_obs = len(joint_names) + len(col_names) + 3 # trocar por 
        dim_obs = len(self.make_observation)

        high_obs = np.inf*np.ones([dim_obs])

        self.total_reward = 0
        self.total_episodes = 0

        self.action_space = gym.spaces.Box(self.low_act, self.high_act)
        self.observation_space = gym.spaces.Box(-high_obs, high_obs)

    def make_action(self, actions):
        for joint_handle, action in zip(self.joint_handles, actions):
            self.obj_set_position_target(joint_handle, action)

    def reset(self):
        if self.sim_running:
            self.stop_simulation()

        self.obj_set_position(self.ball_handle, [random.uniform(-0.587, -0.437), random.uniform(0.39, 0.565), 0.40874])

        self.start_simulation()

        print("Last episode " + str(self.total_episodes) + " reward: " + str(self.total_reward))

        self.total_episodes += 1

        self.make_action(random.uniform(self.low_act, self.high_act))

        return self.make_observation()

    def make_observation(self):
        observation  = []
        
        # processmnto de imagem from arquivo import fumcao

        self.ball_z = self.obj_get_position(self.ball_handle, -1)[2]
        
        ball_position = self.obj_get_position(self.ball_handle, relative_to=self.kinect_handle)
        #print(ball_position)
        observation += ball_position

        self.LfingerSensors3 = 0
        self.RfingerSensors3 = 0
        for joint_handle in self.joint_handles:
            observation.append(self.obj_get_joint_angle(joint_handle))
            if joint_handle <= len(self.joint_handles)/2:
                if self.obj_get_joint_angle(joint_handle):
                    self.LfingerSensors3 += 1
            else:
                if self.obj_get_joint_angle(joint_handle):
                    self.RfingerSensors3 += 1

        self.Ltouch = False
        self.Rtouch = False
        if self.LfingerSensors3 >= 3:
            self.Ltouch = True
        if self.RfingerSensors3 >= 3:
            self.Rtouch = True

        observation += (self.Ltouch, self.Rtouch)

        for col_handle in self.col_handles:
            observation.append(self.read_collision(col_handle))
            #print(self.read_collision(col_handle))

        self.makeobservation = observation

        #print("Tamanho da observacao:", len(observation))
        #print(observation)

        return np.asarray(observation).astype('float32')

    def render(self):
        pass

    def step(self, action):
        #print(action)
        self.make_action(action)

        self.step_simulation()

        o = self.make_observation()
        reward = 0

        reward += 10 if self.Ltouch or self.Rtouch else 1 if self.LfingerSensors3 > 0 or self.LfingerSensors3 > 0 else -1

        # force = self.obj_read_force_sensor(handleDoSensor) # retorna vetor de torque e vetor de forca linear
        # se force for none, n ta pronto o dado
        # se force 0

        done = self.ball_z < +2.8608e-1 or self.Ltouch or self.Rtouch# or total 

        self.total_reward += reward

        return o, reward, done, ''


if __name__ == "__main__":
    env = NAOBallEnv()

    while True:
        env.reset()
        for _ in range(200):
            o = env.step(28*[random.uniform(-6.4, 6.4)])