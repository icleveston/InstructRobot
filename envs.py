from pyrep.backend import sim
from pyrep.backend._sim_cffi import lib
from pyrep import PyRep
import gym
import math
import numpy as np
from pyrep.robots.arms.nao_arm import NAOLeftArm, NAORightArm
from pyrep.robots.end_effectors.nao_hand import NAOHand
from robots import NAOAgent
from utils import normalize_target_angle_obs, normalize_joints_obs, normalize_ball_position_obs
from pyrep.objects.vision_sensor import VisionSensor
import itertools

#POS_MIN, POS_MAX = [-0.9, 0.88, 0.815], [-1.05, 0.99, 0.815]
#POS_MIN, POS_MAX = [-0.5190, 0.79, 0.815], [-1.49, 1.22, 0.815]

POS_MIN = [-0.8690, 0.8450, 0.0]
POS_MAX = [-1.1690, 1.09, 1.0]
from collections import deque
from pyrep.objects import Shape, Dummy

POS_TARGET = [-1.0, 1.02, 0.8150]


class NAOCuboidEnv(gym.Env):

    def __init__(self, scene_file, not_open_vrep, stack_obs=4):
        print("Initing Env")
        self.pr = PyRep()
        self.pr.launch(scene_file, headless=not_open_vrep)
        self.cam = VisionSensor('Vision_sensor')
        self.init = True
        self.reset()

        sim.simSetStringSignal('videoPath', '/home/brain/alana/cog_im/videos/vid1')
        #sim.simSetBoolParameter(sim.sim_boolparam_video_recording_triggered, True)

        self.NAO = NAOAgent(NAOLeftArm(), NAORightArm(), NAOHand(), NAOHand())
        self.target_name = '/Cuboid'
        self.target = Shape(self.target_name)
        self.target.scalefactor = 1.0
        self.head = Shape('NAO_head_link2_visible')
        self.last_let_go = False
        self.time_steped = 0
        # começa a montar o vetor de observação
        #posição das juntas
        self.single_obs, _ = self.make_single_observation()
        self.obs = deque(maxlen=stack_obs)
        [self.obs.append(self.single_obs) for i in range(stack_obs)]

        #self.pr.start_video_recording("training.mp4")

        #window_handle = pyrep._active_windows[0]
        #pyrep.set_simulation_timestep(0.01)

        # Inicializa a gravação de vídeo usando o OpenCV
        #video_out = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))

        dim_obs, _ = self.make_single_observation()
        #high_obs = np.inf * np.ones([dim_obs])
        dim_obs = stack_obs*len(dim_obs)
        high_obs = np.ones([dim_obs])
        low_obs = np.zeros([dim_obs])
        self.action_space = gym.spaces.Box(np.float32(self.NAO.low_act), np.float32(self.NAO.high_act))
        #self.observation_space = gym.spaces.Box(np.float32(-high_obs), np.float32(high_obs))
        self.observation_space = gym.spaces.Box(np.float32(low_obs), np.float32(high_obs))
        #self.observation_space = gym.spaces.Box(84, 84, 3)

        #self.observation_space = gym.spaces.Box(low=0, high=1.0,
        #           shape=(3, 84, 84), dtype=np.float32)


    def get_pr(self):
        return self.pr

    def get_low_act(self):
        return self.NAO.low_act

    def get_high_act(self):
        return self.NAO.high_act

    def get_path(self, directory=None):
        self.directory = directory[:len(directory) - len('/params.pkl')]

    def make_single_observation(self):
        observation = []
        count_collistions = 0

        for joint_handle in self.NAO.joint_handles:
            observation.append(sim.simGetJointPosition(joint_handle))

        for col_handle in self.NAO.col_handles:
            observation.append(sim.simReadCollision(col_handle))

        target_pos = self.target.get_position(relative_to=self.head)
        observation += list(target_pos)

        target_angle = self.target.get_orientation(relative_to=self.head)
        observation += list(target_angle)

        for fing_name in self.NAO.col_fingers:
            if sim.simReadCollision(sim.simGetCollisionHandle(fing_name)):
                count_collistions += 1

        self.ball_z = self.target.get_position()[2]

        ((l_ax, l_ay, l_az), (r_ax, r_ay, r_az)) = self.NAO.get_tip_position()
        tx, ty, tz = self.target.get_position()
        l_dist = np.sqrt((l_ax - tx) ** 2 + (l_ay - ty) ** 2 + (l_az - tz) ** 2)

        r_dist = np.sqrt((r_ax - tx) ** 2 + (r_ay - ty) ** 2 + (r_az - tz) ** 2)

        observation.append(l_dist)
        observation.append(r_dist)

        return observation, count_collistions

    def make_observation(self):
        ob, collisions_count = self.make_single_observation()
        self.obs.append(ob)
        #collisions_count = ob[26:44]
        #collisions_count = np.sum(collisions_count)
        observation = list(itertools.chain.from_iterable(self.obs))
        observation = np.asarray(observation).astype('float32')
        #print(f'max obs: {np.max(observation)} min obs: {np.min(observation)}')
        #observation = np.clip(observation, a_min=-1.0, a_max=1.0)
        return observation, collisions_count

    def reset(self):
        if self.pr.running:
            self.epochN += 1
            self.pr.stop()
            #if self.epochN <= 20:
                #print(f'AQUIIII - Epoch {self.epochN}')
                #print(sim.sim_boolparam_video_recording_triggered)
                #exit()
                #sim.simSetBoolParameter(sim.sim_boolparam_video_recording_triggered, 1)
                #print('video')
                #sim.simSetStringSignal('videoPath', 'home/brain/alana/cog_im/rec/video{}'.format(self.epochN))
                #sim.sim_stringparam_video_filename(1)

        #self.episode_positions = []
        self.nSteps = 0
        #self.total_reward = 0
        #self.ball_z = 0
        #self.ball_time_caught = 0
        #self.ball_time_raised = 0
        self.pr.start()

        if self.init:
            print("Successfully Initiated")
            self.init = False
            self.epochN = 0
            #self.pr.set_boolean_parameter(PyRep.BoolParam.VIDEO_RECORDING_TRIGGERED, False)
            #self.pr.set_simulation_parameters({'sim_boolparam_video_recording_triggered': False})

            #client_id = self.pr.get_client_id()
            #sim.simxSetBooleanParameter(client_id, sim.sim_boolparam_video_recording_triggered, False,
                                        #sim.simx_opmode_oneshot)
            #sim.simSetBoolParameter(sim.sim_boolparam_video_recording_triggered, True)
            #sim.simSetBoolParameter(sim.sim_boolparam_video_recording_triggered, True)
            #sim.simxSetBooleanParameter(sim.sim_boolparam_video_recording_triggered, False,
            #                            sim.simx_opmode_oneshot)

            #sim.simSetStringSignal("simBoolparamVideoRecordingTriggered", False)
            #client_id = self.pr.simulator
            #client_id = self.pr.simulator.client_id
            #sim.simxSetStringSignal(client_id, "simBoolparamVideoRecordingTriggered", "false", sim.simx_opmode_oneshot)
            return
        else:
            #pos = list(np.random.uniform(POS_MIN, POS_MAX))
            self.target.set_position(POS_TARGET)
            # Scale objects size to multisizes
            #lib.simScaleObjects([self.target.get_handle()], 1, self.target.scalefactor, False)
            #self.target.scalefactor = np.random.uniform(0.7, 1.1)
            #lib.simScaleObjects([self.target.get_handle()], 1, self.target.scalefactor, False)
            #self.target.scalefactor = 1 / self.target.scalefactor
            # Set NAO's joint to the initial position
            self.NAO.set_initial_joint_positions()
            # Set an initial random position for the joints every reset,
            # in order to increase the different initial states
            #self.NAO.make_action(np.random.uniform(self.NAO.low_act,
            #                                       self.NAO.high_act)
            #                     )
            self.pr.step()
            obb, _ = self.make_observation()
            #obb = np.append(obb, 0.0)
            return obb

    def reset_ball(self):
        self.target.set_position(POS_TARGET)

    def step(self, action):

        self.NAO.make_action(action)
        self.pr.step()
        observation, colisions = self.make_observation()

        if self.ball_z <= 0.5:
            reward = -10.0
            self.reset_ball()
        elif self.ball_z > 0.81:
            reward = 10.0*colisions
        else:
            reward = colisions

        self.time_steped += self.pr.get_simulation_timestep()
        self.nSteps += 1

        frame = (self.cam.capture_rgb() * 255).astype(np.uint8)
        frame = np.asarray(frame)
        frame = np.transpose(frame, (2,0,1))

        return observation, reward, frame

    def render(self):
        pass

    def close(self):
        self.pr.stop()
        self.pr.shutdown()

