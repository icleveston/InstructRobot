"""
A NAO tries to touch a ball, if it catches with more than 3 fingers, it.
restarts the scene.
This script contains examples of:
    - SAC Enviroment Setting.
    - Using multiple arms and fingers.
    - Using arms and hands.
"""

import numpy as np
from envs import NAOCuboidEnv
import math
import random
from pyrep.backend import sim
from pyrep.backend._sim_cffi import lib
from os.path import dirname, join, abspath
from pyrep import PyRep




if __name__ == "__main__":
    scene_file = '/home/brain/alana/cog_im/Scenes/scene_NAO_SAC_envV1.8.7_cuboid.ttt'

    env = NAOCuboidEnv(scene_file, not_open_vrep=False)



    #collision_name = sim.simGetCollisionObjectName(id_obj)
    #collision_name = sim.simGetCollisionHandle('NAOHand_thumb2_link#0')
    #print(collision_name)

    # Crie uma lista vazia para armazenar os nomes das colisões

    #exit()
    on = True
    cont = 0
    while on:
        env.reset()
        #env.init_video(cont)
        cont += 1
        for _ in range(1000):

            #acts = [-119.5, 29.0,
            #       0.0, 0.0, 0.0, 0.0, -104.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0,
            #       0.0, 0.0, 0.0, 0.0, -104.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 0.0, 0.0, 0.0]

            #acts = np.array(acts) * math.pi / 180
            #print(acts)
            #acts = env.get_high_act()

            o = env.step(np.random.uniform(env.get_low_act(), env.get_high_act()))

            #print(env.NAO.get_joint_positions())
            if o[2]:
            	break
        #env.stop_video()
        if _ == 10:
            on = False

    print('Done!')
    env.close()
