NAoVrepEnv


# Proposed Framework
Here are details regarding the implementation of the proposed experiments,further doc will describes the design of the pseudocode of the controller environment for achieving the desired affordance learning, with the proposedsystem architecture, and finally which the affordances learning are going to happen.

## Simulated environment

To carry out the experiments, we chose a robotics high fidelity simulator. CoppeliaSim, once known as [V-Rep](https://www.coppeliarobotics.com/), is cross-platform and allows the creation of portable, scalable, and easily maintainable content. It will enable different implementations, such as: Directly inside its environment with Lua scripts, externally by Lua Add-ons, using Plug-ins, ROS (Robot Operating System) integration, or with a custom TCP/IP Client-Server integration through a remote API. The latter functionality is the chosen one.

## PyRep toolkit

PyRep is a toolkit for robot learning research and is based on CoppeliaSim/V-REP. It has been going under development as an open-source since 2019. It allows starting the simulation thread of CoppeliaSim from Python. Therefore, it can execute the PyRep python code synchronously with the simulation loop, increasing the speed compared with the Python client that interacts with CoppeliaSim through remote procedure calls. This structure is particularly interesting for learning algorithms, mainly used for robot learning research, reinforcement learning, imitation learning, state estimation, et cetera. [PyRep](https://github.com/stepjam/PyRep) has been written and provided by Stephen James (Imperial College London) with the CoppeliaSim developers team acknowledgment and approval.

The steps described on the python library Github home page, respecting their requirements and development dependencies, were followed to have a successful installation. The model was executed on a Linux SO Laptop, although initial development was taken in a Windows-based Laptop. The simulations run primarily in Headless Mode, which means to keep the learning stage without the graphical user interface and to execute our test cases with a "Headless Mode." That last saves some costs and time when running multiple instances in parallel.


## NAO Robot Model

This model of NAO robot for CoppeliaSim is courtesy of Marco Cognetti \cite{cognetti2015whole}. The mesh data and the pre-recorded movement data are courtesy of Aldebaran Con., the company behind such a machine's development.

NAO is a small (58 cm) humanoid robot developed by Aldebaran Robotics. It has two legs, each with 5 degrees of freedom, the other two arms each have 5 degrees, the pelvis has one each, the neck has another two, and the last three fingers each have 2 to 3 (a total of 43 Degrees of freedom), two cameras can see all joints on Figure \ref{NAOJoints}. From the sensor's point of view, there are two cameras (up and down) on NAO's head, as shown in Figure \ref{NAOOverview}. However, only one camera can be used at a time to obtain a monocular vision system. This limitation happens because the power consumption reduces the robot's autonomy, and the two cameras cannot be activated simultaneously.


Also, the limited CPU (1.6 GHz) does not allow the processing of data streams from two cameras and data from other sensors. Subsequently, NAO experienced heating problems. Since each of its 43 motors and each sensor requires power, the more sensors there are, the bigger the probability is of the system's overheating. After explaining all of these, the reinforcement learning environment strategy will be transferred and put into the real world, which means solving simple and complex problems rather than computing resource requirements. That was one of the biggest challenges of many from this project.


## Rlkit toolkit

Rlkit is a reinforcement learning framework whose algorithms are implemented in the PyTorch library. The PyTorch library is an open-source machine learning library based on the Torch library, used for AI-related applications, and is mainly developed by Facebook's AI Research Lab (FAIR). Rlkit uses a new multi-world code that requires explicit environmental registration and has an online algorithm mode. They only have Q-networks, no unnecessary policy regularization terms, numerically stable Jacobian calculations, modularity, and readable code.

Rlkit has the following implemented algorithms:
- Skew-Fit
- Reinforcement Learning with Imagined Goals (RIG)
- Temporal Difference Models (TDMs)
- Hindsight Experience Replay (HER)
- (Double) Deep Q-Network (DQN)
- Soft Actor-Critic (SAC)
- Twin Delayed Deep Deterministic Policy Gradient (TD3)
- Advantage Weighted Actor-Critic (AWAC)

The most significant advantage of [Rlkit](https://github.com/vitchyr/rlkit) is that it supports the rllab library, which provides interfaces with MuJoCo and OpenAi Gym. MuJoCo \cite{todorov2012mujoco}, and OpenAI Gym \cite{brockman2016openai} is another reinforcement learning toolkit that contains a wide range of environmental and physical environment relationships and allows to simplify complex environment simulations related to enhanced variant perception and calculation. All these combinations allow the custom environment to be trained in a virtual environment and then converted into the real world.
