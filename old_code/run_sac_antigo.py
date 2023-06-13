import numpy as np
#from envs.marta_sac_env import MartaVrepEnvSac
from NAOVrepSac import NAOBallEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.networks import FlattenMlp
from rlkit.envs.wrappers import NormalizedBoxEnv

def experiment(variant):
    env = NormalizedBoxEnv(NAOBallEnv())

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    net_size = variant['net_size']
    qf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    algorithm = SoftActorCritic(
        env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=10000000,
            num_steps_per_epoch=1200,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=1000,
            discount=0.99,
            reward_scale=1,

            soft_target_tau=5e-3,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
        ),
        net_size=512,
    )
    setup_logger('name-of-experiment', variant=variant)
    ptu.set_gpu_mode(False)  # optionally set the GPU (default=False)
    experiment(variant)
