from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
from rlkit.core import logger

from NAO_SAC_policy import NAOBallEnv
from rlkit.envs.wrappers import NormalizedBoxEnv

filename = str(uuid.uuid4())

def simulate_policy(args, env_init):
    data = torch.load(args.file)
    policy = data['evaluation/policy']
    env = NormalizedBoxEnv(env_init)
    env.get_path(args.file)
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    for _ in range(400):
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            render=True,
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=250,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    env_init = NAOBallEnv()

    simulate_policy(args, env_init)
    print('Done!')
    env_init.close()
