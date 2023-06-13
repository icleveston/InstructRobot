import argparse
from distutils.util import strtobool
import numpy as np



def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="NAO_RL_classic_baseline",
        help="the name of this experiment")
    parser.add_argument("--run-name", type=str, default=None,
                        help="experiment name")
    parser.add_argument("--num-actions", type=int, default=26)
    parser.add_argument("--learning_rate", type=float, default=1e-5,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--load-model", type=bool, default=False)

    #Brims parameters
    parser.add_argument("--nlayers", type=int, default=1, help="number of layers")
    parser.add_argument('--nhid', nargs='+', type=int, default=[512])
    parser.add_argument('--topk', nargs='+', type=int, default=[2])
    parser.add_argument('--num_blocks', nargs='+', type=int, default=[4])
    parser.add_argument("--ninp", type=int, default=128, help="embedding input")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout")
    parser.add_argument("--use_inactive", type=bool, default=True)
    parser.add_argument("--blocked_grad", type=bool, default=True)

    # Algorithm specific arguments
    parser.add_argument("--num-rollouts", type=int, default=32,
        help="the number of parallel game environments")
    parser.add_argument("--device-num", type=int, default=0,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=1,
        help="the number of mini-batches")
    parser.add_argument("--num-epochs", type=int, default=20,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    parser.add_argument("--wandb-project-name", type=str, default="NAO_testes",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")

    args = parser.parse_args()
    args.steps_sample = int(args.num_rollouts * args.num_steps)
    #args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_updates = args.total_timesteps // args.steps_sample
    # fmt: on
    return args


def function_with_args_and_default_kwargs(optional_args=None, **kwargs):
    parser = argparse.ArgumentParser()
    for k, v in kwargs.items():
        parser.add_argument('--' + k, default=v)
    #args, unknown = parser.parse_args(optional_args)
    args, unknown = parser.parse_known_args(optional_args)

    print('argumentos desconhecidos')
    print(unknown)
    return args



def normalize_joints_obs(joints, low_obs, high_obs):
    y = (joints - low_obs)/(high_obs - low_obs)

    return y.tolist()

def normalize_ball_position_obs(ball_position, min_position, max_position):
    y = (ball_position - min_position) / (max_position - min_position)
    y = np.clip(y, 0, 1)
    return y.tolist()

def normalize_target_angle_obs(ball_angle, low_angle, high_angle):
    y = (ball_angle - low_angle) / (high_angle - low_angle)
    return y

def denormalize_action(action, low_action , high_action, l, u):
    y = (high_action - low_action)*((action - l)/(u - l)) + low_action
    return y


def discount_reward(rewards, gamma, num_rollouts):
    discounted_rewards = np.zeros_like(rewards)

    for env_id in range(0, num_rollouts):
        rw = rewards[:, env_id]
        R = 0
        for t in reversed(range(0, len(rw))):
            # update the total discounted reward
            R = R * gamma + rw[t]
            discounted_rewards[t, env_id] = R
    return discounted_rewards
