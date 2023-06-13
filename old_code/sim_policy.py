from rlkit.samplers.util import rollout
from rlkit.torch.core import PyTorchModule
from rlkit.torch.pytorch_util import set_gpu_mode
from rlkit.torch.sac.policies import MakeDeterministic
import joblib
import uuid
from rlkit.core import logger

from NAOVrepSac import NAOBallEnv
from rlkit.envs.wrappers import NormalizedBoxEnv

filename = str(uuid.uuid4())


def simulate_policy():
    #data = joblib.load(args.file)
    #data = joblib.load(r"C:\Users\renan-note\Documents\VrepNAOSac\data\name-of-experiment\name-of-experiment_2019_08_16_00_09_38_0000--s-0\params.pkl")
    data = joblib.load('/home/nanbaima/Documents/naovrepenv/data/params.pkl')
    policy = MakeDeterministic(data['policy'])
    # env = data['env']
    env = NormalizedBoxEnv(NAOBallEnv())
    # env.reset_instance()
    print("Policy loaded")
    if False:
        set_gpu_mode(True)
        policy.cuda()
    if isinstance(policy, PyTorchModule):
        policy.train(False)

    while True:
        path = rollout(
            env,
            policy,
            max_path_length=1000,
            animated=True,
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()


if __name__ == "__main__":

    simulate_policy()
