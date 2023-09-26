import random
from PIL import Image
import argparse
from distutils.util import strtobool

from Main import Main


class ValidateEnv(Main):

    def __init__(self, headless: bool = False):
        super().__init__(headless=headless)

    def validate_observations(self) -> None:
        self.env.reset()

        for index in range(128):
            action = [random.randrange(-3, 3) for i in range(26)]

            obs, _ = self.env.step(action)

            print(obs[-1][0])

            im = Image.fromarray(obs[-1][1], mode="RGB")
            im.save(f"./out/{index}.png")

    def validate_joints_nao(self) -> None:
        self.env.validate_joints_nao()

    def validate_collisions_nao(self) -> None:
        self.env.validate_collisions_nao()


def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument("--val-obs", type=strtobool, default=False, required=False, help="Validate observations.")
    arg.add_argument("--val-joint-nao", type=strtobool, default=False, required=False, help="Validate NAO's joints.")
    arg.add_argument("--val-collisions-nao", type=strtobool, default=False, required=False,
                     help="Validate NAO's collisions.")

    return vars(arg.parse_args())


if __name__ == "__main__":

    args = parse_arguments()

    if args['val_obs']:
        ValidateEnv().validate_observations()
    elif args['val_joint_nao']:
        ValidateEnv(headless=False).validate_joints_nao()
    elif args['val_collisions_nao']:
        ValidateEnv(headless=False).validate_collisions_nao()
