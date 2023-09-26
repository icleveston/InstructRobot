import os
import torch
import argparse

from Main import Main


class Test(Main):

    def __init__(self, headless: bool = False):
        super().__init__(headless=headless)

    @torch.no_grad()
    def test(self, experiment_root_dir: str) -> None:

        # Set the folders
        self.output_path = os.path.join('out', experiment_root_dir)
        self.checkpoint_path = os.path.join(self.output_path, 'checkpoint')
        self.info_path = os.path.join(self.output_path, 'info')
        self.images_path = os.path.join(self.output_path, 'images')

        # Load the model
        self._load_checkpoint(best=False)


def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument("--experiment-root-dir", type=str, required=True, help="Experiment root directory.")

    return vars(arg.parse_args())


if __name__ == "__main__":

    args = parse_arguments()

    if args['experiment_root_dir']:
        Test().test(args["experiment_root_dir"])

