import argparse
from distutils.util import strtobool
import os
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageFont, ImageDraw


def plot(experiment_root_dir: str,
         clip_type: str,
         load_best_observation: bool,
         width: int,
         height: int,
         fps: int,
         draw_instruction: bool,
         save_as: str) -> None:

    font = ImageFont.truetype("Main/Roboto/Roboto-Medium.ttf", size=10)

    # Create plot path if it does not exist
    plot_path: str = os.path.join('out', experiment_root_dir, 'plot')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Define which observation to load
    filename: str = "best_observation.tar" if load_best_observation else "last_observation.tar"

    # Set the observation path
    observation_path: str = os.path.join('out', experiment_root_dir, 'info', filename)

    # Load the observations
    observations = torch.load(observation_path)

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((height, width)),
        transforms.ToPILImage(),
    ])

    basename: str = "best_observation" if load_best_observation else "last_observation"

    video_data = []

    # Select rollout
    rollout_n: int = int(input(f"Which rollout would you like to save? [0-{len(observations)-1}]: "))

    # Change the output filename
    basename += f"_rollout_{rollout_n}"

    # Select the rollout
    observation = observations[rollout_n]

    # For each frame in the rollout
    for index, o in enumerate(observation):

        # Transform the images
        top = np.array(trans(o[-1]["frame_top"]))
        front = np.array(trans(o[-1]["frame_front"]))

        # Select the clip type
        if clip_type == "ALL":
            image_array = np.concatenate((top, front), axis=0)
        elif clip_type == "TOP":
            image_array = top
        else:
            image_array = front

        image = Image.fromarray(image_array, mode="RGB")

        if draw_instruction and "instruction" in o[-1].keys():

            image_editable = ImageDraw.Draw(image)

            image_editable.text((10, 115), o[-1]["instruction"], (0, 0, 0), align="center", font=font)

        video_data.append(np.asarray(image))

    # Export video data
    export(save_as, basename, fps, plot_path, video_data)


def export(save_as: str, basename: str, fps: int, plot_path: str, video_data):
    video_data = np.asarray(video_data)
    video_data = np.transpose(video_data, (0, 3, 1, 2))
    video_data = video_data.reshape(1, *video_data.shape)
    b, t, c, h, w = video_data.shape
    n_rows = 2 ** ((b.bit_length() - 1) // 2)
    n_cols = video_data.shape[0] // n_rows
    video_data = np.reshape(video_data, newshape=(n_rows, n_cols, t, c, h, w))
    video_data = np.transpose(video_data, axes=(2, 0, 4, 1, 5, 3))
    video_data = np.reshape(video_data, newshape=(t, n_rows * h, n_cols * w, c))

    # Create the clip
    clip = ImageSequenceClip(list(video_data), fps=fps)

    # Export as gif
    if save_as == 'gif':

        clip_path = os.path.join(plot_path, f"{basename}.gif")
        clip.write_gif(clip_path)

    elif save_as == 'mp4':

        clip_path = os.path.join(plot_path, f"{basename}.mp4")
        clip.write_videofile(clip_path)

    else:

        # For each frame
        for i, f in enumerate(clip.iter_frames()):
            im = Image.fromarray(f)

            # Save frame
            im.save(os.path.join(plot_path, f"{basename}_{i}.png"))

    print(f"Saved to: {plot_path}")


def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument("--experiment-root-dir", type=str, required=True, help="Experiment root directory.")
    arg.add_argument("--type", type=str, default="ALL", dest="clip_type", required=True,
                     choices=["ALL", "TOP", "FRONT"], help="Clip type (ALL, TOP, FRONT).")
    arg.add_argument("--load_best_observation", type=strtobool, default=False,
                     required=False, help="Load either last or best observation.")
    arg.add_argument("--width", type=int, default=256, required=False, help="Clip width.")
    arg.add_argument("--height", type=int, default=128, required=False, help="Clip height.")
    arg.add_argument("--fps", type=int, default=8, required=False, help="Clip fps.")
    arg.add_argument("--draw-instruction", type=strtobool, default=False, required=False,
                     help="Should draw the instruction.")
    arg.add_argument("--save-as", type=str, default='gif', required=False,
                     help="Save clip as (gif, mp4, png).")

    return vars(arg.parse_args())


if __name__ == '__main__':
    args = parse_arguments()

    plot(**args)
