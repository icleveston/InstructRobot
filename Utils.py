import argparse
import os
import pickle
import shutil
import torch
import wandb
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFont, ImageDraw
from csv import writer


def percentage_error_formula(x, amount_variation): round(x / amount_variation * 100, 3)


def save_wandb(in_queue, resume, conf, model_name, images_path, info_path, checkpoint_path):
    # Init Wandb
    wandb.init(
        project=str(conf),
        name=model_name,
        id=model_name,
        resume=resume
    )

    while True:
        data = in_queue.get()
        in_queue.task_done()

        # Unpack data
        return_info, loss_info, lr, eps, action_std, observations, current_step, agent_state, optim_state, \
            scheduler_state, best_mean_episodic_return = data

        # Unpack observations
        #actions = []
        #observations = []

        #for r in obs_rollout:
            #actions.append(r[0].cpu().data.numpy())
            #observations.append(r[1])

        # Log Wandb
        wandb.log(
            {
                "mean_episodic_return": return_info["mean"],
                "loss/actor": loss_info["actor"],
                "loss/entropy": loss_info["entropy"],
                "loss/critic": loss_info["critic"],
                "lr": np.float32(lr),
                "action_std": np.float32(action_std),
                "eps": np.float32(eps),
                "video": wandb.Video(_format_video_wandb(observations), fps=8),
                #f"actions-{current_step}": wandb.Table(columns=[f"a{i}" for i in range(26)], data=actions)
            }, step=current_step)

        # Dump observation data
        with open(os.path.join(images_path, f"observation.p"), "wb") as f:
            pickle.dump(observations, f)

        row = [
            current_step,
            loss_info["actor"],
            loss_info["critic"],
            loss_info["entropy"],
            return_info["mean"],
            return_info["std"]
        ]

        # Save training history
        with open(os.path.join(info_path, 'history.csv'), 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(row)
            f_object.close()

        is_best = return_info["mean"] > best_mean_episodic_return

        # Save the checkpoint for each rollout
        _save_checkpoint({
            "current_step": current_step,
            "best_mean_episodic_return": best_mean_episodic_return,
            "model_state": agent_state,
            "optim_state": optim_state,
            "scheduler_state": scheduler_state,
            "eps_clip": eps,
            "action_std": action_std,
        }, checkpoint_path, is_best)


def _format_video_wandb(last_obs_rollout) -> np.array:
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 256)),
        transforms.ToPILImage(),
    ])

    video = []

    font = ImageFont.truetype("Roboto/Roboto-Medium.ttf", size=10)

    for index, o in enumerate(last_obs_rollout):
        image_array = np.concatenate((trans(o[-1][1]), trans(o[-1][2])), axis=0)

        image = Image.fromarray(image_array, mode="RGB")

        image_editable = ImageDraw.Draw(image)

        image_editable.text((10, 115), o[-1][0], (0, 0, 0), align="center", font=font)

        video.append(np.asarray(image))

    video = np.asarray(video)
    video = np.transpose(video, (0, 3, 1, 2))

    return video


def _save_checkpoint(state, checkpoint_path, is_best):
    # Set the checkpoint path
    ckpt_path = os.path.join(checkpoint_path, "checkpoint.tar")

    # Save the checkpoint
    torch.save(state, ckpt_path)

    # Save the best model
    if is_best:
        # Copy the checkpoint to the best model
        shutil.copyfile(ckpt_path, os.path.join(checkpoint_path, "best_model.tar"))


def online_mean_and_sd(images):
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    b, c, h, w = images.shape
    nb_pixels = b * h * w
    sum_ = torch.sum(images, dim=[0, 2, 3])
    sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
    fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
    snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

    cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

