# pip install "metaworld>=2.0.0" "gymnasium>=0.29" "lerobot>=0.3.0" "imageio[ffmpeg]"
import os
import sys
import numpy as np
import torch
import json
import metaworld
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters
import types
from gymnasium import spaces

from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionPolicy,
)
from lerobot.constants import ACTION, OBS_IMAGES

from lerobot.utils.random_utils import set_seed
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import imageio
from copy import deepcopy
from typing import Optional
from torch import Tensor
from einops import rearrange
from collections import deque

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # to avoid multi-gpu allocation
HOME = os.environ.get("DSLR_HOME", os.getcwd())
sys.path.append(HOME)
from envs.metaworld.mw_smolvla_eval import get_args

def eval_seeds(task_name, repo_id, seed):
    # Check device is available
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # create env and task
    with open(os.path.join(HOME, "envs/metaworld/tasks_metaworld.json"), "r") as f:
        mw_data = json.load(f)
    task_desc = mw_data["TASK_DESCRIPTIONS"][task_name]
    env = gym.make("Meta-World/MT1", env_name=task_name, render_mode="rgb_array", seed=seed, num_goals=1)
    set_seed(seed) # lerobot seed
    video_dir = f"envs/metaworld/videos/smolvla/{task_name}/{seed}"

    #get the policies
    policy = SmolVLAPolicy.from_pretrained(repo_id)
    device = "cuda:0"
    policy = policy.to(device).eval()

    same_actions_as_first = False
    video_prefix = "unique_noise"
    rollout_smolvla(
        env,
        policy,
        same_actions_as_first,
        video_prefix,
        device,
        video_dir, 
        task_desc
    )

    same_actions_as_first = True
    video_prefix = "same_noise"
    rollout_smolvla(
        env,
        policy,
        same_actions_as_first,
        video_prefix,
        device,
        video_dir, 
        task_desc
    )
    env.close()



def rollout_smolvla(
    env: metaworld.SawyerXYZEnv,
    policy: SmolVLAPolicy,
    same_noise_as_first: bool,
    video_prefix: str,
    device,
    video_dir, 
    task_desc
):
    max_frames = 300
    all_rollout_frames = []
    init_noise_generator = torch.Generator(device=device)
    init_noise_generator.manual_seed(4221)
    rec_env = RecordVideo(env, video_folder=video_dir+f"_{video_prefix}", episode_trigger=lambda ep: True)
    actions_shape = (1, policy.config.chunk_size, policy.config.max_action_dim)

    # set some params
    successes = 0
    n_episodes = 10
    n_record = 10

    if same_noise_as_first:
        # save a single initial noise
        saved_noise = policy.model.sample_noise(actions_shape, device)
    with torch.no_grad():
        for i in range(n_episodes):
            mw_env = rec_env if i < n_record else env
            obs, info = mw_env.reset()
            frame = mw_env.render()
            frames = [frame]
            i+=1

            # set the init noise
            if same_noise_as_first:
                noise = saved_noise
            else:  # new action
                noise = policy.model.sample_noise(actions_shape, device)
            
            for _ in range(max_frames):
                # lerobot maintains an obs queue of it's own
                obs_dict = {
                    "pixels": frame.copy(),                    # HxWx3 uint8
                    "agent_pos": obs[:4].astype(np.float32),
                    "environment_state": obs[:39].astype(np.float32),
                }
                batch = preprocess_observation(obs_dict)
                batch = {
                    k: v.to(device if torch.cuda.is_available() else "cpu", non_blocking=True)
                    for k, v in batch.items()
                }
                
                batch["task"]= task_desc
                # lerobot maintains an action queue too
                action = policy.select_action(batch, noise=noise).cpu().numpy().astype(np.float32).squeeze()
                action = np.clip(action, mw_env.action_space.low, mw_env.action_space.high)
                obs, reward, _, _, info = mw_env.step(action)
                frame = mw_env.render()
                frames.append(frame)
                if info.get("success", 0.0) > 0.0:
                    successes += 1
                    break
            
            # Pad to max length with last frame
            while len(frames) < max_frames:
                frames.append(deepcopy(frames[-1]))

            all_rollout_frames.append(frames)
        
        # --- Create overlay video ---
        overlay_frames = []
        for frame_idx in range(max_frames):
            overlay = np.mean(
                [rollout[frame_idx] for rollout in all_rollout_frames], axis=0
            ).astype(np.uint8)
            overlay_frames.append(overlay)

        imageio.mimsave(f"{video_dir}_{video_prefix}/{video_prefix}_overlay.mp4", overlay_frames, fps=30)
        print("Saved overlay.mp4")
        
        success_rate = successes / n_episodes
        print(f"Saved MP4(s) under: {video_dir}")
        print(f">>> Success rate over {n_episodes} episodes: {success_rate:.2f}")
    

if __name__ == "__main__":
    args = get_args()
    eval_seeds(args.task_name, args.repo_id, args.seeds[0])
