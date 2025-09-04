from lerobot_dsrl import generate_steerable_diffpo_pusht_gym_env, DiffpoEnvWrapper
import numpy as np
import imageio
import gymnasium as gym
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import gym_pusht
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.envs.utils import preprocess_observation
import torch
from stable_baselines3 import SAC
from typing import Optional, Dict, Any, List
from gymnasium.wrappers import TimeLimit
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from gymnasium.wrappers import RescaleAction


class ResetOptionsWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        options: Optional[Dict[str, Any]] = None,
        seeds: Optional[List[int]] = None,
    ):
        super().__init__(env)
        self.options = options
        self.seeds = seeds if seeds is not None else []
        self._seed_idx = 0

    def reset(self, **kwargs):
        # Cycle through seeds if provided
        if self.seeds:
            seed = self.seeds[self._seed_idx % len(self.seeds)]
            kwargs["seed"] = seed
            self._seed_idx += 1

        if self.options is not None:
            kwargs["options"] = self.options

        return self.env.reset(**kwargs)

def eval_base_policy(env, seeds, video_parent_dir = "videos", video_dir_name = "base"):
    # make policy
    policy = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht").to(device)

    # results storage
    all_episode_returns = []
    video_dir_path = f"{video_parent_dir}/{video_dir_name}"

    os.makedirs(video_dir_path, exist_ok=True)

    for episode_idx, seed in enumerate(seeds, start=1):
        policy.reset()
        obs, info = env.reset()
        frames = [env.render()]
        total_reward = 0.0
        terminated, truncated = False, False

        for t in range(max_timesteps):
            obs = preprocess_observation(obs)
            obs = {
                k: v.to(device if torch.cuda.is_available() else "cpu", non_blocking=True)
                for k, v in obs.items()
            }

            with torch.inference_mode():
                action = policy.select_action(obs)

            action = action.to("cpu").numpy().squeeze(0)
            obs, reward, terminated, truncated, info = env.step(action)
            reward = -1
            total_reward += reward

            frames.append(env.render())

            if terminated or truncated:
                break

        # save video
        video_path = f"{video_dir_path}/episode_{episode_idx}_seed_{seed}.mp4"
        imageio.mimsave(video_path, frames, fps=20)
        print(f"Saved video for seed {seed} -> {video_path}, returns: {total_reward}")
        # save returns
        all_episode_returns.append(total_reward)
        np.save(f"{video_dir_path}/returns.npy", all_episode_returns)

       

def eval_random_policy(env, seeds, max_timesteps, copy_first_action, success_threshold, sac_path, deterministic=True,  video_parent_dir = "videos", video_dir_name = "latent_random"):
    video_dir_path = f"{video_parent_dir}/{video_dir_name}"
    os.makedirs(video_dir_path, exist_ok=True)
    # make diffpo
    policy = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht").to(device)

    # Me adding DDIM
    policy.diffusion.noise_scheduler = DDIMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        # beta_schedule is important
        # this is the best we found
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type=policy.diffusion.config.prediction_type,  # or sample
    )

    # Wrap the env
    wrapped_env = DiffpoEnvWrapper(env, policy, options, seed=None, success_threshold=success_threshold,copy_first_action=copy_first_action)

    if sac_path is not None:
        action_min = np.ones([2])*-1
        action_max = np.ones([2])
        # linearlly normalize obs/action to [-1,1]
        wrapped_env = RescaleAction(wrapped_env, min_action=action_min, max_action=action_max)
        model = SAC.load(sac_path,env=wrapped_env)

    max_frames = max_timesteps
    all_rollout_frames = []
    all_episode_returns = []

    init_noise_generator = torch.Generator(device="cuda")
    init_noise_generator.manual_seed(4221)

    for rollout in range(len(seeds)):
        obs, info = wrapped_env.reset()
        total_rewards = 0
        # get the initial frame
        frames = wrapped_env.get_wrapper_attr("frames")
        for current_step in range(max_frames):
            # Action select
            # Random action
            # action is shape (batch_size, horizon, action dim)
            if sac_path is not None:
                action = model.predict(obs, deterministic=deterministic)[0]
            else:
                action = wrapped_env.action_space.sample()

            # Step in env in wrapper model.
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            total_rewards += reward
            frames += wrapped_env.get_wrapper_attr("frames")

            if terminated or truncated:
                break
        print(f"terminated:{terminated}, returns: {total_rewards}")
        
        # get frames
        # Save individual video
        video_path = f"{video_dir_path}/episode_{rollout}_seed_{seeds[rollout]}.mp4"
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Saved {video_path}, terminated:{terminated}, returns: {total_rewards}")

        # save the returns
        all_episode_returns.append(total_rewards)
        np.save(f"{video_dir_path}/returns.npy", all_episode_returns)

    #     # Pad to max length with last frame
    #     while len(frames) < max_frames:
    #         frames.append(deepcopy(frames[-1]))

    #     all_rollout_frames.append(frames)

    # # --- Create overlay video ---
    # overlay_frames = []

    # for frame_idx in range(max_frames):
    #     overlay = np.mean(
    #         [rollout[frame_idx] for rollout in all_rollout_frames], axis=0
    #     ).astype(np.uint8)
    #     overlay_frames.append(overlay)

    # imageio.mimsave(f"{video_dir_path}/overlay.mp4", overlay_frames, fps=30)
    # print("Saved overlay.mp4")



if __name__ == "__main__":
    ## SAC Eval stuff
    sac_model_dir = "./my_models"
    sac_ckpt = "graceful_breezev20.zip"
    sac_path = f"{sac_model_dir}/{sac_ckpt}"
    copy_first_action = True
    deterministic = True
    video_parent_dir = "eval_videos"

    if sac_path is None:
        video_dir_name = "latent_random"
    else:
        video_dir_name = f"latent_{sac_ckpt.split('.')[0]}"

    # pusht options
    success_threshold = 0.9

    options = None
    # reset_state = np.array([314, 201, 187.21077193, 275.01629149, np.pi / 4.0])
    # options = {"reset_to_state": reset_state}

    max_timesteps = 1000  # PushT already capped at 300 steps
    device = "cuda"

    gym_handle = "gym_pusht/PushT-v0"
    gym_kwargs = {
        "obs_type": "pixels_agent_pos",
        "render_mode": "rgb_array",
        "visualization_width": 384,
        "visualization_height": 384,
        "max_episode_steps": max_timesteps,
    }

    # list of seeds to evaluate
    seeds = list(range(100))
    env = gym.make(gym_handle, disable_env_checker=True, **gym_kwargs)
    env.unwrapped.success_threshold = success_threshold
    env = ResetOptionsWrapper(env, options=options, seeds=seeds)
    env = TimeLimit(env, max_episode_steps=max_timesteps)
    
    # eval_base_policy(env, seeds, video_parent_dir=video_parent_dir, video_dir_name="base")

    eval_random_policy(env, seeds, max_timesteps, copy_first_action=copy_first_action, success_threshold=success_threshold, sac_path=sac_path,deterministic=deterministic,video_parent_dir=video_parent_dir, video_dir_name=video_dir_name)
