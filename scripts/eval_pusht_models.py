from lerobot_dsrl import generate_steerable_diffpo_pusht_gym_env
import numpy as np
import imageio
import gymnasium as gym
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import gym_pusht
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.envs.utils import preprocess_observation
import torch
from typing import Optional, Dict, Any, List
from gymnasium.wrappers import TimeLimit
import matplotlib.pyplot as plt
import os


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


if __name__ == "__main__":
    success_threshold = 0.9

    options = None
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
    seeds = list(range(1, 6))  # [1,2,3,4,5]
    env = gym.make(gym_handle, disable_env_checker=True, **gym_kwargs)
    env.success_threshold = success_threshold
    env = ResetOptionsWrapper(env, options=options, seeds=seeds)
    env = TimeLimit(env, max_episode_steps=max_timesteps)

    # make policy
    policy = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht").to(device)

    # results storage
    all_rewards = []
    success_flags = []

    os.makedirs("videos", exist_ok=True)

    for episode_idx, seed in enumerate(seeds, start=1):
        policy.reset()
        obs, info = env.reset()
        frames = [env.render()]
        episode_rewards = []
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

            total_reward += reward
            episode_rewards.append(total_reward)

            frames.append(env.render())

            if terminated or truncated:
                break

        # save video
        video_path = f"videos/pusht_rollout_seed{seed}.mp4"
        imageio.mimsave(video_path, frames, fps=20)
        print(f"Saved video for seed {seed} -> {video_path}")

        # store results
        all_rewards.append(episode_rewards)
        success_flags.append(terminated and not truncated)

    # compute success rate
    success_rate = np.mean(success_flags)
    print(f"\nAverage success rate over {len(seeds)} seeds: {success_rate:.2f}")

    # plot reward curves
    plt.figure(figsize=(8, 5))
    for i, rewards in enumerate(all_rewards):
        plt.plot(rewards, label=f"Seed {seeds[i]}")
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Reward")
    plt.title("Reward Curves per Episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reward_curves.png")
    plt.show()

    env.close()
