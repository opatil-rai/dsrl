from lerobot_dsrl import generate_steerable_diffpo_pusht_gym_env
import numpy as np
import imageio
from copy import deepcopy
import gymnasium as gym
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import gym_pusht
from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionPolicy,
)
from lerobot.envs.utils import preprocess_observation
import torch

import gymnasium as gym
from typing import Optional, Dict, Any
from gymnasium.wrappers import TimeLimit

class ResetOptionsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, options: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
        super().__init__(env)
        self.options = options
        self.seed = seed

    def reset(self, **kwargs):
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.options is not None:
            kwargs["options"] = self.options
        return self.env.reset(**kwargs)

if __name__ == "__main__":
    # reset_state = np.array([314, 201, 187.21077193, 275.01629149, np.pi / 4.0])
    # options = {"reset_to_state": reset_state}
    options = None

    max_timesteps = 1000
    device="cuda"

    gym_handle = "gym_pusht/PushT-v0"
    # default kwargs for the pusht gym env
    gym_kwargs = {
        "obs_type": "pixels_agent_pos",
        "render_mode": "rgb_array",
        "visualization_width": 384,
        "visualization_height": 384,
        "max_episode_steps": 300,
    }
    env = gym.make(gym_handle, disable_env_checker=True, **gym_kwargs)
    env = ResetOptionsWrapper(env, options)
    env = TimeLimit(env, max_episode_steps=max_timesteps)


    # make diffpo
    policy = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht").to(device)


    # Me adding DDIM
    # policy.diffusion.noise_scheduler = DDIMScheduler(
    #     num_train_timesteps=100,
    #     beta_start=0.0001,
    #     beta_end=0.02,
    #     # beta_schedule is important
    #     # this is the best we found
    #     beta_schedule="squaredcos_cap_v2",
    #     clip_sample=True,
    #     set_alpha_to_one=True,
    #     steps_offset=0,
    #     prediction_type=policy.diffusion.config.prediction_type,  # or sample
    # )
    
    obs, info = env.reset()
    frames = []
    # capture initial frame
    frames.append(env.render())

    for i_step in range(max_timesteps):
        obs = preprocess_observation(obs)
        obs = {
            k: v.to(
                "cuda" if torch.cuda.is_available() else "cpu", non_blocking=True
            )
            for k, v in obs.items()
        }

        with torch.inference_mode():
            action = policy.select_action(obs)

        action = action.to("cpu").numpy().squeeze(0)

        obs, reward, terminated, truncated, info = env.step(action)
        print(reward)

        # capture frame
        frames.append(env.render())

        if terminated or truncated:
            break
    print(f"terminated/truncated: {terminated}/{truncated}")
    # save to mp4
    imageio.mimsave("pusht_rollout.mp4", frames, fps=20)

    env.close()