import gymnasium as gym
from lerobot.policies.pi0.modeling_pi0 import (
    PI0Policy,
)
from lerobot.envs.utils import preprocess_observation, add_envs_task
import torch
import imageio
import os

class Pi0EnvWrapper(gym.Env):
    def __init__(
        self,
        env: gym.Env,
        policy: PI0Policy,
        save_frames: bool = True,
    ):
        super().__init__()

        self.env = env
        self.policy = policy
        self.policy.eval()

        self._last_obs = None  # last obs we got from env (this is a raw obs from env, used for base policy)

        # if save_frames is true, then all the frames during the current action chunk execution are saved. step refreshes it, reset refreshes and adds first obs frame
        self.save_frames = save_frames
        if self.save_frames:
            self.frames = []

    def reset(self, **kwargs):
        # TODO load gym seed if it was given at construction
        obs, info = self.env.reset(**kwargs)
        # save the obs for base policy to use in next step
        self._last_obs = obs
        # reset policy
        self.policy.reset()
        # reset frames
        if self.save_frames:
            # reset the frames, and add initial
            self.frames = [self.render()]

def run_basic_pi0_aloha(seeds : list[int] = [0], device : str = "cuda"):
    video_dir_path = "aloha_pi0_videos_5000ckpt"

    os.makedirs(video_dir_path, exist_ok=True)
    import gym_aloha
    fps = 30
    max_steps = 1000

    # TODO: Add insert
    gym_kwargs = {
        "obs_type": "pixels_agent_pos",
        "render_mode": "rgb_array",
    }
    env = gym.make("gym_aloha/AlohaTransferCube-v0", **gym_kwargs)

    # policy = PI0Policy.from_pretrained("BrunoM42/pi0_aloha_transfer_cube").to(device)
    policy = PI0Policy.from_pretrained("../lerobot/outputs/train/pi0_transfer_cube/checkpoints/005000/pretrained_model").to(device)
    # policy = PI0Policy.from_pretrained("lerobot/pi0")

    for rollout, seed in enumerate(seeds):
        policy.reset()

        total_rewards = 0
        observation, info = env.reset(seed=seed)

        frames = [env.render()]

        for step in range(max_steps):
            observation = preprocess_observation(observation)
        
            observation = {
                key: observation[key].to(device, non_blocking=True) for key in observation
            }
            # Set task description
            # TODO: For now, just using blank
            observation["task"] = ["transfer_cube"]

            with torch.inference_mode():
                action = policy.select_action(observation)

            # Convert to CPU / numpy.
            action = action.to("cpu").numpy()
            assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"
            action = action[0]

            # Apply the next action.
            observation, reward, terminated, truncated, info = env.step(action)

            frames.append(env.render())
            done = truncated or terminated
            total_rewards += reward

            if done:
                break

        print(f"Done: {done} / episode returns: {total_rewards}")
        video_file_name = f"aloha_pi0_rollout{rollout}_seed{seed}.mp4"
        imageio.mimsave(f"{video_dir_path}/{video_file_name}", frames, fps=fps)

def main():
    seeds = [0]*10
    run_basic_pi0_aloha(seeds=seeds)

if __name__ == "__main__":
    main()