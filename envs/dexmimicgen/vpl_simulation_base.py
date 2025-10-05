"""Modified from visuomotor/simulation_base.py. Removed policy loading code and references to torch3d.
Made it a gym env."""

import math
import time
from collections import defaultdict, deque
from copy import deepcopy
from typing import Any, Deque

import gymnasium as gym
import numpy as np
import torch
from einops import rearrange
from lightning import LightningModule
from numpy import typing as npt
from omegaconf import DictConfig

# from visuomotor.data.rotation_transformer import RotationTransformer
from visuomotor.data.utils import create_key_to_history_mapping
from visuomotor.models.model_registry import REGISTRY


class VPLSimulationBase(gym.Env):

    def __init__(
        self,
        env_name: str,
        current_epoch: int,
        task_index: int,
        num_steps: int,
        config: DictConfig,
        artifact_name: str | None = None,
    ) -> None:

        self.run_check_observation_space = True
        self.config = config
        # self.env_type = config.simulation.get("env_type")
        self.env_type = "dexmimicgen"
        self.metadata_gcs_prefix = self.config.simulation.metadata_gcs_prefix
        self.abs_action = config.simulation.get("abs_action", False)
        self.env_name = env_name
        self.epoch = current_epoch
        self.num_envs = self.config.simulation.get("num_envs_per_actor", 1)

        self.obs_visual = self.config.data.obs_keys.get("visual", list())
        self.obs_state = self.config.data.obs_keys.get("state", dict())

        # Configure history lengths
        self.n_obs_history_visual = self.config.data.get("n_obs_history_visual")
        self.n_obs_history_state = self.config.data.get("n_obs_history_state")

        self.max_n_obs_history = max(self.n_obs_history_visual, self.n_obs_history_state)
        self.key_to_n_obs_history = create_key_to_history_mapping(
            obs_visual=self.obs_visual,
            obs_state=self.obs_state.keys(),
            n_obs_history_visual=self.n_obs_history_visual,
            n_obs_history_state=self.n_obs_history_state,
        )

        self.task_index = task_index

        # self.rotation_transformer = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")

        self.num_steps = num_steps if num_steps is not None else self.config.simulation.get("num_steps", 100)
        print(f"{env_name} running with {self.num_steps}")

        self.num_episodes = self.config.simulation.get("num_episodes", 10) // self.config.simulation.get(
            "ray_parallel_actors", 1
        )
        self.action_execution_steps = self.config.simulation.get("action_execution_steps", 1)
        base_bucket = self.config.simulation.get("base_bucket", "bdai-common-storage")

        # --- Enable profiling optionally via config ---
        self.enable_profiling = self.config.simulation.get("enable_profiling", False)

        print(f"Creating {self.num_envs} environments..")
        self.env = self.setup_env(base_bucket=base_bucket)
        print("Environment created.")

    def setup_env(self, base_bucket: str) -> gym.Env:
        """
        Return:
            Gymnasium environment.
        """
        raise NotImplementedError

    def process_n_obs(self, last_n_obs: Deque) -> dict:
        """Args:
        last_n_obs: Iterable of last n observations
        """
        assert len(last_n_obs) == self.max_n_obs_history

        last_n_batches = defaultdict(list)
        for i_obs, obs in enumerate(last_n_obs):
            batch = self.process_obs(obs)
            for k, v in batch.items():
                n_obs_history = self.key_to_n_obs_history[k]
                if i_obs < n_obs_history:
                    arr = torch.from_numpy(v)
                    # Convert float64 to float32
                    # When unbatched, it is already float32.
                    # Ideally we'd figure out why unbatched state obs are float64
                    if arr.dtype == torch.float64:
                        arr = arr.to(torch.float32)
                    last_n_batches[k].append(arr)

        for k, v in last_n_batches.items():
            # If the observations already are batched, so that the leading dimension
            # is over the environments, insert our time dimension _after_ that
            # Otherwise, the time dimension should be the initial dimension
            if self.batched:
                last_n_batches[k] = torch.stack(v, dim=1)
            else:
                last_n_batches[k] = torch.stack(v, dim=0)

        return last_n_batches

    def process_obs(self, obs: dict) -> dict:
        """
        Args:
            obs: dictionary of observations returned from each step of simulation.
        Return:
            Dictionary with input data for `predict_action` of a policy.
        """
        batch = {}

        if "color" in self.obs_visual:
            batch["color"] = self.get_color(obs)
        if "point_cloud" in self.obs_visual:
            batch["point_cloud"] = self.get_point_cloud(obs)
        if "depth" in self.obs_visual:
            batch["depth"] = self.get_depth(obs)

        for key in self.obs_state:
            if key == "eef_pose":
                batch["eef_pose"] = self.get_eef_pose(obs)
            else:
                batch[key] = obs[key]

        return batch

    def add_task_info_to_batch(self, batch: dict, obs: dict) -> None:

        batch["task_index"] = [self.task_index]
        batch["task_description"] = obs["task_description"]
        if self.batched:
            batch["task_index"] = batch["task_index"] * self.num_envs
            batch["task_description"] = [batch["task_description"]] * self.num_envs
        batch["task_index"] = np.array(batch["task_index"], dtype=np.int64)

    def run_simulation(self, pl_module, init_noise_mode="different", init_noise=None) -> dict[str, Any]:
        """
        Runs the simulation loop while collecting simple profiling stats (mean/std)
        for various operations. Prints the results at the end if profiling is enabled.
        """

        # Create a helper to conditionally record profiling times.
        if self.enable_profiling:
            profile_data = defaultdict(list)
            def record_profile(key: str, start_time: float) -> None:
                profile_data[key].append(time.perf_counter() - start_time)
        else:
            def record_profile(key: str, start_time: float) -> None:
                pass

        success_count = 0
        total_count = 0
        total_rewards = 0
        frames = []
        episode_max_reward = []
        num_episode_batches = math.ceil(self.num_episodes / self.num_envs)

        # for same_noise_each_ep
        batch_size = 1 # # can be 1 for viz
        horizon = self.config.action_head.horizon
        action_dim = self.config.action_head.action_dim
        
        assert num_episode_batches == 1, "episodic iterations happen in run_vpl"
        for episode_i in range(num_episode_batches):
            if self.num_envs == 1:
                print(f"Running sim episode {episode_i}/{self.num_episodes}, epoch {self.epoch}")
            else:
                print(
                    f"Running sim episode {episode_i}-{episode_i+self.num_envs}/{self.num_episodes}, epoch {self.epoch}"
                )

            # Reset environment
            start_t = time.perf_counter()
            # skipping self reset
            # obs = self.reset()
            obs = self.env.reset()
            self.success = {metric: [False] * self.num_envs for metric in self.env.is_success()}
            obs["task_description"] = self.task_description
            record_profile("reset", start_t)

            # Initialize last_n_obs
            last_n_obs: Deque = deque([obs] * self.max_n_obs_history)

            # Collect initial frame (timed)
            start_t = time.perf_counter()
            frames.append(self.get_frame(obs))
            record_profile("get_frame", start_t)

            batch_episode_max_reward = np.zeros(self.num_envs)

            # set the init noise
            if init_noise_mode=="same_noise_as_first":
                noise = init_noise
            elif init_noise_mode=="same_noise_each_ep":
                noise = torch.randn((batch_size, horizon, action_dim))
            else:
                noise=None

            # Simulation steps
            steps_per_episode = int(self.num_steps / self.action_execution_steps) + 1
            for i in range(steps_per_episode):
                print(f"Step, {i}/{steps_per_episode} " f"{episode_i}/{num_episode_batches}")

                # process_n_obs
                start_t = time.perf_counter()
                batch = self.process_n_obs(last_n_obs)
                record_profile("process_n_obs", start_t)

                # add_task_info_to_batch
                start_t = time.perf_counter()
                self.add_task_info_to_batch(batch, obs)
                record_profile("add_task_info", start_t)

                # predict_action
                start_t = time.perf_counter()
                pl_module.eval()
                action = pl_module.predict_action(batch, batched=self.batched, noise=noise)
                record_profile("predict_action", start_t)

                # # Convert 10D -> xyzrpy if needed
                # if action.shape[-1] == 10 and self.config.data.get("action_convert_7d_to_10d"):
                #     # Convert 10d action to 7d
                #     pos = action[..., :3]
                #     rot = action[..., 3:9]
                #     gripper = action[..., 9:10]
                #     rot = self.rotation_transformer.inverse(rot)
                #     action = np.concatenate([pos, rot, gripper], axis=-1)
                #     out_dim = action.shape[-1]
                #     assert (
                #         out_dim == 7
                #     ), f"Expected action shape to be 7d after inverse rotation transformer, got {out_dim}"

                # If diffusion policy returns horizon of actions
                if (action.ndim == 2 and self.num_envs == 1) or (action.ndim == 3 and self.batched):
                    done = False
                    for action_step in range(self.action_execution_steps):
                        if action.ndim == 3:
                            current_action = action[:, action_step]
                        else:
                            current_action = action[action_step]

                        # env step
                        start_t = time.perf_counter()
                        # skipping self.step
                        # next_obs, reward, done = self.step(current_action)
                        next_obs, reward, done, _ = self.env.step(current_action)
                        self.is_success()
                        next_obs["task_description"] = self.task_description
                        record_profile("env_step", start_t)
                        
                        shift_observation_window_by_one(last_n_obs, next_obs)

                        # get_frame
                        start_t = time.perf_counter()
                        frames.append(self.get_frame(next_obs))
                        record_profile("get_frame", start_t)

                        total_rewards += reward if isinstance(reward, float) else sum(reward)
                        batch_episode_max_reward = np.maximum(batch_episode_max_reward, np.array(reward))

                        if not isinstance(done, bool):
                            done = done.all()
                        if done or i * self.action_execution_steps + action_step >= self.num_steps - 1:
                            break
                    if not isinstance(done, bool):
                        done = done.all()  # type: ignore
                    if done:
                        break
                else:
                    # Single-step action
                    start_t = time.perf_counter()
                    
                    # skipping self.step
                    # next_obs, reward, done = self.step(action)
                    next_obs, reward, done, _ = self.env.step(action)
                    self.is_success()
                    next_obs["task_description"] = self.task_description
                    record_profile("env_step", start_t)

                    shift_observation_window_by_one(last_n_obs, next_obs)

                    start_t = time.perf_counter()
                    frames.append(self.get_frame(next_obs))
                    record_profile("get_frame", start_t)

                    total_rewards += reward if isinstance(reward, float) else sum(reward)
                    batch_episode_max_reward = np.maximum(batch_episode_max_reward, np.array(reward))

                    if not isinstance(done, bool):
                        done = done.all()
                    if done:
                        break

                obs = deepcopy(next_obs)

            # is_success
            start_t = time.perf_counter()
            successes = self.is_success()
            record_profile("is_success", start_t)

            if not isinstance(successes, bool):
                success_count += sum(successes)
                total_count += len(successes)
            else:
                if successes:
                    success_count += 1
                total_count += 1

            episode_max_reward.extend(batch_episode_max_reward)

        # Build results
        results = {
            "epoch": self.epoch,
            "success_count": float(success_count),
            "num_episodes": total_count,
            "env_name": self.env_name,
            "total_rewards": total_rewards,
            "episode_max_reward": np.array(episode_max_reward),
        }

        # don't save videos to save upload time to wandb
        if self.config.simulation.get("save_videos", True):
            frames_np = np.array(frames)
            # if video is batched, wandb expects batch first
            if frames_np.ndim == 5:
                frames_np = rearrange(frames_np, "n b h w c -> b n h w c")
            results["video"] = frames_np

        #
        # --- Print profiling summary (mean & std in seconds) if enabled ---
        #
        if self.enable_profiling:
            print("\n=== Profiling Summary (mean ± std in seconds) ===")
            for key in sorted(profile_data.keys()):
                times = np.array(profile_data[key])
                mean_t = times.mean()
                std_t = times.std()
                print(f"{key:20s}: {mean_t:.6f} ± {std_t:.6f} (N={len(times)})")
            total_time = sum([sum(times) for times in profile_data.values()])
            print(f"{'Total':20s}: {total_time:.6f}")

        return results

    def get_color(self, obs: dict) -> npt.NDArray:
        """
        Return array of (N, H, W, 3) images where N is number of cameras.
        Values are expected to be between 0-255, to be normalized by the policy.
        """
        raise NotImplementedError

    def get_eef_pose(self, obs: dict) -> npt.NDArray:
        """
        Return eef_pose as 1D array.
        """
        raise NotImplementedError

    def get_frame(self, obs: dict) -> npt.NDArray:
        """
        Return frame for logging video to wandb.
        """
        image = self.get_color(obs).astype(np.uint8)
        image = rearrange(image, "... n h w c -> ... c (n h) w")
        return image

    def get_point_cloud(self, obs: dict) -> npt.NDArray:
        """
        Return point cloud processed the same way as training data.
        """
        raise NotImplementedError

    def get_depth(self, obs: dict) -> npt.NDArray:
        raise NotImplementedError

    def is_success(self) -> bool | list[bool]:
        raise NotImplementedError

    def reset(self) -> dict:
        """
        Return the observation dict from a reset
        """
        raise NotImplementedError

    def step(self, action: npt.NDArray) -> tuple[dict, float | np.ndarray, bool | np.ndarray]:
        """
        Return the observation dict, reward, and done from the step.
        """
        raise NotImplementedError

    @property
    def batched(self) -> bool:
        return self.num_envs > 1


def shift_observation_window_by_one(last_n_obs: Deque, new_obs: dict) -> None:
    last_n_obs.appendleft(new_obs)
    last_n_obs.pop()
