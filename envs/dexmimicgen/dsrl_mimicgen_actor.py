import math
import time
import torch
from collections import defaultdict, deque
from copy import deepcopy
from typing import Any, Deque

import numpy as np
from einops import rearrange
# from visuomotor.data.rotation_transformer import RotationTransformer
from visuomotor.simulation.mimicgen_actor import MimicgenActor
from visuomotor.simulation.simulation_base import shift_observation_window_by_one



class DSRLMimicgenActor(MimicgenActor):
    def run_simulation(self, init_noise_mode="different", init_noise=None) -> dict[str, Any]:
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
        
        assert num_episode_batches == 1, "i did not write this part"
        for episode_i in range(num_episode_batches):
            # if self.num_envs == 1:
            #     print(f"Running sim episode {episode_i}/{self.num_episodes}, epoch {self.epoch}")
            # else:
            #     print(
            #         f"Running sim episode {episode_i}-{episode_i+self.num_envs}/{self.num_episodes}, epoch {self.epoch}"
            #     )
            assert self.num_envs == 1, "only support 1 for now"

            # Reset environment
            start_t = time.perf_counter()
            obs = self.reset()
            record_profile("reset", start_t)

            # Initializev last_n_obs
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
                self.pl_module.eval()
                action = self.pl_module.predict_action(batch, batched=self.batched, noise=noise)
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
                        next_obs, reward, done = self.step(current_action)
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
                    next_obs, reward, done = self.step(action)
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


