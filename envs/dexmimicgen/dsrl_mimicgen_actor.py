"""For deterministic object placement, modify dexmimicgen environments to collapse the sampling range
and set the initialization noise to None"""
import json
import os
import math
import time
import torch
import h5py
from collections import defaultdict, deque
from copy import deepcopy
from typing import Any, Deque
import random
import gymnasium as gym
import numpy as np
from einops import rearrange

from visuomotor.simulation.async_vector_env import AsyncVectorEnv
# from visuomotor.data.rotation_transformer import RotationTransformer
from visuomotor.simulation.mimicgen_actor import MimicgenActor, MimicgenEnvWrapper
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



    def setup_env(self, base_bucket: str) -> gym.Env:
        # This import is required to register the mimicgen and dexmimicgen environments
        if self.env_type == "mimicgen":
            import mimicgen  # noqa: F401
        elif self.env_type == "dexmimicgen":
            import dexmimicgen  # noqa: F401
        else:
            raise ValueError(
                f"Wrong env_type in simulation config: {self.env_type}, only `mimicgen` and `dexmimicgen` are available"
            )
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.obs_utils as ObsUtils

        # change from vpl
        env_metadata = get_env_metadata_from_dataset(dataset_path="/lam-248-lambdafs/teams/proj-compose/opatil/datasets/two_arm_threading.hdf5")
        self.task_description = "dexmimicgen"

        obs_keys = self.config.data.obs_keys.visual
        self.use_image_obs = True if "color" in obs_keys or "point_cloud" in obs_keys else False
        # We need depth in order to generate the point clouds
        self.use_depth_obs = True if "depth" in obs_keys or "point_cloud" in obs_keys else False

        env_metadata["env_kwargs"]["camera_depths"] = self.use_depth_obs
        env_metadata["env_kwargs"]["camera_names"] = self.config.simulation.get(
            "sim_cameras", ["agentview", "robot0_eye_in_hand"]
        )
        env_metadata["env_kwargs"]["use_object_obs"] = False
        if self.abs_action:
            env_metadata["env_kwargs"]["controller_configs"]["control_delta"] = False
            env_metadata["env_kwargs"]["controller_configs"]["input_type"] = "absolute"

        # This key in the dataset metadata breaks robomimic so directly removing it
        if "env_lang" in env_metadata["env_kwargs"]:
            env_metadata["env_kwargs"].pop("env_lang")

        if self.env_type == "mimicgen" and "point_cloud" in self.obs_visual:
            from visuomotor.simulation.pointcloud_robomimic_env import PointCloudRobomimicEnv

            env_class = PointCloudRobomimicEnv
            # we cannot run_check_observation_space if using point_cloud because of discrepancy in
            # `observation_spaces` between the env and the dummy_env making a check fail even though the sim
            # will run fine
            self.run_check_observation_space = False
        else:
            env_class = None

        visual_obs_shapes = self.get_visual_obs_shapes(env_metadata["env_kwargs"]["camera_names"])

        # load data processing version which matches the input data
        def env_fn() -> MimicgenEnvWrapper:
            print("about to create env")
            env = EnvUtils.create_env_for_data_processing(
                env_class=env_class,
                env_meta=env_metadata,
                camera_names=env_metadata["env_kwargs"]["camera_names"],
                camera_height=env_metadata["env_kwargs"]["camera_heights"],
                camera_width=env_metadata["env_kwargs"]["camera_widths"],
                reward_shaping=env_metadata["env_kwargs"]["reward_shaping"],
                render=False,
                render_offscreen=True,
                use_image_obs=self.use_image_obs,
                use_depth_obs=self.use_depth_obs,
            )
            print("created env in worker")
            env = MimicgenEnvWrapper(env, visual_obs_shapes=visual_obs_shapes)
            print("wrapped env")
            return env

        def dummy_env_fn() -> MimicgenEnvWrapper:
            print("about to create env in dummy")
            env = EnvUtils.create_env_for_data_processing(
                env_class=env_class,
                env_meta=env_metadata,
                camera_names=env_metadata["env_kwargs"]["camera_names"],
                camera_height=env_metadata["env_kwargs"]["camera_heights"],
                camera_width=env_metadata["env_kwargs"]["camera_widths"],
                reward_shaping=env_metadata["env_kwargs"]["reward_shaping"],
                render=False,
                render_offscreen=False,
                use_image_obs=False,
                use_depth_obs=False,
            )
            env = MimicgenEnvWrapper(env, visual_obs_shapes=visual_obs_shapes, dummy=True)
            return env

        self.image_keys = sorted([f"{camera_name}_image" for camera_name in env_metadata["env_kwargs"]["camera_names"]])

        spec = dict(
            obs=dict(
                low_dim=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                rgb=self.image_keys,
            ),
        )
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=spec)

        print("starting async envs")
        if self.batched:
            env = AsyncVectorEnv(
                [env_fn] * self.num_envs,
                dummy_env_fn=dummy_env_fn,
                run_check_observation_space=self.run_check_observation_space,
            )
        else:
            env = env_fn()

        print("inited environments")
        # We have to manually set the seed here, otherwise each sim worker will have the same seed
        env.seed(random.randint(0, 2**31))

        return env

def get_env_metadata_from_dataset(dataset_path, ds_format="robomimic"):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    if ds_format == "robomimic":
        env_meta = json.loads(f["data"].attrs["env_args"])
    else:
        raise ValueError
    f.close()
    return env_meta
