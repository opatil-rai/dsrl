"""Modified from visuomotor/mimicgen_actor.py."""

import json
import os
import random
from typing import Any, Optional
import h5py
import google.auth
import gymnasium as gym
import numpy as np
from einops import rearrange
from google.auth.exceptions import OAuthError
from google.cloud import storage
from gymnasium import spaces
from numpy import typing as npt

from visuomotor.simulation.async_vector_env import AsyncVectorEnv
from visuomotor.utils.paths import get_base_gcs_path
from vpl_simulation_base import VPLSimulationBase


class DMGEnvWrapper(gym.Env):

    def __init__(self, env: gym.Env, visual_obs_shapes: dict, dummy: bool = False):
        self.env = env
        # The dummy environment is started with no rendering because of the error described in the AsyncVectorEnv init.
        # Since the dummy environment needs to have the same action space as the non-dummy environments, we add blank
        # observations.
        self.dummy = dummy
        self.visual_obs_shapes = visual_obs_shapes
        action_shape = [self.env.action_dimension]
        action_space = spaces.Box(low=-1, high=1, shape=action_shape, dtype=np.float32)
        self.action_space = action_space

        sample_obs = self.env.get_observation()
        obs_template = {}
        for k in sample_obs:
            obs_template[k] = sample_obs[k].shape

        observation_space = spaces.Dict()
        if dummy:
            for observation_name, shape in self.visual_obs_shapes.items():
                observation_space[observation_name] = spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        for key, value in obs_template.items():
            shape = value
            if key.endswith("image"):
                min_value, max_value = 0, 1
            elif key.endswith("depth"):
                min_value, max_value = 0, 1
            elif key.endswith("voxels"):
                min_value, max_value = 0, 1
            elif key.endswith("point_cloud"):
                min_value, max_value = -10, 10
            elif key.endswith("quat"):
                min_value, max_value = -1, 1
            elif key.endswith("qpos"):
                min_value, max_value = -1, 1
            elif key.endswith("pos"):
                min_value, max_value = -1, 1
            else:
                min_value, max_value = -np.inf, np.inf

            this_space = spaces.Box(low=min_value, high=max_value, shape=shape, dtype=np.float32)
            observation_space[key] = this_space
        self.observation_space = observation_space

    def seed(self, seed: int) -> None:
        np.random.seed(seed=seed)

    def is_success(self) -> bool | list[bool]:
        success = self.env.is_success()

        return success

    def get_observation(self) -> dict:
        obs = self.env.get_observation()
        if self.dummy:
            for observation_name, shape in self.visual_obs_shapes.items():
                obs[observation_name] = np.zeros(shape)

        return obs

    def reset(self, seed: Optional[int] = None) -> dict:
        if seed is not None:
            np.random.seed(seed=seed)
        obs = self.env.reset()
        if self.dummy:
            for observation_name, shape in self.visual_obs_shapes.items():
                obs[observation_name] = np.zeros(shape)

        return obs

    def step(self, *args: Any) -> tuple[dict, float, dict, bool]:
        obs, reward, info, done = self.env.step(*args)
        if self.dummy:
            for observation_name, shape in self.visual_obs_shapes.items():
                obs[observation_name] = np.zeros(shape)

        return obs, reward, info, done

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


class DexMimicGenEnv(VPLSimulationBase):

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
        def env_fn() -> DMGEnvWrapper:
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
            env = DMGEnvWrapper(env, visual_obs_shapes=visual_obs_shapes)
            print("wrapped env")
            return env

        def dummy_env_fn() -> DMGEnvWrapper:
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
            env = DMGEnvWrapper(env, visual_obs_shapes=visual_obs_shapes, dummy=True)
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

    def get_visual_obs_shapes(self, camera_names: list[str]) -> dict:

        shapes_config = self.config.simulation.get("visual_obs_shapes")
        if shapes_config is None:
            # Default is to expect only images are used in simulation
            return {f"{camera}_image": (84, 84, 3) for camera in camera_names}

        visual_obs_shapes: dict[str, tuple] = {}
        for camera_name in camera_names:
            camera_name_image = f"{camera_name}_image"
            visual_obs_shapes[camera_name_image] = shapes_config.color_shape
            if self.use_depth_obs:
                camera_name_image = f"{camera_name}_depth"
                visual_obs_shapes[camera_name_image] = shapes_config.depth_shape

        if "point_cloud" in self.config.data.obs_keys.visual:
            visual_obs_shapes["point_cloud"] = shapes_config.point_cloud_shape
            visual_obs_shapes["voxel"] = shapes_config.voxel_shape

        return visual_obs_shapes

    def get_point_cloud(self, obs: dict) -> npt.NDArray:
        return np.array(obs["point_cloud"])

    def get_color(self, obs: dict) -> npt.NDArray:
        color = np.array([obs[image_key] for image_key in self.image_keys])
        if self.batched:
            color = rearrange(color, "n b h w c -> b n h w c")
        return color

    def reset(self) -> dict:
        # keep track of successes
        obs = self.env.reset()

        self.success = {metric: [False] * self.num_envs for metric in self.env.is_success()}
        obs["task_description"] = self.task_description
        return obs

    def is_success(self) -> bool | list[bool]:
        cur_success_metrics = self.env.is_success()
        for metric in self.success:
            if self.num_envs == 1:
                cur_success_metrics[metric] = [cur_success_metrics[metric]]
            self.success[metric] = [
                a or b for a, b in zip(self.success[metric], cur_success_metrics[metric], strict=True)
            ]
        return self.success["task"]

    def step(self, action: npt.NDArray) -> tuple[dict, float | np.ndarray, bool | np.ndarray]:

        obs, reward, done, _ = self.env.step(action)
        self.is_success()
        obs["task_description"] = self.task_description


        return obs, reward, done
