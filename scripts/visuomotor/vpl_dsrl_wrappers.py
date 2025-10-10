import os
os.environ["MUJOCO_GL"] = "egl"
import torch
import time
import wandb
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from numpy import typing as npt
from typing import Deque
from einops import rearrange
from collections import defaultdict
from collections import deque as _dq

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from stable_baselines3.common.vec_env import VecEnvWrapper
from visuomotor.data.utils import create_key_to_history_mapping
from scripts.visuomotor.build_dmg_env import get_env_metadata_from_dataset

IGNORE_SAC = bool(os.getenv("IGNORE_SAC", 0)) # ignores sac outputs for debugging

class ActionChunkWrapper(gym.Env):
    """Execute a chunk (sequence) of low-level actions in one step.
    The exposed action space is the underlying env.action_space repeated 'act_steps' times.
    """
    def __init__(self, env, cfg):
        self.env = env
        self.act_steps = int(cfg.data.horizon)

        base_low = env.action_space.low
        base_high = env.action_space.high
        self.action_space = spaces.Box(
            low=np.tile(base_low, self.act_steps),
            high=np.tile(base_high, self.act_steps),
            dtype=np.float32,
        )
        # Use underlying observation space (don't assume cfg.obs_dim)
        self.observation_space = env.observation_space
        self.count = 0

        self.num_steps = cfg.simulation.envs[0].get("num_steps", 400)
        self.num_episodes = cfg.simulation.get("num_episodes", 10) // cfg.simulation.get(
            "ray_parallel_actors", 1
        )
        self.action_execution_steps = cfg.simulation.get("action_execution_steps", 1)


    def reset(self, seed=None):
        obs = self.env.reset(seed=seed)
        self.success = {metric: [False] for metric in self.env.is_success()}
        self.count = 0
        return obs, {}

    def step(self, action):
        if len(action.shape) == 1:
            action = action.reshape(self.act_steps, -1)
        obs_ = []
        reward_ = []
        done_ = []
        info_ = []
        done_i = False
        
        for i in range(self.action_execution_steps):
            self.count += 1
            obs_i, reward_i, done_i, info_i = self.env.step(action[i])
            self.is_success()
            obs_.append(obs_i)
            reward_.append(reward_i)
            done_.append(done_i)
            info_.append(info_i)

        obs = obs_[-1]
        reward = sum(reward_)
        done = np.max(done_)
        info = info_[-1]
        info["is_success"] = self.is_success() # override the is_success set by EnvRobosuite
        if self.count >= self.num_steps:
            done = True # this is the first place in the env chain where done can be set to True
        return obs, reward, done, False, info # converts 4-tuple to 5-tuple expected of gym Envs

    def render(self):
        return self.env.render()

    def close(self):
        return

    def is_success(self) -> bool | list[bool]:
        cur_success_metrics = self.env.is_success()
        for metric in self.success:
            cur_success_metrics[metric] = [cur_success_metrics[metric]]
            self.success[metric] = [
                a or b for a, b in zip(self.success[metric], cur_success_metrics[metric], strict=True)
            ]
        return self.success["task"]


def timed(name: str, cuda: bool = True):
    """Decorator to measure execution time of instance methods and store on self._timings.
    Args:
        name: key under self._timings to accumulate (last value stored; could extend to avg).
        cuda: synchronize CUDA for more accur0ate timing when GPU present.
    """
    def _decorator(fn):
        def _wrapped(self, *args, **kwargs):
            if not hasattr(self, '_timings'):
                self._timings = {}
            if cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = fn(self, *args, **kwargs)
            if cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            self._timings[name] = dt
            return result
        return _wrapped
    return _decorator

TIMING_ENABLED: bool = False  # Global switch for all timing & logging. Set False to disable all timing overhead & wandb logs.
TIMING_LOG_INTERVAL: int = 10  # Log timing metrics every N calls to step_async. Increase to reduce overhead / noise.


class VPLPolicyEnvWrapper(VecEnvWrapper):
    """Essentially has all the methods from vpl_simulation_base and dexmimicgen_env"""

    def __init__(self, env, base_policy, init_noise_mode="unit", dataset_path=None):
        super().__init__(env)
        self.config = base_policy.config
        self.action_horizon = self.config.data.horizon
        self.action_dim = self.config.action_head.action_dim
        self.init_noise_mode = init_noise_mode  # controls initial noise dimensionality fed to diffusion
        self.env = env
        self.num_envs = self.env.num_envs # multi-processed envs
        self.device = "cuda:0"
        self.base_policy = base_policy
        self.base_policy.eval()
        self.obs = None  # stores the last obs
        self._step_counter = 0

        # define action space
        mag = 2 # for sufficient support on a gaussian
        if init_noise_mode == 'unit':
            low = -mag * np.ones(1)
            high = mag * np.ones(1)
        elif init_noise_mode == 'action_dim':
            low = -mag * np.ones(self.action_dim)
            high = mag * np.ones(self.action_dim)
        else:  # full flattened sequence
            low = -mag * np.ones(self.action_dim * self.action_horizon)
            high = mag * np.ones(self.action_dim * self.action_horizon)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # define obs space
        self.obs_dim = self.config.action_head.input_dim
        self.observation_space = spaces.Box(
            # low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # ----  mirrors SimulationBase ----
        self.task_index = 0
        self.run_check_observation_space = True
        self.epoch = 99
        self.env_type = "dexmimicgen"

        self.metadata_gcs_prefix = self.config.simulation.metadata_gcs_prefix
        self.abs_action = self.config.simulation.get("abs_action", False)
        self.env_name = self.config.simulation.envs[self.task_index]["name"]
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

        # --- mirrors mimicgen env setup ---
        env_metadata = get_env_metadata_from_dataset(dataset_path=dataset_path)
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
        visual_obs_shapes = self.get_visual_obs_shapes(env_metadata["env_kwargs"]["camera_names"])
        self.image_keys = sorted([f"{camera_name}_image" for camera_name in env_metadata["env_kwargs"]["camera_names"]])  

        if IGNORE_SAC:
            print(">>> IGNORE_SAC is set; SAC actions will be ignored and random noise passed to VPL")

    @timed('inference_s')
    def step_async(self, actions):
        obs = self.obs
        # reset per-cycle timings (env & encode measured elsewhere)
        if not hasattr(self, '_timings'):
            self._timings = {}
        self._timings.setdefault('env_step_s', 0.0)
        self._timings.setdefault('encode_obs_s', 0.0)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        n_envs = actions.shape[0]
        mode = self.init_noise_mode
        if mode == 'unit':
            actions = actions.view(n_envs, 1, 1).repeat(1, self.action_horizon, self.action_dim)
        elif mode == 'action_dim':
            actions = actions.view(n_envs, 1, self.action_dim).repeat(1, self.action_horizon, 1)
        else:  # full flattened or already shaped
            if actions.ndim == 2:
                actions = actions.view(n_envs, self.action_horizon, self.action_dim)

        if IGNORE_SAC:
            actions = None # for testing
        diffused_actions = self.predict_vpl_action(obs, noise=actions)
        t_dispatch0 = time.perf_counter()
        self.venv.step_async(diffused_actions)
        self._timings['dispatch_overhead_s'] = time.perf_counter() - t_dispatch0

    def step_wait(self):
        t_env0 = time.perf_counter()
        obs, rewards, dones, infos = self.venv.step_wait()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._timings['env_step_s'] = time.perf_counter() - t_env0
        obs = self.encode_obs(obs, batched=self.batched) # encode for diffpo (timed via decorator)
        self.obs = obs
        obs_out = self.fuse_feats(obs) # for SAC
        obs_out = obs_out.detach().cpu().numpy()

        # --- wandb logging ---
        self._step_counter += 1
        if TIMING_ENABLED and wandb is not None and wandb.run is not None and (self._step_counter % TIMING_LOG_INTERVAL == 0):
            wandb.log({
                'timing/inference_s': self._timings.get('inference_s', 0.0),
                'timing/env_step_s': self._timings.get('env_step_s', 0.0),
                'timing/encode_obs_s': self._timings.get('encode_obs_s', 0.0),
                'timing/predict_vpl_action_s': self._timings.get('predict_vpl_action_s', 0.0),
                'timing/dispatch_overhead_s': self._timings.get('dispatch_overhead_s', 0.0),
                'timing/step': self._step_counter,
            })
        for done_idx, done in enumerate(dones):
            if done:
                infos[done_idx]['terminal_observation'] = obs_out[done_idx]
        return obs_out, rewards, dones, infos
        # return np.expand_dims(np.array([1]*self.num_envs), axis=1), rewards, dones, infos
    

    def reset(self):
        obs = self.venv.reset()
        obs = self.encode_obs(obs, batched=self.batched) # encode for diffpo
        self.obs = obs
        obs_out = self.fuse_feats(obs) # for SAC
        if TIMING_ENABLED and wandb.run is not None and hasattr(self, '_timings'):
            wandb.log({'timing/encode_obs_s': self._timings.get('encode_obs_s', 0.0), 'timing/step': self._step_counter})
        # return np.expand_dims(np.array([1]*self.num_envs), axis=1)
        return obs_out.detach().cpu().numpy()
 
    # Modified from MimicgenActor
    @property
    def batched(self) -> bool:
        # return self.env.num_envs > 1
        return True # always batched for vpl as even when n_envs=1, there is a batch dim

    def fuse_feats(self, features: dict[str, npt.NDArray | torch.Tensor]) -> npt.NDArray:
        return self.base_policy.head._fuse_features(features)

    @torch.no_grad()
    @timed('encode_obs_s')
    def encode_obs(self, obs: dict[str, npt.NDArray | torch.Tensor], batched: bool = False) -> npt.NDArray:
        # If a single observation dict is provided, expand to history deque
        bp = self.base_policy
        last_n = _dq([obs] * self.max_n_obs_history)
        batch = self.process_n_obs(last_n)
        # Attach task info if available
        # if hasattr(self, 'add_task_info_to_batch'):
        #     self.add_task_info_to_batch(batch, obs if isinstance(obs, dict) else {})
        if "color" in batch:
            batch["color"] = bp.reshape_color_input(batch["color"], batched)
        batch = bp.cast_inputs_to_appropriate_type(batch, batched)
        batch = bp.prepare_inputs(batch)
        features = bp.encode(batch)
        return features

    @torch.no_grad()
    @timed('predict_vpl_action_s')
    def predict_vpl_action(self, features: dict[str, npt.NDArray | torch.Tensor], noise=None) -> npt.NDArray:
        # additionally handle batch inference
        bp = self.base_policy
        action = bp.head.predict_action(features, noise=noise)
        if bp.normalizer is not None:
            action = bp.normalizer["action"].unnormalize(action)
        action = action.cpu().numpy()
        return action


    # ---- Ported from MimicgenEnv in VPL ----
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