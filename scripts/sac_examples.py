import gymnasium as gym
import argparse
import time
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
from gymnasium.wrappers import TimeLimit, FrameStackObservation, FlattenObservation
from stable_baselines3.her import HerReplayBuffer
import copy
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
import numpy as np
from gymnasium.wrappers import RescaleAction, RescaleObservation
from gym_hil.mujoco_gym_env import MAX_GRIPPER_COMMAND
from stable_baselines3.common.callbacks import BaseCallback
import glob
import os

class EvalSaveCallback(EvalCallback):
    """
    Extended EvalCallback that saves model checkpoints to wandb
    at every eval_freq step with unique names.
    """
    def __init__(self, *args, run_id, save_to_wandb=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_to_wandb = save_to_wandb
        self.run_id = run_id

    def _on_step(self) -> bool:
        result = super()._on_step()

        # Only act at eval frequency
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            step = self.num_timesteps
            ckpt_path = os.path.join(self.best_model_save_path, f"checkpoint_{step}.zip")
            self.model.save(ckpt_path)

            if self.save_to_wandb:
                artifact = wandb.Artifact(name=f"agent_{self.run_id}", type="model")
                artifact.add_file(ckpt_path, name=f"checkpoint_{step}.zip")
                wandb.log_artifact(artifact)

        return result


class VideoRecorderCallback(BaseCallback):
    """
    Logs VecVideoRecorder videos to wandb.
    Assumes videos are being saved by VecVideoRecorder.
    """
    def __init__(self, video_folder, log_freq=5000, verbose=0, wandb_prefix : str = "videos", delete_after_log: bool=True):
        super().__init__(verbose)
        self.video_folder = video_folder
        self.log_freq = log_freq
        self.wandb_prefix = wandb_prefix
        self.delete_after_log = delete_after_log

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            video_files = glob.glob(os.path.join(self.video_folder, "*.mp4"))
            if len(video_files) > 0:
                latest_video = max(video_files, key=os.path.getctime)
                wandb.log({
                    f"{self.wandb_prefix}/video": wandb.Video(latest_video, fps=4, format="mp4")
                }, step=self.num_timesteps)
            
                if self.delete_after_log:
                    os.remove(latest_video)
        return True

class RewardGripperClosingWrapper(gym.Wrapper):
    """
    this is for gym_hil to encourage closing gripper a bit
    """
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = self.unwrapped.get_gripper_pose() / MAX_GRIPPER_COMMAND
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # info["discrete_penalty"] = 0.0
        # if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
        #     action[-1] > 0.5 and self.last_gripper_pos < 0.1
        # ):
        #     # info["discrete_penalty"] = self.penalty
        #     reward -= self.penalty
        reward += float(self.last_gripper_pos > 0.5) * 0.01

        self.last_gripper_pos = self.unwrapped.get_gripper_pose() / MAX_GRIPPER_COMMAND
        return observation, reward, terminated, truncated, info

class GripperActionWrapper(gym.ActionWrapper):
    """
    Wrapper that processes gripper control commands.

    This wrapper quantizes and processes gripper commands, adding a sleep time between
    consecutive gripper actions to prevent rapid toggling.
    """

    def __init__(self, env, quantization_threshold: float = 0.2, gripper_sleep: float = 0.0):
        """
        Initialize the gripper action wrapper.

        Args:
            env: The environment to wrap.
            quantization_threshold: Threshold below which gripper commands are quantized to zero.
            gripper_sleep: Minimum time in seconds between consecutive gripper commands.
        """
        super().__init__(env)
        self.quantization_threshold = quantization_threshold
        self.gripper_sleep = gripper_sleep
        self.last_gripper_action_time = 0.0
        self.last_gripper_action = None

    def action(self, action):
        """
        Process gripper commands in the action.

        Args:
            action: The original action from the agent.

        Returns:
            Modified action with processed gripper command.
        """
        if self.gripper_sleep > 0.0:
            if (
                self.last_gripper_action is not None
                and time.perf_counter() - self.last_gripper_action_time < self.gripper_sleep
            ):
                action[-1] = self.last_gripper_action
            else:
                self.last_gripper_action_time = time.perf_counter()
                self.last_gripper_action = action[-1]

        gripper_command = action[-1]
        # Gripper actions are between 0, 2
        # we want to quantize them to -1, 0 or 1
        gripper_command = gripper_command - 1.0

        if self.quantization_threshold is not None:
            # Quantize gripper command to -1, 0 or 1
            gripper_command = (
                np.sign(gripper_command) if abs(gripper_command) > self.quantization_threshold else 0.0
            )

            gripper_command = gripper_command + 1

            action[-1] = gripper_command

        return action

    def reset(self, **kwargs):
        """
        Reset the gripper action tracking.

        Args:
            **kwargs: Keyword arguments passed to the wrapped environment's reset.

        Returns:
            The initial observation and info.
        """
        obs, info = super().reset(**kwargs)
        self.last_gripper_action_time = 0.0
        self.last_gripper_action = None
        return obs, info


class PushTPoseReward(gym.Wrapper):
    """
    Changes reward to be based on goal pose MSE instead of coverage
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def scale_array(self, arr, in_mins, in_maxs, out_mins, out_maxs):
        """
        Linearly scales an N-dimensional numpy array dimension-wise,
        using per-dimension input and output ranges.

        Parameters
        ----------
        arr : np.ndarray
            Array of shape (..., N).
        in_mins : array-like
            Minimum values for each input dimension (length N).
        in_maxs : array-like
            Maximum values for each input dimension (length N).
        out_mins : array-like
            Minimum values for each output dimension (length N).
        out_maxs : array-like
            Maximum values for each output dimension (length N).

        Returns
        -------
        np.ndarray
            Scaled array of same shape as arr.
        """
        arr = np.asarray(arr, dtype=float)
        in_mins, in_maxs = np.asarray(in_mins), np.asarray(in_maxs)
        out_mins, out_maxs = np.asarray(out_mins), np.asarray(out_maxs)

        denom = np.where(in_maxs - in_mins == 0, 1, in_maxs - in_mins)
        normed = (arr - in_mins) / denom  # scale to [0,1] per dim
        scaled = normed * (out_maxs - out_mins) + out_mins
        return scaled

    def step(self, action):
        obs_next, old_reward, terminated, truncated, info = self.env.step(action)

        scaled_goal_pose = self.scale_array(info['goal_pose'], np.array([0,0,0]), np.array([512,512,2*np.pi]), np.array([-1,-1,-1]), np.array([1,1,1]))
        scaled_block_pose = self.scale_array(info['block_pose'], np.array([0,0,0]), np.array([512,512,2*np.pi]), np.array([-1,-1,-1]), np.array([1,1,1]))

        pose_reward = 0
        pose_reward += np.linalg.norm(scaled_goal_pose[:2] - scaled_block_pose[:2])
        d_direct = abs(scaled_goal_pose[2] - scaled_block_pose[2])
        pose_reward += min(d_direct, (2*np.pi) - d_direct)
        pose_reward *= -1

        reward = old_reward + 0.01*pose_reward

        return obs_next, reward, terminated, truncated, info

class RewardAsPotentialWrapper(gym.Wrapper):
    """
    Reward shaping where the potential function Phi(s) is the base env's reward.
    Shaped reward: r' = gamma * Phi(s') - Phi(s).
    The base reward is discarded.
    """
    def __init__(self, env: gym.Env, gamma: float = 0.99):
        super().__init__(env)
        self.gamma = gamma
        self._last_potential = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_potential = None
        return obs, info

    def step(self, action):
        obs_next, base_reward, terminated, truncated, info = self.env.step(action)

        if self._last_potential != None:
            phi_prev = self._last_potential 
        else:
            phi_prev = base_reward 
        phi_next = base_reward
        shaped_reward = self.gamma * phi_next - phi_prev
        reward = shaped_reward # -1 is sparse reward for all actions.

        self._last_potential = phi_next
        return obs_next, reward, terminated, truncated, info

def task_import(task):
    if task == "pusht":
        import gym_pusht
    elif task =="pusht_latent":
        import gym_pusht
    elif task == "gym_hil":
        import gym_hil
    elif "fetch" in task:
        import gymnasium_robotics


def make_env(video_folder, record_trigger, other_config):
    # ObservationAction Normalizing wrapper
    if task == "pusht":
        env = gym.make(other_config["env_name"],render_mode="rgb_array")

        obs_min = np.ones([5])*-1
        obs_max = np.ones([5])
        action_min = np.ones([2])*-1
        action_max = np.ones([2])
        
        # linearlly normalize obs/action to [-1,1]
        env = RescaleAction(env, min_action=action_min, max_action=action_max)
        env = RescaleObservation(env, min_obs=obs_min, max_obs=obs_max)

        # turn the dense reward into a potential shaped reward function
        # env = RewardAsPotentialWrapper(env)
        env = PushTPoseReward(env)

        # stack obs
        env = FrameStackObservation(env, stack_size=2)
        # time limit
        env = TimeLimit(env, max_episode_steps=100)
    elif task == "pusht_latent":
        # Check device is available
        device = "cuda"
        if video_folder == "eval":
            reset_state = np.array([314, 201, 187.21077193, 275.01629149, np.pi / 4.0])
            options = {"reset_to_state": reset_state}
            gym_reset_seed = 1234522325 # or None for no fixed seed
        elif video_folder == "train":
            # Reset env, save specific state
            options = None # No hard-coded reset state
            gym_reset_seed = None
            # reset_state = np.array([314, 201, 187.21077193, 275.01629149, np.pi / 4.0])
            # options = {"reset_to_state": reset_state}
            # gym_reset_seed = 1234522325 # or None for no fixed seed
        from lerobot_dsrl import generate_steerable_diffpo_pusht_gym_env
        env = generate_steerable_diffpo_pusht_gym_env(device=device, options=options, seed=gym_reset_seed, desired_action_dim=other_config["desired_action_dim"],
                                                      num_inference_steps=100)
        # make action min/max -1,1 based on desired_action_dim
        action_min = np.ones([other_config["desired_action_dim"]])*-1
        action_max = np.ones([other_config["desired_action_dim"]])
        # linearlly normalize obs/action to [-1,1]
        env = RescaleAction(env, min_action=action_min, max_action=action_max)
        env = TimeLimit(env, max_episode_steps=50)
    elif "fetch" in task:
        env = gym.make(other_config["env_name"],render_mode="rgb_array", max_episode_steps=100)
        env = FlattenObservation(env)

    elif task in ["gym_hil"]:
        env = gym.make(other_config["env_name"],reward_type="dense",render_mode="rgb_array",random_block_position=False)
        # flatten the obs
        env = FlattenObservation(env)

        # Change obs bounds to be empircally bounded for normalization
        raw_obs_min =  np.array([-3.05651307e+00, -1.81523693e+00, -2.97463274e+00, -3.13562918e+00,
 -2.91523480e+00, -2.34676152e-02, -1.40393531e+00, -2.63995972e+01,
 -1.33440380e+01, -3.00091286e+01, -1.67922821e+01, -1.03519049e+01,
 -1.20307131e+01, -9.53336048e+00,  0.00000000e+00, -1.63084328e-01,
 -4.85410154e-01, -1.95764005e-02,  1.68003678e-01, -4.58048105e-01,
  1.44774178e-02])
        raw_obs_max = np.array([ 3.04035187e+00,  1.82339013e+00,  2.98502374e+00, -1.86848361e-02,
  2.90192127e+00,  3.77346945e+00,  2.94248080e+00,  3.29576530e+01,
  1.42244177e+01,  2.69660110e+01,  1.06645355e+01,  1.15072269e+01,
  1.11341019e+01,  9.48305702e+00,  2.55000000e+02,  8.64770710e-01,
  4.66381371e-01,  8.33268702e-01,  6.98901296e-01,  3.61549705e-01,
  7.84597993e-02])
        env.observation_space = gym.spaces.Box(
            low=raw_obs_min,
            high=raw_obs_max,
            dtype=np.float32
        )
        obs_min = np.ones([21])*-1
        obs_max = np.ones([21])
        # TODO: the bounds of observation space are -inf,inf so can't easily be rescaled, come back to this?
        env = RescaleObservation(env, min_obs=obs_min, max_obs=obs_max)

        # action_min = np.ones([7])*-1
        # action_max = np.ones([7])
        # linearlly normalize obs/action to [-1,1]
        # env = RescaleAction(env, min_action=action_min, max_action=action_max)


        # Remove orientation control / scale down action magnitude
        from gym_hil.wrappers.hil_wrappers import EEActionWrapper, DEFAULT_EE_STEP_SIZE
        env = RewardGripperClosingWrapper(env)
        # env = GripperActionWrapper(env)
        ee_action_step_size = {k: v for k, v in DEFAULT_EE_STEP_SIZE.items()}
        env = EEActionWrapper(env, ee_action_step_size=ee_action_step_size,use_gripper=True)

        # stack obs
        # env = FrameStackObservation(env, stack_size=2)
        # time limit
        env = TimeLimit(env, max_episode_steps=50)
    else:
        env = gym.make(other_config["env_name"],render_mode="rgb_array")

    env = Monitor(env)  # record stats such as returns

    # vectorizing time and video stuff
    env = DummyVecEnv([lambda: env])
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}/{video_folder}",
        record_video_trigger=lambda x: x % record_trigger == 0,
        video_length=200,
    )
    return env



def get_args():
    parser = argparse.ArgumentParser(description="Training DSRL SAC on push-T")
    parser.add_argument(
        "--task_name",
        type=str,
        default="pusht_latent",
        help="Lerobot environment name (e.g., pusht_latent).",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="dsrl-sac",
        help="Experiment specific name to add to the artifacts",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="List of random seeds (space separated). Example: --seed 0 42 1337",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # stats_window_size: default (100), determines how many episodes to go over for reporting evals like returns, success, etc.
    args = get_args()
    cfgs = {
        "sac":
        {
            "pendulum":{
                "policy": "MlpPolicy",
                "learning_rate": 0.001,
            },
            "hopper":{
                "policy": "MlpPolicy",
                "learning_rate": 0.001,
            },
            "pusht":{
                "policy": "MlpPolicy",
                "learning_rate": 0.001,
            },
            "pusht_latent":
            {
                "policy": "MlpPolicy",
                "learning_rate":0.001,
                "policy_kwargs" : {"net_arch": [512,512]}
            },
            "gym_hil":{
                "policy": "MlpPolicy",
                "learning_rate": 0.001,
            },
            "fetch_reach":{
                "policy": "MlpPolicy",
                "learning_rate": 0.001,
            },
            "fetch_pick" :{
                "policy": "MlpPolicy",
                # "replay_buffer_class": HerReplayBuffer,
                # "replay_buffer_kwargs": {"n_sampled_goal":4,
                #                          "goal_selection_strategy":"future"},
                "batch_size": 512,
                "buffer_size":1_000_000,
                "learning_rate": 1e-3,
                "gamma": 0.95,
                "tau": 	0.05,
                "ent_coef":"auto",
                "train_freq": 1,
                "gradient_steps": 1,
                "seed": 42,
                
            },
        },
        "other":
        {
            "pendulum":{
                "total_timesteps": 20_000,
                "env_name": "Pendulum-v1",
            },
            "hopper":{
                "total_timesteps": 1e6,
                "env_name": "Hopper-v5",
            },
            "pusht":{
                "total_timesteps": 1e6,
                "env_name": "gym_pusht/PushT-v0",
            },
            "pusht_latent":{
                "total_timesteps": 1e6,
                "desired_action_dim":2
            },
            "gym_hil":{
                "total_timesteps": 1e6,
                "env_name": "gym_hil/PandaPickCubeBase-v0",
            },
            "fetch_reach":{
                "total_timesteps": 1e6,
                "env_name": "FetchReachDense-v4",
            },
            "fetch_pick" :{
                "total_timesteps": 1e6,
                "env_name": "FetchPickAndPlaceDense-v4",
            },
        }

    }

    task = args.task_name
    task_import(task)
    sac_config = cfgs["sac"][task]
    other_config = cfgs["other"][task]

    run = wandb.init(
        project="dsrl_sb3",
        name='_'.join([args.exp_name,str(datetime.now())]),
        config=sac_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )

    train_record_freq = 2000
    env = make_env("train", record_trigger=train_record_freq, other_config=other_config) 
    eval_env = make_env("eval", record_trigger=1,other_config=other_config) # trigger on every step of eval, eval recording happens at eval_freq
    model = SAC(env=env, verbose=1, tensorboard_log=f"runs/{run.id}", **sac_config)

    # CALLBACKS
    # eval callback
    eval_freq = 5000
    eval_callback = EvalSaveCallback(
        eval_env,
        best_model_save_path="./logs/checkpoints",
        log_path="./logs/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,  # rendering in eval slows things down a lot
        save_to_wandb=True,
        run_id = run.id
    )
    # wandb callback
    wandb_callback = WandbCallback(
                    gradient_save_freq=train_record_freq,
                    model_save_path=f"models/{run.id}",
                    model_save_freq=eval_freq,
                    verbose=2,
                    )

    train_video_callback = VideoRecorderCallback(
        video_folder=f"videos/{run.id}/train",
        log_freq=train_record_freq, # same as rate we record training videos
        wandb_prefix="rollout", # logging to rollout instead of train lets the videos be next to monitored train returns
        delete_after_log=True
    )

    eval_video_callback = VideoRecorderCallback(
        video_folder=f"videos/{run.id}/eval",
        log_freq=eval_freq, # same as rate we record eval videos
        wandb_prefix="eval",
        delete_after_log=True
    )


    # all callbacks together
    callback = CallbackList([eval_callback, wandb_callback, train_video_callback, eval_video_callback])

    model.learn(total_timesteps=other_config["total_timesteps"],
                callback=callback
                )
    run.finish()