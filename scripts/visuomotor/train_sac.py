"""
TODO:
- Also consider observation history (ActionChunkWrapper)
"""

import os
import sys
import time
import argparse
import torch
import glob
import copy
import numpy as np
from random import random
from datetime import datetime
import lightning as L
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import TimeLimit, RescaleAction, FlattenObservation

# os.environ["WANDB_MODE"]="disabled"
os.environ["WANDB_INIT_TIMEOUT"]="600"
os.environ["WANDB_START_METHOD"]="thread"
import wandb
from wandb.integration.sb3 import WandbCallback
from visuomotor.models.model_registry import REGISTRY
from visuomotor.models.protocols import Policy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from scripts.visuomotor.build_dmg_env import build_dmg_env
from scripts.visuomotor.vpl_dsrl_wrappers import ActionChunkWrapper
from scripts.visuomotor.vpl_dsrl_wrappers import VPLPolicyEnvWrapper

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



def get_args():
    parser = argparse.ArgumentParser(description="Training DSRL SAC for VPL")
    parser.add_argument(
        "--ckpt-name",
        type=str,
        default="bdaii/two_arm_threading-lrenaux/diffpo-2jkfbrtz-pnb9crec:v4",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="dmg-sac",
        help="Experiment specific name to add to the artifacts",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=10,
        help="Number of DDIM inference steps",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=1,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--n_eval_envs",
        type=int,
        default=1,
        help="Number of parallel evaluation environments",
    )
    parser.add_argument(
        "--noise_factorization",
        type=str,
        default=None,
        help="If to factorize the noise matrix",
        choices=["unit", "action_dim", "sum_outer_prod", None]
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
    args = get_args()
    seed = args.seeds[0]
    ckpt_name = args.ckpt_name
    n_env = args.n_envs
    n_eval_envs = args.n_eval_envs

    np.random.seed(seed)
    torch.manual_seed(seed)
    L.seed_everything(seed)


    run = wandb.init(
        project="dsrl_env",
        name='_'.join([args.exp_name,str(datetime.now())]),
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )

    # Base policy for SAC
    policy = REGISTRY.build_policy_from_artifact_name(ckpt_name, load_swa_weights=False)    
    policy.config.simulation.num_envs_per_actor = 1 # parallelize through SubprocVecEnv
    policy.config.num_inference_steps=10 # hardcoding
    task_index = 0

    env_name = policy.config.simulation.envs[task_index]["name"]
    num_steps = policy.config.simulation.envs[task_index]["num_steps"]

    # Init simulator
    if "metadata_gcs_prefix" not in policy.config.simulation:
        policy.config.simulation["metadata_gcs_prefix"] = "mimicgen_equidiff_pc_abs/core/"
    # Prepare a picklable config dict for worker factories (avoid capturing heavy objects like policy)
    from omegaconf import OmegaConf
    cfg_dict = OmegaConf.to_container(policy.config, resolve=True)
    dataset_path = "/lam-248-lambdafs/teams/proj-compose/opatil/datasets/two_arm_threading.hdf5"

    def make_env_fn(cfg_d):
        def _thunk():
            from omegaconf import OmegaConf as _OC
            local_cfg = _OC.create(cfg_d)
            env = build_dmg_env(
                env_type="dexmimicgen",
                dataset_path=dataset_path,
                config=local_cfg,       
                abs_action=local_cfg.simulation.get("abs_action", False),
            )
            env = ActionChunkWrapper(env, local_cfg, max_episode_steps=num_steps)
            # env = TimeLimit(env, max_episode_steps=50)
            return Monitor(env)
        return _thunk

    env_fns = [make_env_fn(cfg_dict) for _ in range(n_env)]
    eval_env_fns = [make_env_fn(cfg_dict) for _ in range(n_eval_envs)]

    env = SubprocVecEnv(env_fns) if n_env > 1 else DummyVecEnv(env_fns)
    eval_env = SubprocVecEnv(eval_env_fns) if n_eval_envs > 1 else DummyVecEnv(eval_env_fns)
    setattr(eval_env, "render_mode", "rgb_array")
    eval_freq = 1
    eval_env = VecVideoRecorder(
        eval_env,
        f"videos/{run.id}/eval",
        record_video_trigger=lambda x: x % eval_freq == 0,
        video_length=300,
    )

    env = VPLPolicyEnvWrapper(env, policy, init_noise_mode=args.noise_factorization, dataset_path=dataset_path)
    eval_env = VPLPolicyEnvWrapper(eval_env, policy, init_noise_mode=args.noise_factorization, dataset_path=dataset_path)

    sac_config = {"policy": "MlpPolicy", "learning_rate": 0.001}
    model = SAC(env=env, verbose=1, tensorboard_log=f"runs/{run.id}",
                **sac_config)

    train_record_freq = 5000
    # eval callback
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
    train_record_freq = 5000
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

    total_timesteps = 1e6
    model.learn(total_timesteps=total_timesteps,
                callback=callback
                )
    run.finish()