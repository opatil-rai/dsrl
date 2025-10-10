"""
TODO: Also consider observation history (ActionChunkWrapper)
"""

import os
os.environ["MUJOCO_GL"] = "egl"
import sys
import time
import argparse
import torch
import glob
import copy
import yaml
import numpy as np
from random import random
from datetime import datetime
import lightning as L
from types import SimpleNamespace as _SNS
from gymnasium.wrappers import TimeLimit, RescaleAction, FlattenObservation

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.callbacks import BaseCallback

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
        "--task-name",
        type=str,
        default="two_arm_threading",
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


pretrained_policies = {
    "two_arm_drawer_cleanup": r"bdaii/two_arm_drawer_cleanup-lrenaux/diffpo-infgcmfb-8yhbzmqy:v4",
    "two_arm_lift_tray": r"bdaii/two_arm_lift_tray-lrenaux/diffpo-p2xvpiry-vf24t7f7:v4",
    "two_arm_transport": r"bdaii/two_arm_transport-lrenaux/diffpo-0gvai0kn-gjidrqcr:v4",
    "two_arm_three_piece_assembly": r"bdaii/two_arm_three_piece_assembly-lrenaux/diffpo-gu07owxr-aqppcwd5:v4",
    "two_arm_threading": r"bdaii/two_arm_threading-lrenaux/diffpo-e8jwrx3m-uay94zwk:v4"
}

if __name__ == "__main__":
    args = get_args()
    seed = args.seeds[0]
    task_name = args.task_name
    ckpt_name = pretrained_policies[task_name]
    print(">>> Using checkpoint:", ckpt_name)
    n_env = args.n_envs
    n_eval_envs = args.n_eval_envs

    np.random.seed(seed)
    torch.manual_seed(seed)
    L.seed_everything(seed)


    run = wandb.init(
        project="dsrl_env",
        name='_'.join([args.task_name, args.exp_name, str(datetime.now())]),
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
    dataset_path = f"/lam-248-lambdafs/teams/proj-compose/opatil/datasets/{task_name}.hdf5"

    # --- Use SB3 make_vec_env instead of manual env_fns lists ---
    def _single_env_factory():
        from omegaconf import OmegaConf as _OC
        local_cfg = _OC.create(cfg_dict)
        _env = build_dmg_env(
            env_type="dexmimicgen",
            dataset_path=dataset_path,
            config=local_cfg,
        )
        _env = ActionChunkWrapper(_env, local_cfg)
        return _env

    # make_vec_env accepts a callable that returns an Env; choose VecEnv class based on n_env
    vec_cls_train = SubprocVecEnv if n_env > 1 else DummyVecEnv
    env = make_vec_env(
        _single_env_factory,
        n_envs=n_env,
        seed=seed,
        vec_env_cls=vec_cls_train,
        monitor_dir=f"runs/{run.id}/monitor/train",
    )

    vec_cls_eval = SubprocVecEnv if n_eval_envs > 1 else DummyVecEnv
    eval_env = make_vec_env(
        _single_env_factory,
        n_envs=n_eval_envs,
        seed=seed + 1619,  # offset seed so eval workers differ
        vec_env_cls=vec_cls_eval,
        monitor_dir=f"runs/{run.id}/monitor/eval",
    )
    setattr(eval_env, "render_mode", "rgb_array")
    eval_freq = 1000
    eval_env = VecVideoRecorder(
        eval_env,
        f"videos/{run.id}/eval",
        record_video_trigger=lambda x: x % eval_freq == 0,
        video_length=300,
    )

    env = VPLPolicyEnvWrapper(env, policy, init_noise_mode=args.noise_factorization, dataset_path=dataset_path)
    eval_env = VPLPolicyEnvWrapper(eval_env, policy, init_noise_mode=args.noise_factorization, dataset_path=dataset_path)

    # Build cfg namespace for user-specified SAC parameters
    cfg_path = os.path.join(os.path.dirname(__file__), "sac_config.yaml")
    with open(cfg_path, "r") as f:
        sac_cfg = yaml.safe_load(f)["train"]
    train_cfg = _SNS(**sac_cfg)
    logdir = f"runs/{run.id}"
    cfg = _SNS(train=train_cfg, logdir=logdir)

    # Policy architecture (shared for actor and critic)
    post_linear_modules = None
    if cfg.train.use_layer_norm:
        post_linear_modules = [torch.nn.LayerNorm]
    net_arch = [cfg.train.layer_size for _ in range(cfg.train.num_layers)]
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, qf=net_arch),
        activation_fn=torch.nn.Tanh,
        log_std_init=getattr(cfg.train, "log_std_init", 0.0),
        post_linear_modules=post_linear_modules,
        n_critics=cfg.train.n_critics,
    )

    # Instantiate SAC with provided parameter set
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=cfg.train.actor_lr,
        buffer_size=20_000_000,
        learning_starts=1,
        batch_size=cfg.train.batch_size,
        tau=cfg.train.tau,
        gamma=cfg.train.discount,
        train_freq=cfg.train.train_freq,
        gradient_steps=cfg.train.utd,
        action_noise=None,
        optimize_memory_usage=False,
        ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,
        target_update_interval=1,
        target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,
        use_sde=False,
        sde_sample_freq=-1,
        tensorboard_log=cfg.logdir,
        verbose=1,
        policy_kwargs=policy_kwargs,
    )

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
        run_id = run.id,
        n_eval_episodes=5
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

    total_timesteps = 20e6
    model.learn(total_timesteps=total_timesteps,
                callback=callback
                )
    run.finish()

