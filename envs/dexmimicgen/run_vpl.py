#  Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

import argparse
import multiprocessing
from pathlib import Path
from copy import deepcopy
import cv2
import numpy as np
import imageio
import torch
from tqdm import tqdm

from visuomotor.models.model_registry import REGISTRY
from visuomotor.models.protocols import Policy
from visuomotor.ray_train.simulation.constants import MIMICGEN_BENCHMARK_TASKS, MIMICGEN_TASK_STEPS
from dsrl_mimicgen_actor import DSRLMimicgenActor
import lightning as L


def save_video(frames, filename="video.mp4", fps=30):
    # Accept (N,C,H,W) or (N,H,W,C) in RGB
    if frames.ndim != 4: raise ValueError("frames must be 4D")
    if frames.shape[1] == 3 and frames.shape[-1] != 3:
        frames = frames.transpose(0, 2, 3, 1)
    frames = np.clip(frames, 0, 255).astype(np.uint8)

    H, W = frames.shape[1], frames.shape[2]
    H2, W2 = H - (H % 2), W - (W % 2)  # crop to even dims for H.264
    frames = frames[:, :H2, :W2, :]

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(filename, fourcc, fps, (W2, H2))
    if not out.isOpened():  # fallback if H.264 unavailable
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(filename, fourcc, fps, (W2, H2))

    for f in frames:
        out.write(f[..., ::-1])  # RGB -> BGR

    out.release()
    print(f"Video saved as {filename}")


def sanitize_string_for_path(s: str) -> str:
    return s.replace(":", "-")


def get_policy_and_run_sim(
    checkpoint_name: str,
    num_episodes: int = 10,
    num_steps: int | None = None,
    env_name: str | None = None,
    task_index: int = 0,
    num_envs_per_actor: int | None = None,
    init_noise_mode:str = "different"
) -> None:
    policy = REGISTRY.build_policy_from_artifact_name(checkpoint_name, load_swa_weights=False)
    results_dir = Path(__file__).parent / "results" / sanitize_string_for_path(checkpoint_name)
    run_sim_for_policy(
        policy,
        results_dir=results_dir,
        num_episodes=num_episodes,
        num_steps=num_steps,
        env_name=env_name,
        task_index=task_index,
        num_envs_per_actor=num_envs_per_actor,
        init_noise_mode=init_noise_mode
    )


def run_sim_for_policy(
    policy: Policy,
    results_dir: str | Path,
    num_episodes: int = 10,
    num_steps: int | None = None,
    env_name: str | None = None,
    task_index: int = 0,
    num_envs_per_actor: int | None = None,
    init_noise_mode="different"
) -> None:

    policy.config.num_inference_steps=10
    results_dir = Path(results_dir)
    if num_envs_per_actor is not None:
        policy.config.simulation.num_envs_per_actor = num_envs_per_actor
    if env_name is None:
        env_name = policy.config.simulation.envs[task_index]["name"]
        assert isinstance(env_name, str)
    if num_steps is None:
        if "num_steps" in policy.config.simulation.envs[task_index]:
            num_steps = policy.config.simulation.envs[task_index]["num_steps"]
        else:
            task_name, task_number = env_name.rsplit("_", 1)
            assert task_name in MIMICGEN_BENCHMARK_TASKS and task_number in MIMICGEN_BENCHMARK_TASKS[task_name], (
                f"The task in your config: {task_name}_{task_number} is not part of the available "
                f"mimicgen tasks: {MIMICGEN_BENCHMARK_TASKS}"
            )
            num_steps = MIMICGEN_TASK_STEPS[task_name]
        assert isinstance(num_steps, int)

    if "metadata_gcs_prefix" not in policy.config.simulation:
        policy.config.simulation["metadata_gcs_prefix"] = "mimicgen_equidiff_pc_abs/core/"

    ma = DSRLMimicgenActor(
        pl_module=policy,  # type: ignore
        config=policy.config,
        env_name=env_name,
        current_epoch=99,
        task_index=task_index,
        num_steps=num_steps,
    )

    ma.pl_module.config.simulation.save_videos = True
    ma.num_episodes = 1
    ma.env.env.deterministic_reset=True # turn off variation
    sucesses: list[bool] = []
    all_frames = []
    results_dir.mkdir(exist_ok=True, parents=True)
    successes_path = results_dir / "0_successes.txt"
    with open(successes_path, "w") as f:
        f.write("")

    # save a single initial noise
    batch_size = 1 # # can be 1 for viz
    horizon = ma.config.action_head.horizon
    action_dim = ma.config.action_head.action_dim
    max_frames = ma.num_steps

    # set the noise and its mode
    init_noise = None
    if init_noise_mode == "same_noise_as_first":
        init_noise = torch.randn((batch_size, horizon, action_dim))

    for i in tqdm(range(num_episodes)):
        results = ma.run_simulation(init_noise_mode=init_noise_mode, init_noise=init_noise)
        success = results["success_count"]
        print(f"Success {i}: {success}")
        with open(successes_path, "a") as f:
            f.write(f"{str(success)}\n")
        sucesses.append(success)

        results_path_np = results_dir / f"np/episode_{i}.npy"
        results_path_np.parent.mkdir(exist_ok=True, parents=True)
        frames = results["video"]
        
        # Pad to max length with last frame
        while len(frames) < max_frames:
            frames.append(deepcopy(frames[-1]))
        all_frames.append(frames)
        print(f"Saving results to {results_path_np}")
        np.save(results_path_np, frames)
        try:
            if policy.config.simulation.num_envs_per_actor > 1:
                for i_video in range(policy.config.simulation.num_envs_per_actor):
                    results_path_vid = results_dir / f"videos/episode_{i}_{i_video}.mp4"
                    results_path_vid.parent.mkdir(exist_ok=True, parents=True)
                    print(f"Saving video to {results_path_vid}")
                    save_video(frames[i_video], filename=str(results_path_vid))
            else:
                results_path_vid = results_dir / f"videos/episode_{i}.mp4"
                results_path_vid.parent.mkdir(exist_ok=True, parents=True)
                print(f"Saving video to {results_path_vid}")
                save_video(frames, filename=str(results_path_vid))

        except Exception as e:
            print(f"Failed to save video: {e}")

    # --- Create overlay video ---
    overlay_frames = []
    for frame_idx in range(max_frames):
        overlay = np.mean(
            [rollout[frame_idx] for rollout in all_frames], axis=0
        ).astype(np.uint8)
        overlay_frames.append(overlay)

    imageio.mimsave(f"{results_dir}/videos/overlay.mp4", np.swapaxes(np.array(overlay_frames),1, -1), fps=30)
    print(f"successes: {sucesses}")


def main() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    L.seed_everything(2025)

    # setting the start method can avoid some EGL errors.
    # See https://github.com/google-deepmind/mujoco/issues/991
    multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="bdaii/two_arm_threading-lrenaux/diffpo-2jkfbrtz-pnb9crec:v4",
    )
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--env-name", type=str, default="dexmimicgen")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--num-envs-per-actor", type=int, default=1)
    parser.add_argument("--init-noise-mode", type=str, default="different")
    args = parser.parse_args()
    get_policy_and_run_sim(
        checkpoint_name=args.checkpoint_name,
        num_episodes=args.num_episodes,
        num_envs_per_actor=args.num_envs_per_actor,
        num_steps=args.num_steps,
        env_name=args.env_name,
        task_index=args.task_index,
        init_noise_mode=args.init_noise_mode
    )


if __name__ == "__main__":
    main()
