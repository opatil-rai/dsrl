"""
Metaworld instantiates 50 object and goal positions in gym.make, 
and populates them in 'tasks' attribute that are randomly sampled. 
Setting the seed during environment creation will ensure repeatable results.
Check self._last_rand_vec = data["rand_vec"] in SawyerXYZEnv.
The env can also be seeded instead to control the 
instantiation at each reset manually using seeded_rand_vec
"""
import os
import sys
import numpy as np
import torch
import json
import argparse
# sys.path.append(os.path.join(os.path.pardir,"../../"))
import metaworld
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.envs.utils import preprocess_observation
from lerobot.utils.random_utils import set_seed

# set env vars
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # to avoid multi-gpu allocation
HOME = os.environ.get("DSLR_HOME", os.getcwd())


def run_eval(task_name, repo_id, seeds):
    with open(os.path.join(HOME, "envs/metaworld/tasks_metaworld.json"), "r") as f:
        mw_data = json.load(f)
    task_desc = mw_data["TASK_DESCRIPTIONS"][task_name]
    
    sr_list = []
    for seed in seeds:
        # set seed
        env = gym.make("Meta-World/MT1", env_name=task_name, render_mode="rgb_array", seed=seed)
        set_seed(seed) # lerobot seed
        video_dir = f"envs/metaworld/videos/smolvla/{task_name}/{seed}"
        os.makedirs(video_dir, exist_ok=True)
        rec_env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda ep: True)

        # get the policy
        policy = SmolVLAPolicy.from_pretrained(repo_id)
        device = "cuda:0"
        policy = policy.to(device).eval()

        # set some params
        successes = 0
        n_episodes = 50
        n_record = 5

        with torch.no_grad():
            for i in range(n_episodes):
                mw_env = rec_env if i < n_record else env
                obs, info = mw_env.reset()
                frame = mw_env.render()
                i+=1
                for _ in range(300):
                    # lerobot maintains an obs queue of it's own
                    obs_dict = {
                        "pixels": frame.copy(),                    # HxWx3 uint8
                        "agent_pos": obs[:4].astype(np.float32),
                        "environment_state": obs[:39].astype(np.float32),
                    }
                    batch = preprocess_observation(obs_dict)
                    batch = {
                        k: v.to(device if torch.cuda.is_available() else "cpu", non_blocking=True)
                        for k, v in batch.items()
                    }
                    
                    batch["task"]= task_desc
                    # lerobot maintains an action queue too
                    action = policy.select_action(batch).cpu().numpy().astype(np.float32).squeeze()
                    action = np.clip(action, mw_env.action_space.low, mw_env.action_space.high)
                    obs, reward, _, _, info = mw_env.step(action)
                    frame = mw_env.render()
                    if info.get("success", 0.0) > 0.0:
                        successes += 1
                        break

        rec_env.close()
        success_rate = successes / n_episodes
        sr_list.append(success_rate)
        print(f"Saved MP4(s) under: {video_dir}")
        print(f"Seed {seed}: >>> Success rate over {n_episodes} episodes: {success_rate:.2f}")
    print(f"Avg success rate across seeds: {np.average(sr_list)} with std {np.std(sr_list)}")

def get_args():
    parser = argparse.ArgumentParser(description="SmolVLA rollout on Meta-World")
    parser.add_argument(
        "--task_name",
        type=str,
        default="reach-v3",
        help="Meta-World MT1 environment name (e.g., reach-v3, push-v3).",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="msv6/smolvla_metaworld_20k",
        help="HuggingFace repo_id of pretrained SmolVLA checkpoint finetuned on MW.",
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
    run_eval(args.task_name, args.repo_id, args.seeds)