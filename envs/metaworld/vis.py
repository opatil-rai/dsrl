# pip install "metaworld>=2.0.0" "gymnasium>=0.29" "imageio[ffmpeg]"
import metaworld
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

env_name="door-close-v3"
env = gym.make("Meta-World/MT1", env_name=env_name, render_mode="rgb_array")

# record every episode; videos will be saved under ./videos/
env = RecordVideo(env, video_folder=f"envs/metaworld/videos/{env_name}", episode_trigger=lambda ep: True)

for seed in np.random.randint(0, 10000, size=3):
    obs, info = env.reset(seed=seed)
    for t in range(300):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

env.close()
print("Saved MP4(s) to ./videos/")
