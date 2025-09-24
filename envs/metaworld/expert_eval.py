# pip install "metaworld>=2.0.0" "gymnasium>=0.29" "imageio[ffmpeg]"
import metaworld
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
# from metaworld.policies.sawyer_stick_push_v3_policy import SawyerStickPushV3Policy
# from metaworld.policies.sawyer_faucet_open_v3_policy import SawyerFaucetOpenV3Policy
# from metaworld.policies.sawyer_window_open_v3_policy import SawyerWindowOpenV3Policy
# from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy
# from metaworld.policies.sawyer_dial_turn_v3_policy import SawyerDialTurnV3Policy
from metaworld.policies.sawyer_button_press_v3_policy import SawyerButtonPressV3Policy


# env_name="stick-push-v3"
# env_name = "faucet-open-v3"
# env_name = "window-open-v3"
# env_name = "reach-v3"
# env_name = "dial-turn-v3"
env_name = "button-press-v3"


env = gym.make("Meta-World/MT1", env_name=env_name, render_mode="rgb_array")

# record every episode; videos will be saved under save_dir
save_dir = f"envs/metaworld/videos/expert/{env_name}/"
env = RecordVideo(env, video_folder=save_dir, episode_trigger=lambda ep: True)
policy = SawyerButtonPressV3Policy()

for seed in np.random.randint(0, 10000, size=3):
    obs, info = env.reset(seed=seed)
    for t in range(300):
        a = policy.get_action(obs)
        obs, reward, _, _, info = env.step(a)
        if info.get("success", 0.0) > 0.0:
            print(f"### seed {seed} success")
            break

env.close()
print(f"Saved MP4(s) to {save_dir}")
