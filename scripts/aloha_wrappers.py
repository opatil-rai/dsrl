import gymnasium as gym
from lerobot.policies.pi0.modeling_pi0 import (
    PI0Policy,
)
from gymnasium import spaces
import numpy as np
from lerobot.envs.utils import preprocess_observation, add_envs_task
import torch
import imageio
import os
from lerobot.constants import  OBS_STATE, ACTION
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
)

class Pi0EnvWrapper(gym.Env):
    def __init__(
        self,
        env: gym.Env,
        policy: PI0Policy,
        save_frames: bool = True,
        desired_action_dim : int = 14
    ):
        super().__init__()

        self.desired_action_dim = desired_action_dim
        self.env = env
        self.policy = policy
        self.policy.eval()
        # TODO: don't hardcode device
        self.device = get_device_from_parameters(self.policy)
        self.dtype = get_dtype_from_parameters(self.policy)

        self.n_action_steps = self.policy.config.n_action_steps
        self.max_action_dim = self.policy.config.max_action_dim
        self.action_chunk_dim = self.policy.config.n_action_steps * self.policy.config.max_action_dim

        # Set obs space
        # TODO: I'm hardcoding this for mean pooling over spatial tokens + 14 robot state
        self.obs_dim_size = 2080
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim_size,), dtype=np.float32
        )

        # This action space is chunks of the original action space. For simplicity, we say it is a flat action space of action_dim*pred_horizon (although we only will execute n_action_steps amount of them)
        # TODO: Currently has bounds between -3,3 to capture with high probability all samples that fall under gaussian
        if self.desired_action_dim < 1: # 0, or negative number: entire action chunk is action space
            self.action_space = spaces.Box(
                low=-3.0,
                high=3.0,
                shape=(self.action_chunk_dim,),
                dtype=np.float32,
            )
        elif self.desired_action_dim >=1 and self.desired_action_dim <= self.action_chunk_dim: # reasonably in bounds, make it desired
            self.action_space = spaces.Box(
                low=-3.0,
                high=3.0,
                shape=(self.desired_action_dim,),
                dtype=np.float32,
            )
        else: # above desired, raise error
            raise ValueError(f"desired_action_dim is {self.desired_action_dim} but action chunk is size {self.action_dim}*{self.pred_horizon}={self.action_chunk_dim}, make it smaller")

        self._last_obs = None  # last obs we got from env (this is a raw obs from env, used for base policy)

        # if save_frames is true, then all the frames during the current action chunk execution are saved. step refreshes it, reset refreshes and adds first obs frame
        self.save_frames = save_frames
        if self.save_frames:
            self.frames = []

        # Mirror the base env's render_mode if it exists
        self.render_mode = getattr(env, "render_mode", None)

    def render(self):
        if self.render_mode is None:
            raise ValueError(
                "No render_mode specified and base env does not support rendering."
            )
        # Forward render call to the underlying environment
        return self.env.render()

    def reset(self, **kwargs):
        # TODO load gym seed if it was given at construction
        obs, info = self.env.reset(**kwargs)
        
        # save the obs for base policy to use in next step
        self._last_obs = obs
        # reset policy
        self.policy.reset()
        # reset frames
        if self.save_frames:
            # reset the frames, and add initial
            self.frames = [self.render()]

        # Lets process this observation to obs the first obs for policy
        obs = preprocess_observation(obs)
        
        # TODO: Grab device rather than hardcode cuda
        obs = {
            key: obs[key].to("cuda", non_blocking=True) for key in obs
        }

        # Set task description
        # TODO: For now, just using hardcoded transfer_cube, should be environment dependent
        obs["task"] = ["transfer_cube"]

        # TODO: I don't think this is currently true so i'm commenting it out, but check it out
        # if self.config.adapt_to_pi_aloha:
        #     observation[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        obs = self.policy.normalize_inputs(obs)

        # queue should be totally empty due to reset
        assert len(self.policy._action_queue) == 0

        images, img_masks = self.policy.prepare_images(obs)
        state = self.policy.prepare_state(obs)
        lang_tokens, lang_masks = self.policy.prepare_language(obs)

        # TODO: These embeddings seem to include language stuff as well, probably can rmeove, check out embed_prefix for more detail.
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.policy.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        # TODO: prefix_embs have shape [B, 304, 2048] (B, num_spatial_tokens, feature_dim). I am taking mean over num_spatial to just get 2048 dim. should experiment with other approaches.
        # TODO: The original DSRL paper trains a cnn on the images + uses last hidden state of last token, may want to try those out too.
        mean_pool_prefix_embs = prefix_embs.mean(1)
    
        # include the robot state here
        # TODO: For now, using just raw state (14 dim). The policy.model has `state_proj` where state is processed in denoise_step, could use that too
        img_state_emb = torch.cat([mean_pool_prefix_embs,state],axis=1).detach().cpu().numpy()
        
        return img_state_emb, info

    def expand_action(self, action, target_size):
        """
        Repeat (tile) action until reaching target_size.
        If the final length overshoots, it is truncated.
        """
        repeats = -(-target_size // len(action))  # ceil division
        expanded = np.tile(action, repeats)
        return expanded[:target_size]


    def step(self, _action):
        """
        For n_action_steps, keeping calling select_action with the new obs
        (so they get added to queue / we get next queued action),
        accumulate the reward, and return that.
        In future, just maintain my own queue and call generate_actions directly.
        """

        if self.desired_action_dim >= 1: # need to do tiling
            _action = self.expand_action(_action, self.action_chunk_dim)
        # reshape as needed

        _action = torch.tensor(_action).reshape((1, self.n_action_steps, self.max_action_dim)).to(device=self.device, dtype=self.dtype)


        # clear out old frames from other chunks
        if self.save_frames:
            self.frames = []

        cumulative_rewards = 0
        for i_step in range(self.n_action_steps):
            observation = preprocess_observation(self._last_obs)
        
            observation = {
                key: observation[key].to(self.device, non_blocking=True) for key in observation
            }
            # Set task description
            # TODO: For now, just using blank
            observation["task"] = ["transfer_cube"]

            with torch.inference_mode():
                action = self.policy.select_action(observation, noise=_action)

            # Convert to CPU / numpy.
            action = action.to("cpu").numpy()
            assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"
            action = action[0]

            # Apply the next action.
            observation, reward, terminated, truncated, info = self.env.step(action)

            # add latest obs
            if self.save_frames:
                self.frames += [self.render()]

            # Confirm we are running it up to the point that there are no more actions in the action queue.
            if i_step < self.policy.config.n_action_steps - 1:
                assert len(self.policy._action_queue) > 0
            elif i_step == self.policy.config.n_action_steps - 1:
                assert len(self.policy._action_queue) == 0

            # obs
            self._last_obs = observation
            # cumulate rewards
            # TODO: Include option for discounting.
            # TODO: maybe try sparse rewards
            cumulative_rewards += reward
            # cumulative_rewards += reward
            if terminated:
                # big positive reward for finishing task
                # cumulative_rewards += 10
                pass
            # terminated / truncated
            if terminated or truncated:
                break

        assert self._last_obs == observation

        # Lets process this observation to obs the first obs for policy
        obs = preprocess_observation(observation)
        
        # TODO: Grab device rather than hardcode cuda
        obs = {
            key: obs[key].to("cuda", non_blocking=True) for key in obs
        }

        # Set task description
        # TODO: For now, just using hardcoded transfer_cube, should be environment dependent
        obs["task"] = ["transfer_cube"]

        # TODO: I don't think this is currently true so i'm commenting it out, but check it out
        # if self.config.adapt_to_pi_aloha:
        #     observation[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        obs = self.policy.normalize_inputs(obs)

        images, img_masks = self.policy.prepare_images(obs)
        state = self.policy.prepare_state(obs)
        lang_tokens, lang_masks = self.policy.prepare_language(obs)

        # TODO: These embeddings seem to include language stuff as well, probably can rmeove, check out embed_prefix for more detail.
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.policy.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        # TODO: prefix_embs have shape [B, 304, 2048] (B, num_spatial_tokens, feature_dim). I am taking mean over num_spatial to just get 2048 dim. should experiment with other approaches.
        # TODO: The original DSRL paper trains a cnn on the images + uses last hidden state of last token, may want to try those out too.
        mean_pool_prefix_embs = prefix_embs.mean(1)
    
        # include the robot state here
        # TODO: For now, using just raw state (14 dim). The policy.model has `state_proj` where state is processed in denoise_step, could use that too
        img_state_emb = torch.cat([mean_pool_prefix_embs,state],axis=1).detach().cpu().numpy()
        

        # return last obs, terminated, truncated, and info + the cumulative rewards along the way
        # TODO: The truncated comes from the base env, but I think pusht_env always return Fales for truncated (i.e: no timeouts).
        # If we do get timeouts in the env, we should track that correctly + make critic update take it into account (i.e: do bootstrapping if truncated=true, unlike no bootstrapping when terminated=true but truncated=false)
        # TODO: Returns info dict of last step, is that sufficient?
        return img_state_emb, cumulative_rewards, terminated, truncated, info



def run_basic_pi0_aloha(seeds : list[int] = [0], device : str = "cuda"):
    video_dir_path = "aloha_pi0_videos_lastckpt_seed4_fixnoise_clone"
    fix_noise = True

    os.makedirs(video_dir_path, exist_ok=True)
    import gym_aloha
    fps = 30
    max_steps = 1000

    # TODO: Add insert
    gym_kwargs = {
        "obs_type": "pixels_agent_pos",
        "render_mode": "rgb_array",
    }
    env = gym.make("gym_aloha/AlohaTransferCube-v0", **gym_kwargs)

    # policy = PI0Policy.from_pretrained("BrunoM42/pi0_aloha_transfer_cube").to(device)
    # policy = PI0Policy.from_pretrained("../lerobot/outputs/train/pi0_transfer_cube/checkpoints/005000/pretrained_model").to(device)
    policy = PI0Policy.from_pretrained("../lerobot/outputs/train/pi0_transfer_cube/checkpoints/last/pretrained_model").to(device)

    # policy = PI0Policy.from_pretrained("lerobot/pi0")

    # Initial noise shape (action shape)
    # TODO: This assumes batch size is 1, maybe not always true. fix as needed
    # TODO: This is using max_action_dim, try training pi0 with lower max action dim
    actions_shape = (1, policy.config.n_action_steps, policy.config.max_action_dim)

    if fix_noise:
        noise = torch.normal(
                    mean=0.0,
                    std=1.0,
                    size=actions_shape,
                    dtype=torch.float32,
                    device=device,
                )
        # noise = torch.zeros(size=actions_shape, device=device)
        print("setting fixed noise, this should only happen once")

    for rollout, seed in enumerate(seeds):
        policy.reset()

        total_rewards = 0
        observation, info = env.reset(seed=seed)

        frames = [env.render()]

        for step in range(max_steps):
            observation = preprocess_observation(observation)
        
            observation = {
                key: observation[key].to(device, non_blocking=True) for key in observation
            }
            # Set task description
            # TODO: For now, just using blank
            observation["task"] = ["transfer_cube"]

            with torch.inference_mode():
                # Generate initial noise randomly
                if not fix_noise:
                    noise = torch.normal(
                        mean=0.0,
                        std=1.0,
                        size=actions_shape,
                        dtype=torch.float32,
                        device=device,
                    )
                    # noise = None
                        
                action = policy.select_action(observation, noise=noise.clone())

            # Convert to CPU / numpy.
            action = action.to("cpu").numpy()
            assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"
            action = action[0]

            # Apply the next action.
            observation, reward, terminated, truncated, info = env.step(action)

            frames.append(env.render())
            done = truncated or terminated
            total_rewards += reward

            if done:
                break

        print(f"Done: {done} / episode returns: {total_rewards}")
        video_file_name = f"aloha_pi0_rollout{rollout}_seed{seed}.mp4"
        imageio.mimsave(f"{video_dir_path}/{video_file_name}", frames, fps=fps)

def test_gym_wrapper_env():
    import gym_aloha
    fix_noise = True
    video_dir_path = "aloha_pi0_videos_wrapper_fixnoiseseed6"

    os.makedirs(video_dir_path, exist_ok=True)
    seeds = [6]*5
    fps = 30
    max_steps = 2000
    desired_action_dim = 14

    device="cuda"

    # TODO: Add insert
    gym_kwargs = {
        "obs_type": "pixels_agent_pos",
        "render_mode": "rgb_array",
    }
    env = gym.make("gym_aloha/AlohaTransferCube-v0", **gym_kwargs)

    # policy = PI0Policy.from_pretrained("BrunoM42/pi0_aloha_transfer_cube").to(device)
    # policy = PI0Policy.from_pretrained("../lerobot/outputs/train/pi0_transfer_cube/checkpoints/005000/pretrained_model").to(device)
    policy = PI0Policy.from_pretrained("../lerobot/outputs/train/pi0_transfer_cube/checkpoints/last/pretrained_model").to(device)

    wrapped_env = Pi0EnvWrapper(env=env, policy=policy, desired_action_dim=desired_action_dim)
    
    if fix_noise:
        action =  np.random.randn(
                desired_action_dim
            )

    for rollout_idx, seed in enumerate(seeds):
        obs, info = wrapped_env.reset(seed=seed)
        total_rewards = 0

        frames = wrapped_env.get_wrapper_attr("frames")
        for current_step in range(max_steps):
            if not fix_noise:
                action =  np.random.randn(
                                desired_action_dim
                            )
            obs, reward, terminated, truncated, info = wrapped_env.step(action)

            total_rewards += reward
            frames += wrapped_env.get_wrapper_attr("frames")

            done = truncated or terminated
            if done:
                break
        
        print(f"Done: {done} / episode returns: {total_rewards}")
        video_file_name = f"aloha_pi0_rollout{rollout_idx}_seed{seed}.mp4"
        imageio.mimsave(f"{video_dir_path}/{video_file_name}", frames, fps=fps)


def main():
    # seeds = [4]*10
    # run_basic_pi0_aloha(seeds=seeds)

    test_gym_wrapper_env()

if __name__ == "__main__":
    main()