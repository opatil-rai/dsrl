import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import types

from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionPolicy,
)
from lerobot.constants import ACTION, OBS_IMAGES


from lerobot.utils.random_utils import set_seed
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from lerobot.envs.utils import preprocess_observation
import imageio
from copy import deepcopy
from typing import Optional
from torch import Tensor
from einops import rearrange
from collections import deque

from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)

"""
python diffpo_wrapped_gym.py
"""


# ========= inference  ============
def control_conditional_sample(
    self,
    batch_size: int,
    global_cond: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    device = get_device_from_parameters(self)
    dtype = get_dtype_from_parameters(self)

    # THIS IS THE ONLY DIFFERENCE: sets sample to be whatever the last action from step was
    sample = torch.from_numpy(self.selected_noise_sample).to(
        device=get_device_from_parameters(self), dtype=get_dtype_from_parameters(self)
    )
    # We expect sample to be shape (action_dim*pred_horizon), we want it to be (1,pred_horizon,action_dim)
    sample = rearrange(
        sample,
        "(p a) -> 1 p a",
        p=self.config.horizon,
        a=self.config.action_feature.shape[0],
    )

    self.noise_scheduler.set_timesteps(self.num_inference_steps)

    for t in self.noise_scheduler.timesteps:
        # Predict model output.
        model_output = self.unet(
            sample,
            torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
            global_cond=global_cond,
        )
        # Compute previous image: x_t -> x_t-1
        sample = self.noise_scheduler.step(
            model_output, t, sample, generator=generator
        ).prev_sample

    return sample


class SerlTorchObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Dict(
            {"state": spaces.Dict({"global_conditioning": env.observation_space})}
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._wrap_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._wrap_obs(obs), reward, terminated, truncated, info

    def _wrap_obs(self, obs):
        return {"state": {"global_conditioning": obs}}


class DiffpoEnvWrapper(gym.Env):
    def __init__(
        self,
        env: gym.Env,
        policy: DiffusionPolicy,
        options : Optional[dict] = None,
        seed : Optional[int] = None,
        success_threshold: Optional[float] = 0.8,
        save_frames: bool = True,
    ):
        super().__init__()

        # Mirror the base env's render_mode if it exists
        self.render_mode = getattr(env, "render_mode", None)

        # Optional: support rgb_array if base env supports it but has no render_mode set
        if self.render_mode is None and hasattr(env, "metadata"):
            if "render_modes" in env.metadata and "rgb_array" in env.metadata["render_modes"]:
                self.render_mode = "rgb_array"

        self.env = env
        self.policy = policy
        self.policy.eval()  # Ensure policy is in eval mode
        # potentially override success for pusht
        # TODO: this is specific to pusht having hard-coded 95% success. should make a pr to change that
        self.env.env.env.success_threshold = success_threshold
        self.options = options
        self.seed = seed

        # if save_frames is true, then all the frames during the current action chunk execution are saved. step refreshes it, reset refreshes and adds first obs frame
        self.save_frames = save_frames
        if self.save_frames:
            self.frames = []

        self.pred_horizon = policy.config.horizon
        self.n_action_steps = policy.config.n_action_steps
        self.action_dim = policy.config.action_feature.shape[0]
        self.n_obs_steps = policy.config.n_obs_steps

        # TODO: Make this based on the dimensions of the global conditioning OR the raw obs from env. For now, just conditioning
        # TODO: Global cond dim is never explicitly saved in hugginface model, it's built on the fly to make the unet. For now, just hardcoding what the dim is for the default model
        self.global_cond_dim = 132
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.global_cond_dim,), dtype=np.float32
        )

        # This action space is chunks of the original action space. For simplicity, we say it is a flat action space of action_dim*pred_horizon (although we only will execute n_action_steps amount of them)
        # TODO: Currently has bounds between -6,6 to capture with high probability all samples that fall under gaussian
        self.action_space = spaces.Box(
            low=-6.0,
            high=6.0,
            shape=(self.action_dim * self.pred_horizon,),
            dtype=np.float32,
        )
        # override sample method of action_space to use gaussian instead of uniform distriubtion
        # TODO: Clip this so it's between action space?
        # self.action_space.sample = lambda: np.random.randn(
        #     self.action_dim * self.pred_horizon
        # ).astype(np.float32)

        self._last_obs = None  # last obs we got from env (this is a raw obs from env, used for base policy)

        # Overwrite the policy's conditional_sample to use w/e rl policy wants to use
        self.policy.diffusion.conditional_sample = types.MethodType(
            control_conditional_sample, self.policy.diffusion
        )

    def render(self):
        if self.render_mode is None:
            raise ValueError(
                "No render_mode specified and base env does not support rendering."
            )
        # Forward render call to the underlying environment
        return self.env.render()

    def reset(self, **kwargs):
        # Override kwargs for seed and options
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.options is not None:
            kwargs["options"] = self.options

        # TODO load gym seed if it was given at construction
        obs, info = self.env.reset(**kwargs)
        # save the obs for base policy to use in next step
        self._last_obs = obs
        # reset policy
        self.policy.reset()
        # reset frames
        if self.save_frames:
            # reset the frames, and add initial
            self.frames = [self._last_obs["pixels"]]

        # We now process obs into the first obs the agent will actually get (by taking it and populating buffer a bunch).
        # this is identical to what we do in step execept we don't ues the policy buffer, we just make our own with repeated obs
        obs = preprocess_observation(obs)
        obs = {
            k: v.to("cuda" if torch.cuda.is_available() else "cpu", non_blocking=True)
            for k, v in obs.items()
        }

        # this is inside select_action
        obs = self.policy.normalize_inputs(obs)
        if self.policy.config.image_features:
            obs = dict(
                obs
            )  # shallow copy so that adding a key doesn't modify the original
            obs[OBS_IMAGES] = torch.stack(
                [obs[key] for key in self.policy.config.image_features], dim=-4
            )

        # Make the custom queue
        reset_queue = {
            "observation.state": deque(maxlen=self.n_obs_steps),
            "observation.images": deque(maxlen=self.n_obs_steps),
        }
        # populate it w/ normalized input
        reset_queue = populate_queues(reset_queue, obs)

        # this is in predict_action_chunk
        obs = {
            k: torch.stack(list(reset_queue[k]), dim=1) for k in obs if k in reset_queue
        }
        # this is in generate_actions. This is the global conditioning that captures the last n frames for the policy.
        global_cond = self.policy.diffusion._prepare_global_conditioning(obs)
        # make into (global_cond_dim,) shape
        global_cond = rearrange(
            global_cond,
            "1 global_cond_dim -> global_cond_dim",
            global_cond_dim=self.global_cond_dim,
        )
        global_cond = global_cond.detach().cpu().numpy()
        # TODO: Confirm this global_cond matches what the policy actually gets on first step of reset
        return global_cond, info

    def step(self, _action):
        """
        For n_action_steps, keeping calling select_action with the new obs
        (so they get added to queue / we get next queued action),
        accumulate the reward, and return that.
        In future, just maintain my own queue and call generate_actions directly.

        """
        # set policy.selected_noise_sample from _action, which policy will use
        self.policy.diffusion.selected_noise_sample = _action

        # clear out old frames from other chunks
        if self.save_frames:
            self.frames = []

        cumulative_rewards = 0
        for i_step in range(self.n_action_steps):
            obs = preprocess_observation(self._last_obs)
            obs = {
                k: v.to(
                    "cuda" if torch.cuda.is_available() else "cpu", non_blocking=True
                )
                for k, v in obs.items()
            }

            with torch.inference_mode():
                action = self.policy.select_action(obs)

            action = action.to("cpu").numpy().squeeze(0)

            obs, reward, terminated, truncated, info = self.env.step(action)

            # add latest obs
            if self.save_frames:
                self.frames += [obs["pixels"]]

            # Confirm we are running it up to the point that there are no more actions in the action queue.
            if i_step < self.policy.config.n_action_steps - 1:
                assert len(self.policy._queues[ACTION]) > 0
            elif i_step == self.policy.config.n_action_steps - 1:
                assert len(self.policy._queues[ACTION]) == 0

            # obs
            self._last_obs = obs
            # cumulate rewards
            # TODO: Include option for discounting.
            # TODO: use the actual rewards, not just -1 per step (sparse)
            # cumulative_rewards += -1
            cumulative_rewards += reward
            if terminated:
                # big positive reward for finishing task
                cumulative_rewards += 10
            # terminated / truncated
            if terminated or truncated:
                break

        assert self._last_obs == obs
        # process the final obs so we have it for the rl agent as obs instead of pixels/raw state
        # TODO: Add option for supporting pixels/raw states instead of global cond
        # This is what we do before sending it to select_action
        final_obs = preprocess_observation(obs)
        final_obs = {
            k: v.to("cuda" if torch.cuda.is_available() else "cpu", non_blocking=True)
            for k, v in final_obs.items()
        }

        # this is inside select_action
        final_obs = self.policy.normalize_inputs(final_obs)

        if self.policy.config.image_features:
            final_obs = dict(
                final_obs
            )  # shallow copy so that adding a key doesn't modify the original
            final_obs[OBS_IMAGES] = torch.stack(
                [final_obs[key] for key in self.policy.config.image_features], dim=-4
            )
        # this is in predict_action_chunk
        final_obs = {
            k: torch.stack(list(self.policy._queues[k]), dim=1)
            for k in final_obs
            if k in self.policy._queues
        }
        # this is in generate_actions. This is the global conditioning that captures the last n frames for the policy.
        global_cond = self.policy.diffusion._prepare_global_conditioning(final_obs)
        # get is as numpy array for obs
        global_cond = rearrange(
            global_cond,
            "1 global_cond_dim -> global_cond_dim",
            global_cond_dim=self.global_cond_dim,
        )
        global_cond = global_cond.detach().cpu().numpy()
        # return last obs, terminated, truncated, and info + the cumulative rewards along the way
        # TODO: The truncated comes from the base env, but I think pusht_env always return Fales for truncated (i.e: no timeouts).
        # If we do get timeouts in the env, we should track that correctly + make critic update take it into account (i.e: do bootstrapping if truncated=true, unlike no bootstrapping when terminated=true but truncated=false)
        # TODO: Returns info dict of last step, is that sufficient?
        return global_cond, cumulative_rewards, terminated, truncated, info


def generate_steerable_diffpo_pusht_gym_env(
    device: str = "cuda", scheduler_type="DDIM", options : Optional[dict] = {"reset_to_state" : np.array([314, 201, 187.21077193, 275.01629149, np.pi / 4.0])}, seed : Optional[int] = None,
):
    """
    Returns a gym env of a pusht environment with a wrapped diffusion policy, where
    the actions are what initial noise to feed the diffusion policy. Observations are
    latent vector encoding n_obs_history steps of images and cursor position. Rewards are
    cumulative one step returns over the executed chunk of the base policy.
    """
    # make pusht gym env
    import gym_pusht

    gym_handle = "gym_pusht/PushT-v0"
    # default kwargs for the pusht gym env
    gym_kwargs = {
        "obs_type": "pixels_agent_pos",
        "render_mode": "rgb_array",
        "visualization_width": 384,
        "visualization_height": 384,
        "max_episode_steps": 300,
    }
    env = gym.make(gym_handle, disable_env_checker=True, **gym_kwargs)

    # make diffpo
    policy = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht").to(device)

    # Set scheduler to be DDIM or DDPM
    if scheduler_type == "DDIM":
        # Me adding DDIM
        policy.diffusion.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            # beta_schedule is important
            # this is the best we found
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type=policy.diffusion.config.prediction_type,  # or sample
        )
    else:
        # DDPM is default
        pass

    # Wrap the env
    simple_wrapped_env = DiffpoEnvWrapper(env, policy, options, seed)
    # check_env_spaces(simple_wrapped_env)
    # wrapped_env = SerlTorchObsWrapper(simple_wrapped_env)
    # check_env_spaces(wrapped_env)
    return simple_wrapped_env


def eval_main():
    # Check device is available
    device = "cuda"

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(1000)

    # Reset env, save specific state
    # 1) Set to specific state
    reset_state = np.array([314, 201, 187.21077193, 275.01629149, np.pi / 4.0])
    options = {"reset_to_state": reset_state}
    # 2) Set to random state
    # options = None # No hard-coded reset state

    # Random seed
    # 1) Set the seed manually
    gym_reset_seed = 1234522325 # or None for no fixed seed
    # 2) Or set no random seed
    # gym_reset_seed = None

    wrapped_env = generate_steerable_diffpo_pusht_gym_env(device=device, options=options, seed=gym_reset_seed)
    #wrapped_env = generate_steerable_diffpo_pusht_gym_env(device=device, options=None)


    initial_obs, initial_info = wrapped_env.reset()

    same_actions_as_first = False
    video_prefix = "unique_noise"
    rollout_example_policy_in_wrapped_env(
        wrapped_env,
        same_actions_as_first,
        video_prefix,
    )

    same_actions_as_first = True
    video_prefix = "same_noise"
    rollout_example_policy_in_wrapped_env(
        wrapped_env,
        same_actions_as_first,
        video_prefix,
    )


def rollout_example_policy_in_wrapped_env(
    wrapped_env,
    same_actions_as_first,
    video_prefix,
):
    num_rollouts = 20
    max_frames = 200
    all_rollout_frames = []

    init_noise_generator = torch.Generator(device="cuda")
    init_noise_generator.manual_seed(4221)

    if same_actions_as_first:
        # save a single initial noise
        saved_action = wrapped_env.action_space.sample()
    for rollout in range(num_rollouts):
        observation, info = wrapped_env.reset()
        total_rewards = 0
        # get the initial frame
        frames = wrapped_env.get_wrapper_attr("frames")
        for current_step in range(max_frames):
            # Action select
            # Random action
            # action is shape (batch_size, horizon, action dim)
            if same_actions_as_first:
                action = saved_action
            else:  # new action
                action = wrapped_env.action_space.sample()

            # Step in env in wrapper model.
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            total_rewards += reward
            frames += wrapped_env.get_wrapper_attr("frames")

            if terminated or truncated:
                break
        print(f"terminated:{terminated}, returns: {total_rewards}")
        # get frames
        # Save individual video
        video_filename = f"{video_prefix}_rollout_{rollout}.mp4"
        imageio.mimsave(video_filename, frames, fps=30)
        print(f"Saved {video_filename}")

        # Pad to max length with last frame
        while len(frames) < max_frames:
            frames.append(deepcopy(frames[-1]))

        all_rollout_frames.append(frames)

    # --- Create overlay video ---
    overlay_frames = []

    for frame_idx in range(max_frames):
        overlay = np.mean(
            [rollout[frame_idx] for rollout in all_rollout_frames], axis=0
        ).astype(np.uint8)
        overlay_frames.append(overlay)

    imageio.mimsave(f"{video_prefix}_overlay.mp4", overlay_frames, fps=30)
    print("Saved overlay.mp4")


def check_env_spaces(env, steps=5):
    # Check observation from reset
    obs, info = env.reset()
    assert env.observation_space.contains(
        obs
    ), "Reset obs does not match observation_space!"
    assert isinstance(info, dict)
    print("✅ Reset observation matches observation_space")

    # Check sampled action
    action = env.action_space.sample()
    assert env.action_space.contains(
        action
    ), "Sampled action does not match action_space!"
    print("✅ Sampled action matches action_space")

    # Check several steps
    for i in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(
            obs
        ), f"Step {i}: obs does not match observation_space!"
        assert isinstance(info, dict)
    print(f"✅ All {steps} step observations match observation_space")


if __name__ == "__main__":
    eval_main()