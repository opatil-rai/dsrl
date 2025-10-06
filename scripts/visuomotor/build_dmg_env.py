"""For deterministic object placement, modify dexmimicgen environments to collapse the sampling range
and set the initialization noise to None"""
import json
import os
import h5py

from visuomotor.simulation.async_vector_env import AsyncVectorEnv
from envs.dexmimicgen.dexmimicgen_env import DMGEnvWrapper

def build_dmg_env(
    *,
    env_type: str,
    dataset_path: str,
    config,
    abs_action: bool,
):
    """Standalone builder for MimicGen / DexMimicGen environments.
    Parameters
    ----------
    env_type : {'mimicgen','dexmimicgen'}
        Which environment family to load.
    dataset_path : str
        Path to robomimic-format dataset to read env metadata from.
    config : Any
        Full config object (expects .data.obs_keys.visual and .simulation fields).
    abs_action : bool
        Whether to force absolute action controller behavior.
    batched : bool
        Whether to create an AsyncVectorEnv.
    num_envs : int
        Number of parallel envs if batched.
    get_visual_obs_shapes : Callable
        Function to compute visual obs shapes given camera names.
    run_check_observation_space : bool, default True
        Passed through to AsyncVectorEnv.

    Returns
    -------
    env : gym.Env
        Constructed (possibly vectorized) environment.
    attrs : dict
        Dictionary with keys: task_description, use_image_obs, use_depth_obs,
        image_keys, run_check_observation_space.
    """
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    if env_type == "mimicgen":
        import mimicgen  # noqa: F401
    elif env_type == "dexmimicgen":
        import dexmimicgen  # noqa: F401
    else:
        raise ValueError(f"Unsupported env_type {env_type}")

    env_metadata = get_env_metadata_from_dataset(dataset_path=dataset_path)
    task_description = "dexmimicgen" if env_type == "dexmimicgen" else "mimicgen"

    obs_keys = config.data.obs_keys.visual
    use_image_obs = True if "color" in obs_keys or "point_cloud" in obs_keys else False
    use_depth_obs = True if "depth" in obs_keys or "point_cloud" in obs_keys else False

    env_metadata["env_kwargs"]["camera_depths"] = use_depth_obs
    env_metadata["env_kwargs"]["camera_names"] = config.simulation.get(
        "sim_cameras", ["agentview", "robot0_eye_in_hand"]
    )
    env_metadata["env_kwargs"]["use_object_obs"] = False
    if abs_action:
        env_metadata["env_kwargs"]["controller_configs"]["control_delta"] = False
        env_metadata["env_kwargs"]["controller_configs"]["input_type"] = "absolute"

    if "env_lang" in env_metadata["env_kwargs"]:
        env_metadata["env_kwargs"].pop("env_lang")

    if env_type == "mimicgen" and "point_cloud" in obs_keys:
        from visuomotor.simulation.pointcloud_robomimic_env import PointCloudRobomimicEnv
        env_class = PointCloudRobomimicEnv
    else:
        env_class = None

    def _compute_visual_obs_shapes(camera_names: list[str]) -> dict:
        shapes_config = config.simulation.get("visual_obs_shapes")
        if shapes_config is None:
            # Default expect only RGB images
            return {f"{camera}_image": (84, 84, 3) for camera in camera_names}
        visual_obs_shapes: dict[str, tuple] = {}
        for cam in camera_names:
            visual_obs_shapes[f"{cam}_image"] = shapes_config.color_shape
            if use_depth_obs:
                visual_obs_shapes[f"{cam}_depth"] = shapes_config.depth_shape
        if "point_cloud" in obs_keys:
            visual_obs_shapes["point_cloud"] = shapes_config.point_cloud_shape
            visual_obs_shapes["voxel"] = shapes_config.voxel_shape
        return visual_obs_shapes

    visual_obs_shapes = _compute_visual_obs_shapes(env_metadata["env_kwargs"]["camera_names"])

    def env_fn():
        print("about to create env")
        env_local = EnvUtils.create_env_for_data_processing(
            env_class=env_class,
            env_meta=env_metadata,
            camera_names=env_metadata["env_kwargs"]["camera_names"],
            camera_height=env_metadata["env_kwargs"]["camera_heights"],
            camera_width=env_metadata["env_kwargs"]["camera_widths"],
            reward_shaping=env_metadata["env_kwargs"]["reward_shaping"],
            render=False,
            render_offscreen=True,
            use_image_obs=use_image_obs,
            use_depth_obs=use_depth_obs,
        )
        print("created env in worker")
        env_local = DMGEnvWrapper(env_local, visual_obs_shapes=visual_obs_shapes)
        print("wrapped env")
        return env_local
    
    image_keys = sorted([f"{camera_name}_image" for camera_name in env_metadata["env_kwargs"]["camera_names"]])
    spec = dict(
        obs=dict(
            low_dim=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
            rgb=image_keys,
        ),
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=spec)
    env = env_fn()
    print("inited environments")
    return env

def get_env_metadata_from_dataset(dataset_path, ds_format="robomimic"):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    if ds_format == "robomimic":
        env_meta = json.loads(f["data"].attrs["env_args"])
    else:
        raise ValueError
    f.close()
    return env_meta
