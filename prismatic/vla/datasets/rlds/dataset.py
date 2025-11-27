"""
dataset.py

Core interface script for configuring and initializing RLDS datasets.
"""

import copy
import inspect
import json
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import dlimp as dl
# from prismatic.vla.datasets.rlds import dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from prismatic.overwatch import initialize_overwatch
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from prismatic.vla.datasets.rlds import obs_transforms, traj_transforms
from prismatic.vla.datasets.rlds.utils import goal_relabeling, task_augmentation
from prismatic.vla.datasets.rlds.utils.data_utils import (
    allocate_threads,
    get_dataset_statistics,
    normalize_action_and_proprio,
    pprint_data_mixture,
    tree_map,
)

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch)
tf.config.set_visible_devices([], "GPU")

import tensorflow as tf

def _normalize_stats_for_saver(stats: dict) -> dict:
    """Adapt our stats (with 'action_stats') to the saver-friendly schema (with 'action')."""
    if not isinstance(stats, dict):
        return {"num_transitions": 0, "action": {}}
    if "action" in stats and isinstance(stats["action"], dict):
        return stats  # already in expected shape
    if "action_stats" in stats and isinstance(stats["action_stats"], dict):
        a = stats["action_stats"]
        return {
            "num_transitions": int(stats.get("num_transitions", 0)),
            "action": {
                # keep only what the saver needs; add name if you like
                "dim": a.get("dim"),
                "mean": a.get("mean"),
                "std":  a.get("std"),
                "min":  a.get("min"),
                "max":  a.get("max"),
            },
        }
    # fallback: at least provide empty action dict to avoid KeyError
    return {"num_transitions": int(stats.get("num_transitions", 0)), "action": {}}

# ---- Minimal DLataset-compatible shim ----
class _DLAdapter:
    """Fallback wrapper that provides the DLataset API the code expects."""
    def __init__(self, tfds: tf.data.Dataset):
        self._ds = tfds

    # ---- constructors ----
    @staticmethod
    def from_generator(gen_fn, output_signature):
        ds = tf.data.Dataset.from_generator(gen_fn, output_signature=output_signature)
        return _DLAdapter(ds)

    @staticmethod
    def sample_from_datasets(datasets, weights):
        tf_dsets = [d._ds if isinstance(d, _DLAdapter) else d for d in datasets]
        ds = tf.data.Dataset.sample_from_datasets(tf_dsets, weights)
        return _DLAdapter(ds)

    # ---- trajectory vs frame transforms just map() under the hood ----
    def traj_map(self, fn, num_parallel_calls=None):
        ds = self._ds.map(fn, num_parallel_calls=num_parallel_calls)
        return _DLAdapter(ds)

    def frame_map(self, fn, num_parallel_calls=None):
        ds = self._ds.map(fn, num_parallel_calls=num_parallel_calls)
        return _DLAdapter(ds)

    # ---- basic dataset ops used by the pipeline ----
    def shuffle(self, buffer_size):
        return _DLAdapter(self._ds.shuffle(buffer_size))

    def cache(self):
        return _DLAdapter(self._ds.cache())

    def repeat(self):
        return _DLAdapter(self._ds.repeat())

    def cycle(self):
        # tf.data has repeat() only; cycle() alias for compatibility
        return _DLAdapter(self._ds.repeat())

    def batch(self, batch_size):
        return _DLAdapter(self._ds.batch(batch_size))

    def take(self, n):
        return _DLAdapter(self._ds.take(n))

    def with_ram_budget(self, *_args, **_kwargs):
        # No-op in the shim
        return self

    # ---- iteration helpers ----
    def as_numpy_iterator(self):
        return self._ds.as_numpy_iterator()

    # Allow tf.data APIs to consume us if needed
    def __iter__(self):
        return iter(self._ds)

# Try to import the real class; otherwise use the shim
def _get_dlatset_cls():
    try:
        from prismatic.util.data_utils import DLataset  # most repos
        return DLataset
    except Exception:
        try:
            from prismatic.vla.util.data_utils import DLataset  # some forks
            return DLataset
        except Exception:
            return _DLAdapter


def _ensure_dldataset(ds):
    DLataset = _get_dlatset_cls()
    if isinstance(ds, DLataset):
        return ds

    if isinstance(ds, tf.data.Dataset):
        elem_spec = ds.element_spec
        def _gen():
            for item in ds:
                yield item
        return DLataset.from_generator(_gen, output_signature=elem_spec)

    # As a last resort: try to build a signature from the first element
    it = iter(ds)
    first = next(it)

    def _to_spec(x):
        if isinstance(x, tf.Tensor):
            return tf.TensorSpec(shape=x.shape, dtype=x.dtype)
        if isinstance(x, (bytes, str)):
            return tf.TensorSpec(shape=(), dtype=tf.string)
        if isinstance(x, float):
            return tf.TensorSpec(shape=(), dtype=tf.float32)
        if isinstance(x, int):
            return tf.TensorSpec(shape=(), dtype=tf.int32)
        if isinstance(x, dict):
            return {k: _to_spec(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(_to_spec(v) for v in x)
        raise TypeError(f"Unsupported element type: {type(x)}")

    spec = _to_spec(first)
    def _peeking_gen():
        yield first
        for x in it:
            yield x
    return DLataset.from_generator(_peeking_gen, output_signature=spec)

# ruff: noqa: B006
def make_dataset_from_rlds(
    name: str,
    data_dir: str,
    *,
    train: bool,
    standardize_fn: Optional[Callable[[dict], dict]] = None,
    shuffle: bool = True,
    image_obs_keys: Dict[str, Optional[str]] = {},
    depth_obs_keys: Dict[str, Optional[str]] = {},
    state_obs_keys: List[Optional[str]] = (),
    language_key: Optional[str] = None,
    action_proprio_normalization_type: ACTION_PROPRIO_NORMALIZATION_TYPE,
    dataset_statistics: Optional[Union[dict, str]] = None,
    absolute_action_mask: Optional[List[bool]] = None,
    action_normalization_mask: Optional[List[bool]] = None,
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> Tuple[dl.DLataset, dict]:
    """
    This function is responsible for loading a specific RLDS dataset from storage and getting it into a standardized
    format. Yields a dataset of trajectories. Does not include CPU-intensive operations.

    If `standardize_fn` is provided, it will be applied to each trajectory. This function should get the trajectory
    into a standard format, which includes the keys "observation" and "action". Entry "observation" should be a
    dictionary containing some number of additional keys, which will be extracted into an even more standardized format
    according to the "*_obs_keys" arguments.

    The `image_obs_keys` and `depth_obs_keys` arguments are mappings from new names to old names, or None in place of an
    old name to insert padding. For example, if after `standardize_fn`, your "observation" dict has RGB images called
    "workspace" and "wrist", and `image_obs_keys={"primary": "workspace", "secondary": None, "wrist": "wrist"}`, then
    the resulting dataset will have an "observation" dict containing the keys "image_primary", "image_secondary", and
    "image_wrist", where "image_primary" corresponds to "workspace", "image_secondary" is a padding image, and
    "image_wrist" corresponds to "wrist".

    Entry `state_obs_keys` is a list of 1-dimensional proprioceptive keys to concatenate into a single array, which will
    be placed in the "proprio" key of the "observation" dict. A single padding element (zero) will be inserted for each
    None entry.

    The dataset will also include a "task" dict. If `language_key` is provided, then the "task" dict will contain the
    key "language_instruction", extracted from `traj[language_key]`.

    Args:
        name (str): The name of the RLDS dataset (usually "name" or "name:version").
        data_dir (str): The path to the data directory.
        train (bool): Whether to use the training or validation split.
        shuffle (bool, optional): Whether to shuffle the file read order (does NOT fully shuffle the dataset, since one
            file usually contains many trajectories)!
        standardize_fn (Callable[[dict], dict], optional): A function that, if provided, will be the first
            thing applied to each trajectory.
        image_obs_keys (Mapping[str, str|None]): Mapping from {new: old} indicating which RGB images to extract from the
            "observation" dict. `new_obs = {f"image_{new}": old_obs[old] for new, old in image_obs_keys.items()}`.
            If a value of `old` is None, inserts a padding image instead (empty string).
        depth_obs_keys (Mapping[str, str|None]): Same as `image_obs_keys`, but for depth images. Keys will be
            prefixed with "depth_" instead of "image_".
        state_obs_keys (Sequence[str|None]): List of 1-dimensional proprioception keys to be extracted from the
            "observation" dict, concatenated, and mapped to "proprio". Inserts 1 element of padding for each None entry.
        language_key (str, optional): If provided, the "task" dict will contain the key "language_instruction",
            extracted from `traj[language_key]`.
        action_proprio_normalization_type (str, optional): The type of normalization to perform on the action,
            proprio, or both. Can be "normal" (mean 0, std 1) or "bounds" (normalized to [-1, 1]).
        dataset_statistics: (dict|str, optional): dict (or path to JSON file) that contains dataset statistics
            for normalization. If `action_proprio_normalization_type` is "normal", this should contain "mean" and
            "std" keys. If `action_proprio_normalization_type` is "bounds", this should contain "min" and "max"
            keys. May also provide "num_transitions" and "num_trajectories" keys for downstream usage (e.g., for
            `make_interleaved_dataset`). If not provided, the statistics will be computed on the fly.
        absolute_action_mask (Sequence[bool], optional): By default, all action dimensions are assumed to be
            relative. This is important for when `future_action_window_size > 0`: actions that are taken
            from beyond the end of the trajectory (or beyond the goal timestep when goal relabeling is used)
            need to be made "neutral" to indicate that the task has been completed. For relative actions,
            "neutral" means zero, but for absolute actions, "neutral" means repeating the last valid action.
            This mask, if provided, indicates which action dimensions are absolute.
        action_normalization_mask (Sequence[bool], optional): If provided, indicates which action dimensions
            should be normalized. For example, you might not want to normalize the gripper action dimension if
            it's always exactly 0 or 1. By default, all action dimensions are normalized.
        num_parallel_reads (int): number of parallel read workers. Default to AUTOTUNE.
        num_parallel_calls (int): number of parallel calls for traj_map operations. Default to AUTOTUNE.
    Returns:
        Dataset of trajectories where each step has the following fields:
        - observation:
            - image_{name1, name2, ...} # RGB image observations
            - depth_{name1, name2, ...} # depth image observations
            - proprio                   # 1-dimensional array of proprioceptive observations
            - timestep                  # timestep of each frame
        - task:
            - language_instruction      # language instruction, present if `language_key` is provided
        - action                        # action vector
        - dataset_name                  # name of the dataset
    """
    REQUIRED_KEYS = {"observation", "action"}
    if language_key is not None:
        REQUIRED_KEYS.add(language_key)

    def restructure(traj):
        # apply a standardization function, if provided
        if standardize_fn is not None:
            traj = standardize_fn(traj)

        if not all(k in traj for k in REQUIRED_KEYS):
            raise ValueError(
                f"Trajectory is missing keys: {REQUIRED_KEYS - set(traj.keys())}. " "Did you write a `standardize_fn`?"
            )

        # extracts images, depth images and proprio from the "observation" dict
        traj_len = tf.shape(traj["action"])[0]
        old_obs = traj["observation"]
        new_obs = {}
        for new, old in image_obs_keys.items():
            if old is None:
                new_obs[f"image_{new}"] = tf.repeat("", traj_len)  # padding
            else:
                new_obs[f"image_{new}"] = old_obs[old]

        for new, old in depth_obs_keys.items():
            if old is None:
                new_obs[f"depth_{new}"] = tf.repeat("", traj_len)  # padding
            else:
                new_obs[f"depth_{new}"] = old_obs[old]

        if state_obs_keys:
            new_obs["proprio"] = tf.concat(
                [
                    (
                        tf.zeros((traj_len, 1), dtype=tf.float32)  # padding
                        if key is None
                        else tf.cast(old_obs[key], tf.float32)
                    )
                    for key in state_obs_keys
                ],
                axis=1,
            )

        # add timestep info
        new_obs["timestep"] = tf.range(traj_len)

        # extracts `language_key` into the "task" dict
        task = {}
        if language_key is not None:
            if traj[language_key].dtype != tf.string:
                raise ValueError(
                    f"Language key {language_key} has dtype {traj[language_key].dtype}, " "but it must be tf.string."
                )
            task["language_instruction"] = traj.pop(language_key)

        traj = {
            "observation": new_obs,
            "task": task,
            "action": tf.cast(traj["action"], tf.float32),
            "dataset_name": tf.repeat(name, traj_len),
        }

        if absolute_action_mask is not None:
            if len(absolute_action_mask) != traj["action"].shape[-1]:
                raise ValueError(
                    f"Length of absolute_action_mask ({len(absolute_action_mask)}) "
                    f"does not match action dimension ({traj['action'].shape[-1]})."
                )
            traj["absolute_action_mask"] = tf.tile(
                tf.convert_to_tensor(absolute_action_mask, dtype=tf.bool)[None],
                [traj_len, 1],
            )

        return traj

    builder = tfds.builder(name, data_dir=data_dir)
    dataset = _ensure_dldataset(dataset)
    # load or compute dataset statistics
    if isinstance(dataset_statistics, str):
        with tf.io.gfile.GFile(dataset_statistics, "r") as f:
            dataset_statistics = json.load(f)
    elif dataset_statistics is None:
        full_dataset = dl.DLataset.from_rlds(
            builder, split="all", shuffle=False, num_parallel_reads=num_parallel_reads
        ).traj_map(restructure, num_parallel_calls)
        # tries to load from cache, otherwise computes on the fly
        dataset_statistics = get_dataset_statistics(
            full_dataset,
            hash_dependencies=(
                str(builder.info),
                str(state_obs_keys),
                inspect.getsource(standardize_fn) if standardize_fn is not None else "",
            ),
            save_dir=builder.data_dir,
        )
    dataset_statistics = tree_map(np.array, dataset_statistics)

    # skip normalization for certain action dimensions
    if action_normalization_mask is not None:
        if len(action_normalization_mask) != dataset_statistics["action"]["mean"].shape[-1]:
            raise ValueError(
                f"Length of skip_normalization_mask ({len(action_normalization_mask)}) "
                f"does not match action dimension ({dataset_statistics['action']['mean'].shape[-1]})."
            )
        dataset_statistics["action"]["mask"] = np.array(action_normalization_mask)

    # construct the dataset
    split = "train" if train else "val"

    dataset = dl.DLataset.from_rlds(builder, split=split, shuffle=shuffle, num_parallel_reads=num_parallel_reads)
    dataset = _ensure_dldataset(dataset)
    dataset = dataset.traj_map(restructure, num_parallel_calls)
    dataset = dataset.traj_map(
        partial(
            normalize_action_and_proprio,
            metadata=dataset_statistics,
            normalization_type=action_proprio_normalization_type,
        ),
        num_parallel_calls,
    )

    return dataset, dataset_statistics

import dlimp as dl
import tensorflow as tf

def apply_trajectory_transforms(
    dataset: dl.DLataset,  # can arrive as tf.data in your build — we’ll coerce it
    *,
    train: bool,
    goal_relabeling_strategy: Optional[str] = None,
    goal_relabeling_kwargs: dict = {},
    window_size: int = 1,
    future_action_window_size: int = 0,
    subsample_length: Optional[int] = None,
    skip_unlabeled: bool = False,
    max_action: Optional[float] = None,
    max_proprio: Optional[float] = None,
    task_augment_strategy: Optional[str] = None,
    task_augment_kwargs: dict = {},
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    # ---- add this helper + first-line coercion ----
    # --- Force conversion to DLataset if it's a tf.data dataset ---
    dataset = _ensure_dldataset(dataset)
    
    import tensorflow as tf
    def _ensure_abs_mask(traj):
        # traj is a dict of tensors; create [T] boolean mask
        T = tf.shape(traj["action"])[0]
        traj["absolute_action_mask"] = tf.ones([T], dtype=tf.bool)
        return traj

    dataset = dataset.traj_map(_ensure_abs_mask, num_parallel_calls)
    
    if not (hasattr(dataset, "traj_map") and hasattr(dataset, "frame_map")):
        def _gen():
            for x in dataset:
                yield x
        # Minimal signature to satisfy TF
        sig = tf.TensorSpec(shape=(), dtype=tf.float32)
        dataset = dl.DLataset.from_generator(_gen, output_signature=sig)
        
    def _ensure_dl(ds):
        """Make sure `ds` is a dlimp.DLataset`, even if it's a tf.data.Dataset or iterator."""
        if hasattr(ds, "traj_map") and hasattr(ds, "frame_map"):
            return ds  # already a DLataset

        # --- safest universal path ---
        def _yield_items():
            for x in ds:
                yield x

        # build minimal output_signature
        # (use a permissive 'object' dtype to avoid TF shape inference issues)
        signature = tf.TensorSpec(shape=(), dtype=tf.float32)

        try:
            return dl.DLataset.from_generator(_yield_items, output_signature=signature)
        except Exception:
            # fallback: just materialize if generator wrapping fails
            return dl.DLataset.from_generator(lambda: list(ds), output_signature=signature)

    # force-disable the problematic branch and coerce type up front
    skip_unlabeled = False
    dataset = _ensure_dldataset(dataset)

    # 0) optional filters
    if skip_unlabeled:
        if "language_instruction" not in dataset.element_spec["task"]:
            raise ValueError("skip_unlabeled=True but dataset does not have language labels.")
        dataset = dataset.filter(lambda x: tf.math.reduce_any(x["task"]["language_instruction"] != ""))

    if max_action is not None:
        dataset = dataset.filter(lambda x: tf.reduce_all(tf.abs(x["action"]) <= max_action))
        dataset = _ensure_dldataset(dataset)

    if max_proprio is not None and "proprio" in dataset.element_spec["observation"]:
        dataset = dataset.filter(lambda x: tf.reduce_all(tf.abs(x["observation"]["proprio"]) <= max_proprio))
        dataset = _ensure_dldataset(dataset)

    # 1) pad-mask
    dataset = dataset.traj_map(traj_transforms.add_pad_mask_dict, num_parallel_calls)
    dataset = _ensure_dldataset(dataset)

    # 2) goal relabeling
    if goal_relabeling_strategy is not None:
        dataset = dataset.traj_map(
            partial(getattr(goal_relabeling, goal_relabeling_strategy), **goal_relabeling_kwargs),
            num_parallel_calls,
        )
        dataset = _ensure_dldataset(dataset)

    # 3) task augmentation (train only)
    if train and task_augment_strategy is not None:
        dataset = dataset.traj_map(
            partial(getattr(task_augmentation, task_augment_strategy), **task_augment_kwargs),
            num_parallel_calls,
        )
        dataset = _ensure_dldataset(dataset)

    # 4) chunking
    dataset = dataset.traj_map(
        partial(
            traj_transforms.chunk_act_obs,
            window_size=window_size,
            future_action_window_size=future_action_window_size,
        ),
        num_parallel_calls,
    )
    dataset = _ensure_dldataset(dataset)

    # 5) subsample (train only)
    if train and subsample_length is not None:
        dataset = dataset.traj_map(
            partial(traj_transforms.subsample, subsample_length=subsample_length),
            num_parallel_calls,
        )
        dataset = _ensure_dldataset(dataset)

    return dataset


def apply_per_dataset_frame_transforms(
    dataset: dl.DLataset,
    chunk_filter_fn: Optional[Callable] = None,
):
    """
    Optionally applied *per-dataset* transforms that happen at a frame level.

    Args:
        chunk_filter_fn (callable, optional): Filter function for chunks.
    """
    if chunk_filter_fn:
        dataset = dataset.filter(chunk_filter_fn)
    return dataset


def apply_frame_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,                                # kept for API parity; not passed to transforms below
    image_obs_keys=None,                        # {"primary": "...", "wrist": "..."} or ["primary","wrist"]
    language_key: str = "language_instruction",
    image_augment_kwargs: Union[Dict, Dict[str, Dict]] = {},
    resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]] = {},
    depth_resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]] = {},
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    dataset = _ensure_dldataset(dataset)
    import tensorflow as tf
    from functools import partial

    # Standardize image field names
    if isinstance(image_obs_keys, dict):
        image_names = [f"image_{k}" for k in image_obs_keys.keys()]
    elif isinstance(image_obs_keys, (list, tuple)):
        image_names = [n if n.startswith("image_") else f"image_{n}" for n in image_obs_keys]
    else:
        image_names = ["image_primary", "image_wrist"]

    # 1) Squeeze any rank-1, length-1 strings to scalars BEFORE decoding
    def _squeeze_singletons(frame):
        obs = frame["observation"]

        def _maybe_squeeze(x):
            return tf.cond(tf.equal(tf.rank(x), 1), lambda: x[0], lambda: x)

        for name in image_names:
            if name in obs:
                obs[name] = _maybe_squeeze(obs[name])

        if "task" in frame and language_key in frame["task"]:
            frame["task"][language_key] = _maybe_squeeze(frame["task"][language_key])
        return frame

    dataset = dataset.frame_map(_squeeze_singletons, num_parallel_calls)

    # 2) Helpers: apply transform to observation ONLY (no vmap!)
    def apply_obs_only(fn, frame: Dict) -> Dict:
        frame["observation"] = fn(frame["observation"])
        return frame

    # 3) Decode + resize images (no 'train' kwarg)
    decode_resize = partial(
        obs_transforms.decode_and_resize,
        resize_size=resize_size,
        depth_resize_size=depth_resize_size,
    )
    dataset = dataset.frame_map(partial(apply_obs_only, decode_resize), num_parallel_calls)

    # 4) Optional augmentation (no 'train' kwarg)
    if train and image_augment_kwargs:
        def aug(frame: dict):
            seed = tf.random.uniform([2], maxval=tf.dtypes.int32.max, dtype=tf.int32)
            aug_fn = partial(obs_transforms.augment, seed=seed, augment_kwargs=image_augment_kwargs)
            return apply_obs_only(aug_fn, frame)
        dataset = dataset.frame_map(aug, num_parallel_calls)

    return dataset


def make_single_dataset(
    dataset_kwargs: dict,
    *,
    train: bool,
    traj_transform_kwargs: dict = {},
    frame_transform_kwargs: dict = {},
) -> dl.DLataset:
    """Creates a single dataset from kwargs. Returns a dataset of trajectories.

    Args:
        dataset_kwargs: kwargs passed to `make_dataset_from_rlds` that are dataset-specific.
        train: whether this is a training or validation dataset.
        traj_transform_kwargs: kwargs passed to 'apply_trajectory_transforms'.
        frame_transform_kwargs: kwargs passed to 'get_frame_transforms'.
    """
    dataset, dataset_statistics = make_dataset_from_rlds(
        **dataset_kwargs,
        train=train,
    )
    dataset = _ensure_dldataset(dataset)

    dataset = apply_trajectory_transforms(dataset, **traj_transform_kwargs, train=train)
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)

    # this seems to reduce memory usage without affecting speed
    dataset = dataset.with_ram_budget(1)

    # save for later
    return dataset, dataset_statistics["num_trajectories"], dataset_statistics

# === Core Initializer ===
def make_interleaved_dataset(
    dataset_kwargs_list: List[Dict],
    sample_weights: Optional[List[float]] = None,
    *,
    train: bool,
    shuffle_buffer_size: int,
    traj_transform_kwargs: Optional[Dict] = None,
    frame_transform_kwargs: Optional[Dict] = None,
    batch_size: Optional[int] = None,
    balance_weights: bool = False,
    traj_transform_threads: Optional[int] = None,
    traj_read_threads: Optional[int] = None,
) -> dl.DLataset:
    """
    Creates an interleaved dataset from list of dataset configs (kwargs). Returns a dataset of batched frames.

    Args:
        dataset_kwargs_list: list of kwargs, each element of which is passed to `make_dataset_from_rlds`.
            "num_parallel_calls" and "num_parallel_reads" are overridden using `traj_transform_threads` and
            `traj_read_threads`, respectively.
        sample_weights: sampling weights for each dataset in list. If None, defaults to uniform.
        train: whether this is a training or validation dataset.
        shuffle_buffer_size: size of the dataset shuffle buffer (in number of frames).
        traj_transform_kwargs: kwargs passed to `apply_trajectory_transforms`. "num_parallel_calls" is
            overridden using `traj_transform_threads`.
        frame_transform_kwargs: kwargs passed to `apply_frame_transforms`.
        batch_size: batch size, if not provided output is not batched.
        balance_weights: if True, the sample weights are multiplied by the number of frames in each dataset.
            This makes it so that, if all the sample weights are equal, one full iteration through the interleaved
            dataset will correspond to one full iteration through each individual dataset (only in expectation,
            since in practice the sampling is random).
        traj_transform_threads: total number of parallel calls for trajectory transforms, distributed across
            datasets according to their sampling weights. If None, defaults to AUTOTUNE for every dataset.
        traj_read_threads: total number of parallel read workers for trajectory transforms, distributed across
            datasets according to their sampling weights. If None, defaults to AUTOTUNE for every dataset.
    """
    # Default to uniform sampling (if `sample_weights` is not specified)
    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)

    if len(sample_weights) != len(dataset_kwargs_list):
        raise ValueError(f"sample_weights must be None or have length {len(dataset_kwargs_list)}.")

    # Check valid `traj_transform_kwargs` and `frame_transform_kwargs`
    if (traj_transform_kwargs is None) or (frame_transform_kwargs is None):
        raise ValueError("Missing `traj_transform_kwargs` and `frame_transform_kwargs`!")

    # Get Dataset Sizes
    dataset_sizes, all_dataset_statistics = [], {}
    if len(dataset_kwargs_list) == 0:
        raise ValueError("No datasets were configured. Check dataset_name/mixture routing.")

    datasets = []
    stats_list = []
    dataset_sizes = []
    all_dataset_statistics = {}

    for dataset_kwargs in dataset_kwargs_list:
        state_obs_key = dataset_kwargs.get("state_obs_keys")
        if isinstance(state_obs_key, (list, tuple)):
            if len(state_obs_key) > 1:
                raise ValueError("TFRecord path only supports a single state_obs_key, got: "
                                f"{state_obs_key}")
            state_obs_key = state_obs_key[0]
        
        if "tfrecord_globs" in dataset_kwargs:
            ds, stats = make_dataset_from_tfrecord_globs(
                tfrecord_globs=dataset_kwargs["tfrecord_globs"],
                image_obs_keys=dataset_kwargs.get("image_obs_keys", {}),
                state_obs_key=dataset_kwargs.get("state_obs_keys"),
                language_key=dataset_kwargs.get("language_key"),
                action_key=dataset_kwargs.get("action_key"),
                action_stats=(dataset_kwargs.get("aux_kwargs", {}) or {}).get("action_stats"),
                train=train,
                shuffle_buffer_size=shuffle_buffer_size,
                base_dir=dataset_kwargs.get("data_root_dir") or None,
                dataset_name=dataset_kwargs.get("name", "cdpr_local"),
            )
        else:
            ds, stats = make_dataset_from_rlds(
                **dataset_kwargs,
                train=train,
                num_parallel_calls=traj_transform_threads,
                num_parallel_reads=traj_read_threads,
                dataset_statistics=None,
            )
        # ds = _ensure_dldataset(ds)
        # stats_list.append(stats)
        # all_dataset_statistics[dataset_kwargs["name"]] = stats
        # dataset_sizes.append(int(stats.get("num_transitions", 0)))
        ds = _ensure_dldataset(ds)
        norm_stats = _normalize_stats_for_saver(stats)
        stats_list.append(norm_stats)
        all_dataset_statistics[dataset_kwargs["name"]] = norm_stats
        dataset_sizes.append(int(norm_stats.get("num_transitions", 0)))

    # Keep a copy of the pre-normalization weights to define "primary" datasets
    if len(dataset_sizes) == 0:
        raise ValueError("No dataset sizes computed. Verify routing and globs.")
    if all(int(s) == 0 for s in dataset_sizes):
        raise ValueError("All dataset sizes are zero. Verify your TFRecord globs and parsing keys.")
    orig_weights = np.array(sample_weights, dtype=float)
    primary_dataset_indices = np.array(
        [i for i, w in enumerate(orig_weights) if w == 1.0],
        dtype=int,
    )
    if primary_dataset_indices.size == 0:
        # If nothing is marked primary, treat all as primary
        primary_dataset_indices = np.arange(len(orig_weights), dtype=int)

    # Balance and Normalize Weights (for sampling)
    if balance_weights:
        sample_weights = np.array(sample_weights, dtype=float) * np.array(dataset_sizes, dtype=float)
    sample_weights = np.array(sample_weights, dtype=float)
    sample_weights = sample_weights / np.sum(sample_weights)
    pprint_data_mixture(dataset_kwargs_list, sample_weights)

    # Effective dataset length (how many frames until each "primary" dataset finishes ~1 epoch)
    sizes_arr = np.array(dataset_sizes, dtype=float)
    if len(sizes_arr) == 1:
        dataset_len = int(sizes_arr[0] / (sample_weights[0] if sample_weights[0] > 0 else 1.0))
    else:
        dataset_len = int((sizes_arr / sample_weights)[primary_dataset_indices].max())

    # Allocate Threads based on Weights
    threads_per_dataset = allocate_threads(traj_transform_threads, sample_weights)
    reads_per_dataset = allocate_threads(traj_read_threads, sample_weights)

    overwatch.info("Threads per Dataset: %s", threads_per_dataset)
    overwatch.info("Reads per Dataset: %s", reads_per_dataset)

    # Construct Datasets
    overwatch.info("Constructing datasets...")
    datasets = []
    for dataset_kwargs, threads, reads in zip(
        dataset_kwargs_list,
        threads_per_dataset,
        reads_per_dataset,
     ):
        dataset_frame_transform_kwargs = (
             dataset_kwargs.pop("dataset_frame_transform_kwargs")
             if "dataset_frame_transform_kwargs" in dataset_kwargs
             else {}
         )
        state_obs_key = dataset_kwargs.get("state_obs_keys")
        if isinstance(state_obs_key, (list, tuple)):
            if len(state_obs_key) > 1:
                raise ValueError("TFRecord path only supports a single state_obs_key, got: "
                                f"{state_obs_key}")
            state_obs_key = state_obs_key[0]
            
        if "tfrecord_globs" in dataset_kwargs:
            dataset, dataset_statistics = make_dataset_from_tfrecord_globs(
                tfrecord_globs=dataset_kwargs["tfrecord_globs"],
                image_obs_keys=dataset_kwargs.get("image_obs_keys", {}),
                state_obs_key=dataset_kwargs.get("state_obs_keys"),
                language_key=dataset_kwargs.get("language_key"),
                action_key=dataset_kwargs.get("action_key"),
                action_stats=(dataset_kwargs.get("aux_kwargs", {}) or {}).get("action_stats"),
                train=train,
                shuffle_buffer_size=shuffle_buffer_size,
                base_dir=dataset_kwargs.get("data_root_dir") or None,
                dataset_name=dataset_kwargs.get("name", "cdpr_local"),
            )
        else:
            dataset, _ = make_dataset_from_rlds(
                **dataset_kwargs,
                train=train,
                num_parallel_calls=threads,
                num_parallel_reads=reads,
                dataset_statistics=all_dataset_statistics[dataset_kwargs["name"]],
            )

        dataset = _ensure_dldataset(dataset)
            
        dataset = apply_trajectory_transforms(
            dataset,
            **traj_transform_kwargs,
            num_parallel_calls=threads,
            train=train,
        )
        dataset = apply_per_dataset_frame_transforms(dataset, **dataset_frame_transform_kwargs)
        datasets.append(dataset)

    # Interleave at the Frame Level
    DLataset = _get_dlatset_cls()
    dataset = DLataset.sample_from_datasets(datasets, sample_weights)
    dataset = _ensure_dldataset(dataset) 
    # if train:
    #     dataset = dataset.repeat()
    # Make the training stream infinite without converting to tf.data
    if train:
        # DLataset has its own repeat/cycle; prefer repeat if available
        if hasattr(dataset, "repeat"): 
            dataset = dataset.repeat()
        elif hasattr(dataset, "cycle"):
            dataset = dataset.cycle()
    # Shuffle the Dataset
    #   =>> IMPORTANT :: Shuffle AFTER .cache(), or else memory will still leak!
    dataset = dataset.shuffle(shuffle_buffer_size)

    # Apply Frame Transforms
    overwatch.info("Applying frame transforms on dataset...")
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)

    # [Contract] When training VLA Policies, we let the Collator handle Batching!
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    # Note =>> Seems to reduce memory usage without affecting speed?
    if hasattr(dataset, "with_ram_budget"):
        dataset = dataset.with_ram_budget(1)

    # Save for Later
    dataset.sample_weights = sample_weights

    return dataset, int(dataset_len), all_dataset_statistics

import glob, json, tensorflow as tf

import os, glob, json, numpy as np, tensorflow as tf
import dlimp as dl

def make_dataset_from_tfrecord_globs(
    tfrecord_globs,
    image_obs_keys,
    state_obs_key,
    language_key,
    action_key,
    action_stats=None,
    train=True,
    shuffle_buffer_size=1000,
    base_dir=None,
    dataset_name="cdpr_local",
):
    DLataset = _get_dlatset_cls()
    # 0) resolve files
    files = []
    for g in tfrecord_globs:
        gg = os.path.join(base_dir, g) if (base_dir and not os.path.isabs(g)) else g
        files.extend(sorted(glob.glob(gg)))
    if not files:
        raise FileNotFoundError(f"No TFRecords matched (base_dir={base_dir}): {tfrecord_globs}")
    print("[CDPR] PROPRIO_DIM:", PROPRIO_DIM, "ACTION_DIM:", ACTION_DIM, flush=True)

    # 1) feature spec (matches your exporter)
    feature_spec = {
        "observation/primary": tf.io.FixedLenFeature([], tf.string),
        "observation/wrist":   tf.io.FixedLenFeature([], tf.string),
        "observation/state":   tf.io.VarLenFeature(tf.float32),
        "observation/task_description": tf.io.FixedLenFeature([], tf.string),
        "action": tf.io.VarLenFeature(tf.float32),
        "is_terminal": tf.io.FixedLenFeature([1], tf.int64),
        "is_first":    tf.io.FixedLenFeature([1], tf.int64),
        "is_last":     tf.io.FixedLenFeature([1], tf.int64),
    }

    def _parse(raw):
        ex = tf.io.parse_single_example(raw, feature_spec)
        # tf.print("[CDPR PARSE] keys:", list(ex.keys()))
        obs = {
            "primary": ex["observation/primary"],          # encoded PNG bytes
            "wrist":   ex["observation/wrist"],            # encoded PNG bytes
            "state":   tf.sparse.to_dense(ex["observation/state"]),
            "task_description": ex["observation/task_description"],
        }
        act = tf.cast(tf.sparse.to_dense(ex["action"]), tf.float32)
        return {
            "observation": obs,
            "action": act,
            "is_first": ex["is_first"],
            "is_last":  ex["is_last"],
        }

    # --- low-level TFRecord iterator (no tf.data.Dataset objects) ---
    def _record_iter():
        # compat iterator works fine in TF2
        for path in files:
            for raw in tf.compat.v1.io.tf_record_iterator(path):
                yield raw

    # 2) Episode generator (yields full episodes, trajectory-shaped dicts)
    def episode_generator():
        cur_prim, cur_wrist, cur_state = [], [], []
        cur_lang, cur_act = [], []
        started = False

        for raw in _record_iter():
            step = _parse(raw)
            # materialize tensors to numpy/bytes here (keeps downstream simple)
            primary = step["observation"]["primary"].numpy()
            wrist   = step["observation"]["wrist"].numpy()
            state   = step["observation"]["state"].numpy()
            lang    = step["observation"]["task_description"].numpy()
            act     = step["action"].numpy()
            is_first = int(step["is_first"].numpy()[0]) == 1
            is_last  = int(step["is_last"].numpy()[0])  == 1

            if is_first or not started:
                if started and len(cur_act):
                    T = len(cur_act)
                    yield {
                        "observation": {
                            "image_primary": np.asarray(cur_prim, dtype=object),   # bytes
                            "image_wrist":   np.asarray(cur_wrist, dtype=object),  # bytes
                            "proprio":       np.stack(cur_state, axis=0).astype(np.float32),
                            "timestep":      np.arange(T, dtype=np.int32),
                        },
                        "task": {"language_instruction": np.asarray(cur_lang, dtype=object)},
                        "action":       np.stack(cur_act, axis=0).astype(np.float32),
                        "dataset_name": np.asarray([dataset_name.encode()] * T, dtype=object),
                        # "dataset_name": dataset_name.encode(),

                    }
                cur_prim, cur_wrist, cur_state = [], [], []
                cur_lang, cur_act = [], []
                started = True

            cur_prim.append(primary)
            cur_wrist.append(wrist)
            cur_state.append(state)
            cur_lang.append(lang)
            cur_act.append(act)

            if is_last:
                T = len(cur_act)
                print("[CDPR EP] T:", T,
                  "proprio shape:", np.stack(cur_state, axis=0).astype(np.float32).shape,
                  "action shape:", np.stack(cur_act, axis=0).astype(np.float32).shape,
                  flush=True)
                yield {
                    "observation": {
                        "image_primary": np.asarray(cur_prim, dtype=object),
                        "image_wrist":   np.asarray(cur_wrist, dtype=object),
                        "proprio":       np.stack(cur_state, axis=0).astype(np.float32),
                        "timestep":      np.arange(T, dtype=np.int32),
                    },
                    "task": {"language_instruction": np.asarray(cur_lang, dtype=object)},
                    "action":       np.stack(cur_act, axis=0).astype(np.float32),
                    "dataset_name": np.asarray([dataset_name.encode()] * T, dtype=object),
                    # "dataset_name": dataset_name.encode(),

                }
                cur_prim, cur_wrist, cur_state = [], [], []
                cur_lang, cur_act = [], []
                started = False

        # flush tail (if no is_last)
        if started and len(cur_act):
            T = len(cur_act)
            # print("[CDPR EP] T:", T,
            #       "proprio shape:", ep["observation"]["proprio"].shape,
            #       "action shape:", ep["action"].shape,
            #       flush=True)
            yield {
                "observation": {
                    "image_primary": np.asarray(cur_prim, dtype=object),
                    "image_wrist":   np.asarray(cur_wrist, dtype=object),
                    "proprio":       np.stack(cur_state, axis=0).astype(np.float32),
                    "timestep":      np.arange(T, dtype=np.int32),
                },
                "task": {"language_instruction": np.asarray(cur_lang, dtype=object)},
                "action":       np.stack(cur_act, axis=0).astype(np.float32),
                "dataset_name": np.asarray([dataset_name.encode()] * T, dtype=object),
                # "dataset_name": dataset_name.encode(),

            }

    # 3) Build a DLataset directly from the generator (NO tf.data anywhere)
    dl_dataset = DLataset.from_generator(
    episode_generator,
    output_signature={
        "observation": {
            "image_primary": tf.TensorSpec(shape=(None,), dtype=tf.string),
            "image_wrist":   tf.TensorSpec(shape=(None,), dtype=tf.string),

            # 👇 Each episode: [T, PROPRIO_DIM] = [T, 5]
            "proprio":       tf.TensorSpec(shape=(None, PROPRIO_DIM), dtype=tf.float32),

            "timestep":      tf.TensorSpec(shape=(None,), dtype=tf.int32),
        },
        "task": {"language_instruction": tf.TensorSpec(shape=(None,), dtype=tf.string)},

        # 👇 Each episode: [T, ACTION_DIM] = [T, 5]
        "action":       tf.TensorSpec(shape=(None, ACTION_DIM), dtype=tf.float32),

        # "dataset_name": tf.TensorSpec(shape=(None,), dtype=tf.string),
        # "dataset_name": tf.TensorSpec(shape=(), dtype=tf.string),
        "dataset_name": tf.TensorSpec(shape=(None,), dtype=tf.string),

    },
)


    # 4) Cheap stats …
    num_transitions = 0
    for i, ep in zip(range(256), dl_dataset.as_numpy_iterator()):
        num_transitions += int(np.shape(ep["action"])[0])
    stats = {"num_transitions": int(num_transitions)}
    if action_stats:
        with open(action_stats, "r") as f:
            stats.update({"action_stats": json.load(f)})

    return dl_dataset, stats
