"""
materialize.py

Factory class for initializing Open-X Embodiment dataset kwargs and other parameters; provides and exports functions for
clear control flow.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

from prismatic.overwatch import initialize_overwatch
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS, ActionEncoding
from prismatic.vla.datasets.rlds.oxe.mixtures import OXE_NAMED_MIXTURES
from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def make_oxe_dataset_kwargs(
    dataset_name: str,
    data_root_dir: Path,
    load_camera_views: Tuple[str] = ("primary",),
    load_depth: bool = False,
    load_proprio: bool = True,
    load_language: bool = True,
    action_proprio_normalization_type = ACTION_PROPRIO_NORMALIZATION_TYPE,
) -> Dict[str, Any]:
    """Generates config (kwargs) for given dataset from Open-X Embodiment."""
    dataset_kwargs = deepcopy(OXE_DATASET_CONFIGS[dataset_name])
    # if dataset_kwargs["action_encoding"] not in [ActionEncoding.EEF_POS, ActionEncoding.EEF_R6, ActionEncoding.JOINT_POS_BIMANUAL]:
    #     raise ValueError(f"Cannot load `{dataset_name}`; only EEF_POS & EEF_R6 & JOINT_POS_BIMANUAL actions supported!")
    enc = dataset_kwargs.get("action_encoding", ActionEncoding.EEF_POS)
    # allow string in configs.py to avoid circular import
    if isinstance(enc, str):
        try:
            enc = getattr(ActionEncoding, enc) if enc.isupper() else ActionEncoding[enc.upper()]
        except Exception:
            raise ValueError(f"Unknown action_encoding '{enc}'. Expected one of: "
                            f"EEF_POS, EEF_R6, JOINT_POS_BIMANUAL")

    dataset_kwargs["action_encoding"] = enc  # normalize in-place

    if enc not in [ActionEncoding.EEF_POS, ActionEncoding.EEF_R6, ActionEncoding.JOINT_POS_BIMANUAL]:
        raise ValueError(f"Unsupported action_encoding: {enc}")

    
    # [Contract] For EEF_POS & EEF_R6 actions, only the last action dimension (gripper) is absolute!
    # Normalize all action dimensions *except* the gripper
    if dataset_kwargs["action_encoding"] is ActionEncoding.EEF_POS:
        dataset_kwargs["absolute_action_mask"] = [False] * 6 + [True]
        dataset_kwargs["action_normalization_mask"] = [True] * 6 + [False]
    elif dataset_kwargs["action_encoding"] is ActionEncoding.EEF_R6:
        dataset_kwargs["absolute_action_mask"] = [False] * 9 + [True]
        dataset_kwargs["action_normalization_mask"] = [True] * 9 + [False]
    elif dataset_kwargs["action_encoding"] is ActionEncoding.JOINT_POS_BIMANUAL:
        dataset_kwargs["absolute_action_mask"] = [True] * 14
        dataset_kwargs["action_normalization_mask"] = [True] * 14
    dataset_kwargs["action_proprio_normalization_type"] = action_proprio_normalization_type

    # Adjust Loaded Camera Views
    if len(missing_keys := (set(load_camera_views) - set(dataset_kwargs["image_obs_keys"]))) > 0:
        raise ValueError(f"Cannot load `{dataset_name}`; missing camera views `{missing_keys}`")

    # Filter
    dataset_kwargs["image_obs_keys"] = {
        k: v for k, v in dataset_kwargs["image_obs_keys"].items() if k in load_camera_views
    }
    dataset_kwargs["depth_obs_keys"] = {
        k: v for k, v in dataset_kwargs["depth_obs_keys"].items() if k in load_camera_views
    }

    # Eliminate Unnecessary Keys
    dataset_kwargs.pop("state_encoding")
    dataset_kwargs.pop("action_encoding")
    if not load_depth:
        dataset_kwargs.pop("depth_obs_keys")
    if not load_proprio:
        dataset_kwargs.pop("state_obs_keys")

    # Load Language
    if load_language:
        dataset_kwargs["language_key"] = "language_instruction"

    # Specify Standardization Transform
    dataset_kwargs["standardize_fn"] = OXE_STANDARDIZATION_TRANSFORMS[dataset_name]

    # Add any aux arguments
    if "aux_kwargs" in dataset_kwargs:
        dataset_kwargs.update(dataset_kwargs.pop("aux_kwargs"))

    return {"name": dataset_name, "data_dir": str(data_root_dir), **dataset_kwargs}


def _coerce_action_encoding(val):
    if isinstance(val, ActionEncoding):
        return val
    if isinstance(val, str):
        key = val.strip().upper()
        if hasattr(ActionEncoding, key):
            return getattr(ActionEncoding, key)
    raise ValueError(f"Unknown action_encoding: {val}")

def make_oxe_dataset_kwargs(
    dataset_name: str,
    data_root_dir: Path,
    load_camera_views: Tuple[str, ...] = ("primary",),
    load_depth: bool = False,
    load_proprio: bool = True,
    load_language: bool = True,
    action_proprio_normalization_type = ACTION_PROPRIO_NORMALIZATION_TYPE,
) -> Dict[str, Any]:
    """
    Generates config (kwargs) for a dataset.

    - If the dataset config contains `tfrecord_globs`, treat it as a plain TFRecord dataset
      (your CDPR case) and return only the keys needed by `make_dataset_from_tfrecord_globs`.
    - Otherwise, fall back to the original OXE/LIBERO path.
    """
    cfg = deepcopy(OXE_DATASET_CONFIGS[dataset_name])

    # -------------------------------
    # Fast-path: plain TFRecord input
    # -------------------------------
    if "tfrecord_globs" in cfg:
        # image_obs_keys may be a list OR a {view_name: key} dict; normalize to list.
        image_obs_keys = cfg.get("image_obs_keys", ["observation/primary"])
        if isinstance(image_obs_keys, dict):
            # Keep only requested camera views (if present)
            image_obs_keys = [image_obs_keys[k] for k in load_camera_views if k in image_obs_keys]
        else:
            # Already a list of feature keys; optionally filter by view names present in the tail
            # e.g., ".../primary" or ".../wrist"
            wanted = set(load_camera_views)
            filtered = [k for k in image_obs_keys if k.split("/")[-1] in wanted]
            if filtered:
                image_obs_keys = filtered

        out: Dict[str, Any] = {
            "name": dataset_name,
            "data_dir": str(data_root_dir),
            "tfrecord_globs": cfg["tfrecord_globs"],
            "image_obs_keys": image_obs_keys,
            "state_obs_key": cfg.get("state_obs_key", "observation/state") if load_proprio else None,
            "language_key": cfg.get("language_key", "observation/task_description") if load_language else None,
            "action_key": cfg.get("action_key", "action"),
            # Optional: action stats JSON written by your exporter
            "action_stats": cfg.get("action_stats", None),
        }
        # Prune Nones (downstream expects missing keys rather than None)
        out = {k: v for k, v in out.items() if v is not None}
        return out

    # ------------------------------------------------------
    # Default path: OXE/LIBERO datasets via TFDS (unchanged)
    # ------------------------------------------------------
    # Backwards/robust handling of action encoding
    enc = cfg.get("action_encoding", ActionEncoding.EEF_POS)
    if isinstance(enc, str):
        enc = getattr(ActionEncoding, enc) if enc.isupper() else ActionEncoding[enc.upper()]
    if enc not in [ActionEncoding.EEF_POS, ActionEncoding.EEF_R6, ActionEncoding.JOINT_POS_BIMANUAL]:
        raise ValueError(f"Unsupported action_encoding: {enc}")

    cfg["action_encoding"] = enc
    # Absolute/normalize masks for OXE encodings (gripper absolute)
    if enc is ActionEncoding.EEF_POS:
        cfg["absolute_action_mask"] = [False]*6 + [True]
        cfg["action_normalization_mask"] = [True]*6 + [False]
    elif enc is ActionEncoding.EEF_R6:
        cfg["absolute_action_mask"] = [False]*9 + [True]
        cfg["action_normalization_mask"] = [True]*9 + [False]
    elif enc is ActionEncoding.JOINT_POS_BIMANUAL:
        cfg["absolute_action_mask"] = [True]*14
        cfg["action_normalization_mask"] = [True]*14
    cfg["action_proprio_normalization_type"] = action_proprio_normalization_type

    # Camera view filtering (allow dict OR list)
    image_obs_keys = cfg.get("image_obs_keys", {})
    if isinstance(image_obs_keys, dict):
        missing = set(load_camera_views) - set(image_obs_keys.keys())
        if missing:
            raise ValueError(f"Cannot load `{dataset_name}`; missing camera views `{missing}`")
        cfg["image_obs_keys"] = {k: v for k, v in image_obs_keys.items() if k in load_camera_views}
    else:
        # list → optionally filter by requested views if they’re encoded in the key suffix
        wanted = set(load_camera_views)
        filtered = [k for k in image_obs_keys if k.split("/")[-1] in wanted]
        if filtered:
            cfg["image_obs_keys"] = filtered

    # Depth
    depth_obs_keys = cfg.get("depth_obs_keys", {})
    if isinstance(depth_obs_keys, dict):
        cfg["depth_obs_keys"] = {k: v for k, v in depth_obs_keys.items() if k in load_camera_views}
        if not load_depth:
            cfg.pop("depth_obs_keys", None)
    else:
        if not load_depth:
            cfg.pop("depth_obs_keys", None)

    # Proprio
    if not load_proprio:
        cfg.pop("state_obs_keys", None)  # tolerate missing

    # Language key (OXE uses "language_instruction"; keep that if present)
    if load_language:
        cfg.setdefault("language_key", "language_instruction")
    else:
        cfg.pop("language_key", None)

    # Standardization (OXE path only)
    cfg["standardize_fn"] = OXE_STANDARDIZATION_TRANSFORMS[dataset_name]

    # Clean up keys we don’t need to pass further (tolerate missing)
    cfg.pop("state_encoding", None)
    cfg.pop("action_encoding", None)
    if "aux_kwargs" in cfg:
        cfg.update(cfg.pop("aux_kwargs"))

    return {"name": dataset_name, "data_dir": str(data_root_dir), **cfg}

def get_oxe_dataset_kwargs_and_weights(
    dataset_or_mixture_name: str,
    data_root_dir: Path,
    load_camera_views: Tuple[str, ...] = ("primary",),
    load_depth: bool = False,
    load_proprio: bool = True,
    load_language: bool = True,
    balance_weights: bool = False,
    action_proprio_normalization_type = ACTION_PROPRIO_NORMALIZATION_TYPE,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    Returns (list_of_dataset_kwargs, sampling_weights).
    Accepts either:
      - a single dataset key present in OXE_DATASET_CONFIGS, or
      - a named mixture present in OXE_NAMED_MIXTURES.
    """
    print("[get_oxe] name:", dataset_or_mixture_name,
      "| data_root_dir:", data_root_dir,
      "| views:", load_camera_views, flush=True)
    
    name = dataset_or_mixture_name
    if name in OXE_DATASET_CONFIGS:
        return [make_oxe_dataset_kwargs(name, data_root_dir)], [1.0]
    if name in OXE_NAMED_MIXTURES:
        ds_kwargs_list, weights = [], []
        for item in OXE_NAMED_MIXTURES[name]:
            # accept both ("dataset_name", weight) and {"name": ..., "weight": ...}
            if isinstance(item, dict):
                dname = item["name"]; w = float(item.get("weight", 1.0))
            else:
                dname, w = item  # (name, weight) tuple
            ds_kwargs_list.append(make_oxe_dataset_kwargs(dname, data_root_dir))
            weights.append(float(w))
        return ds_kwargs_list, weights

    raise ValueError(
        f"Unknown dataset_or_mixture_name='{name}'. "
        f"Known datasets: {list(OXE_DATASET_CONFIGS.keys())}. "
        f"Known mixtures: {list(OXE_NAMED_MIXTURES.keys())}."
    )
# def get_oxe_dataset_kwargs_and_weights(
#     data_root_dir: Path,
#     mixture_spec: List[Tuple[str, float]],
#     load_camera_views: Tuple[str] = ("primary",),
#     load_depth: bool = False,
#     load_proprio: bool = True,
#     load_language: bool = True,
#     action_proprio_normalization_type = ACTION_PROPRIO_NORMALIZATION_TYPE,
# ) -> Tuple[Dict[str, Any], List[float]]:
#     """
#     Generates dataset kwargs for a given dataset mix from the Open X-Embodiment dataset. The returned kwargs
#     (per-dataset configs) and weights can be passed directly to `make_interleaved_dataset`.

#     :param data_root_dir: Base directory containing RLDS/TFDS-formatted datasets (from Open-X)
#     :param mixture_spec: List of (dataset_name, sampling_weight) from `oxe.mixtures.OXE_NAMED_MIXTURES`
#     :param load_camera_views: Camera views to load; see `oxe.dataset_configs.py` for available views.
#     :param load_depth: Load depth information in addition to camera RGB.
#     :param load_proprio: Load proprioceptive state.
#     :param load_language: Load language instructions.
#     :param action_proprio_normalization_type: Normalization scheme to use for proprioceptive actions.

#     return: Tuple of (per_dataset_kwargs, sampling_weights)
#     """
#     included_datasets, filtered_mixture_spec = set(), []
#     for d_name, d_weight in mixture_spec:
#         if d_name in included_datasets:
#             overwatch.warning(f"Skipping Duplicate Dataset: `{(d_name, d_weight)}`")
#             continue

#         included_datasets.add(d_name)
#         filtered_mixture_spec.append((d_name, d_weight))

#     # Assemble Dataset Config (kwargs) and Weights
#     per_dataset_kwargs, sampling_weights = [], []
#     for d_name, d_weight in filtered_mixture_spec:
#         try:
#             per_dataset_kwargs.append(
#                 make_oxe_dataset_kwargs(
#                     d_name,
#                     data_root_dir,
#                     load_camera_views,
#                     load_depth,
#                     load_proprio,
#                     load_language,
#                     action_proprio_normalization_type,
#                 )
#             )
#             sampling_weights.append(d_weight)

#         except ValueError as e:
#             overwatch.warning(f"Skipping `{d_name}` due to Error: {e}")

#     return per_dataset_kwargs, sampling_weights
