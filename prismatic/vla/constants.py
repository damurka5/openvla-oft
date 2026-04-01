"""
Important constants for VLA training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.
"""
import sys
import os
from enum import Enum

# Llama 2 token constants
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2  # '</s>'


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

CDPR_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,          # keep your chunking
    "ACTION_DIM": 5,                 # [x,y,z,yaw,grip]
    "PROPRIO_DIM": 5,                # same 5-D proprio you export
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}


# # Function to detect robot platform from command line arguments
# def detect_robot_platform():
#     cmd_args = " ".join(sys.argv).lower()

#     if "libero" in cmd_args:
#         return "LIBERO"
#     elif "aloha" in cmd_args:
#         return "ALOHA"
#     elif "bridge" in cmd_args:
#         return "BRIDGE"
#     elif "cdpr" in cmd_args:
#         return "CDPR"
#     else:
#         # Default to LIBERO if unclear
#         return "LIBERO"


# # Determine which robot platform to use
# ROBOT_PLATFORM = detect_robot_platform()

# # Set the appropriate constants based on the detected platform
# if ROBOT_PLATFORM == "LIBERO":
#     constants = LIBERO_CONSTANTS
# elif ROBOT_PLATFORM == "ALOHA":
#     constants = ALOHA_CONSTANTS
# elif ROBOT_PLATFORM == "BRIDGE":
#     constants = BRIDGE_CONSTANTS
# elif ROBOT_PLATFORM == "CDPR":
#     constants = CDPR_CONSTANTS

# # Assign constants to global variables
# NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
# ACTION_DIM = constants["ACTION_DIM"]
# PROPRIO_DIM = constants["PROPRIO_DIM"]
# ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

# # Print which robot platform constants are being used (for debugging)
# print(f"Using {ROBOT_PLATFORM} constants:")
# print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
# print(f"  ACTION_DIM = {ACTION_DIM}")
# print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
# print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
# print("If needed, manually set the correct constants in `prismatic/vla/constants.py`!")
def detect_robot_platform():
    # 1) Highest priority: explicit env variable
    env = os.environ.get("VLA_ROBOT", "").strip().lower()
    if env in {"cdpr", "libero", "aloha", "bridge"}:
        return env.upper()

    # 2) Fallback: argv sniffing (case-insensitive)
    cmd_args = " ".join(sys.argv).lower()
    # prefer the more specific match first
    if "cdpr" in cmd_args:
        return "CDPR"
    if "aloha" in cmd_args:
        return "ALOHA"
    if "bridge" in cmd_args:
        return "BRIDGE"
    if "libero" in cmd_args:
        return "LIBERO"
    return "LIBERO"  # default

def _read_positive_int_env(name):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = int(str(raw).strip())
    except ValueError:
        print(f"[WARN] Ignoring invalid {name}={raw!r}; expected a positive integer.")
        return None
    if value <= 0:
        print(f"[WARN] Ignoring invalid {name}={raw!r}; expected a positive integer.")
        return None
    return value


def _apply_dimension_overrides(base_constants):
    out = dict(base_constants)
    overrides = {
        "NUM_ACTIONS_CHUNK": _read_positive_int_env("VLA_NUM_ACTIONS_CHUNK"),
        "ACTION_DIM": _read_positive_int_env("VLA_ACTION_DIM"),
        "PROPRIO_DIM": _read_positive_int_env("VLA_PROPRIO_DIM"),
    }
    if overrides["ACTION_DIM"] is not None and overrides["PROPRIO_DIM"] is None:
        overrides["PROPRIO_DIM"] = overrides["ACTION_DIM"]
    for key, value in overrides.items():
        if value is not None:
            out[key] = value
    active = {key: value for key, value in overrides.items() if value is not None}
    return out, active


ROBOT_PLATFORM = detect_robot_platform()

if ROBOT_PLATFORM == "CDPR":
    constants = CDPR_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA":
    constants = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == "BRIDGE":
    constants = BRIDGE_CONSTANTS
else:
    constants = LIBERO_CONSTANTS

constants, active_overrides = _apply_dimension_overrides(constants)

NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

print(f"Using {ROBOT_PLATFORM} constants:")
print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
if active_overrides:
    print(f"  env overrides = {active_overrides}")
print("If needed, manually set the correct constants in `prismatic/vla/constants.py`!")
