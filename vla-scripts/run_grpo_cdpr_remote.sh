#!/usr/bin/env bash
set -euo pipefail

# Remote launcher for GRPO finetuning on OpenVLA-OFT (CDPR).
# Uses the fast wrapper from RL_VLA_Bootstrapping and the GRPO trainer in openvla-oft.
#
# Usage:
#   bash vla-scripts/run_grpo_cdpr_remote.sh
#
# Optional overrides:
#   NPROC_PER_NODE=2 NUM_PARALLEL_ENVS=8 GRPO_GROUP_SIZE=2 TOTAL_UPDATES=100 \
#   SAVE_EVERY=20 HOLD_STEPS=2 \
#   bash vla-scripts/run_grpo_cdpr_remote.sh

OPENVLA_REPO_ROOT="${OPENVLA_REPO_ROOT:-/root/repo/openvla-oft}"
RLVLA_REPO_ROOT="${RLVLA_REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
SCRIPT_PATH="${SCRIPT_PATH:-$RLVLA_REPO_ROOT/rl_vla_bootstrapping/policy/grpo_finetune_cdpr_fast.py}"
EXTERNAL_GRPO_SCRIPT="${EXTERNAL_GRPO_SCRIPT:-$OPENVLA_REPO_ROOT/vla-scripts/grpo_finetune_cdpr.py}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
# The GRPO trainer forces microbatch_size == minibatch_size under DDP,
# so these defaults stay intentionally conservative on multi-GPU runs.
NUM_PARALLEL_ENVS="${NUM_PARALLEL_ENVS:-8}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-8}"
MICROBATCH_SIZE="${MICROBATCH_SIZE:-8}"
GRPO_GROUP_SIZE="${GRPO_GROUP_SIZE:-2}"
GRPO_GROUP_SELECTION="${GRPO_GROUP_SELECTION:-uniform}"

CATALOG_PATH="${CATALOG_PATH:-/root/repo/CDPR-Dataset/cdpr_dataset/datasets/cdpr_scene_catalog.yaml}"
DESK_TEXTURES_DIR="${DESK_TEXTURES_DIR:-/root/repo/CDPR-Dataset/cdpr_dataset/wrappers/_desk_textures}"
ENV_TRACE_PATH="${ENV_TRACE_PATH:-$RLVLA_REPO_ROOT/runs/cdpr_openvla_grpo_resume_380_to_480_move_to_object/env_trace.jsonl}"

ADAPTER_PATH="${ADAPTER_PATH:-/root/repo/RL_VLA_Bootstrapping/runs/cdpr_openvla_bootstrap_fast_resume_280_to_380/rl/step_0163200/vla_cdpr_adapter}"
ACTION_HEAD_PATH="${ACTION_HEAD_PATH:-/root/repo/RL_VLA_Bootstrapping/runs/cdpr_openvla_bootstrap_fast_resume_280_to_380/rl/step_0163200/action_head_cdpr.pt}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/root/repo/RL_VLA_Bootstrapping/runs/cdpr_openvla_grpo_resume_380_to_480_move_to_object}"
RUN_ID="${RUN_ID:-rl}"

SCENE_POOL_SIZE="${SCENE_POOL_SIZE:-32}"
TEXTURE_POOL_SIZE="${TEXTURE_POOL_SIZE:-10}"
SCENE_SAMPLING="${SCENE_SAMPLING:-round_robin}"
SCENE_REFRESH_EVERY_STEPS="${SCENE_REFRESH_EVERY_STEPS:--1}"

VALIDATE_EVERY_UPDATES="${VALIDATE_EVERY_UPDATES:-50}"
VALIDATION_EPISODES="${VALIDATION_EPISODES:-1}"
VALIDATION_MAX_STEPS="${VALIDATION_MAX_STEPS:-32}"
ROLLOUT_TAP_EVERY_UPDATES="${ROLLOUT_TAP_EVERY_UPDATES:--1}"
TENSORBOARD_EVERY_UPDATES="${TENSORBOARD_EVERY_UPDATES:-1}"
TOTAL_UPDATES="${TOTAL_UPDATES:-100}"
SAVE_EVERY="${SAVE_EVERY:-20}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-170}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-32}"

ACTION_STEP_XYZ="${ACTION_STEP_XYZ:-0.015}"
ACTION_STEP_YAW="${ACTION_STEP_YAW:-0.08}"
HOLD_STEPS="${HOLD_STEPS:-2}"

export PYTHONPATH="$RLVLA_REPO_ROOT:$OPENVLA_REPO_ROOT:${PYTHONPATH:-}"

cd "$OPENVLA_REPO_ROOT"

torchrun --nproc_per_node="$NPROC_PER_NODE" "$SCRIPT_PATH" \
  --external_grpo_script "$EXTERNAL_GRPO_SCRIPT" \
  --adapter_path "$ADAPTER_PATH" \
  --train_loaded_adapter \
  --action_head_path "$ACTION_HEAD_PATH" \
  --resume_actor_stats \
  --catalog_path "$CATALOG_PATH" \
  --desk_textures_dir "$DESK_TEXTURES_DIR" \
  --allowed_objects ycb_apple ycb_pear ycb_peach ycb_b_cups ycb_mug ycb_baseball ycb_plate ycb_bowl \
  --instruction_types move_left move_right move_top move_bottom move_to_object \
  --num_parallel_envs "$NUM_PARALLEL_ENVS" \
  --rollout_steps "$ROLLOUT_STEPS" \
  --max_env_steps "$MAX_ENV_STEPS" \
  --grpo_group_size "$GRPO_GROUP_SIZE" \
  --grpo_group_selection "$GRPO_GROUP_SELECTION" \
  --minibatch_size "$MINIBATCH_SIZE" \
  --microbatch_size "$MICROBATCH_SIZE" \
  --adam_eps 1e-5 \
  --weight_decay 0.0 \
  --no-normalize_advantage \
  --action_step_xyz "$ACTION_STEP_XYZ" \
  --action_step_yaw "$ACTION_STEP_YAW" \
  --hold_steps "$HOLD_STEPS" \
  --delta_closer_reward_coef 0 \
  --delta_farther_penalty_coef 0 \
  --lock_non_commanded_axes \
  --lock_non_commanded_axes_threshold 0.05 \
  --randomize_ee_start \
  --ee_start_x_bounds -0.03 0.03 \
  --ee_start_y_bounds -0.03 0.03 \
  --ee_start_z 0.15 \
  --no-record_trajectory \
  --use_wrapper_cache \
  --no-wrapper_cleanup \
  --prebuild_scene_cache \
  --scene_pool_size "$SCENE_POOL_SIZE" \
  --texture_pool_size "$TEXTURE_POOL_SIZE" \
  --scene_sampling "$SCENE_SAMPLING" \
  --scene_refresh_every_steps "$SCENE_REFRESH_EVERY_STEPS" \
  --validate_every_updates "$VALIDATE_EVERY_UPDATES" \
  --validation_episodes "$VALIDATION_EPISODES" \
  --validation_max_steps "$VALIDATION_MAX_STEPS" \
  --no-save_validation_frames \
  --rollout_tap_every_updates "$ROLLOUT_TAP_EVERY_UPDATES" \
  --tensorboard_every_updates "$TENSORBOARD_EVERY_UPDATES" \
  --total_updates "$TOTAL_UPDATES" \
  --save_every "$SAVE_EVERY" \
  --run_root_dir "$RUN_ROOT_DIR" \
  --run_id "$RUN_ID" \
  --env_trace_path "$ENV_TRACE_PATH" \
  --capture_frames
