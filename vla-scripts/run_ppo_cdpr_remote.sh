#!/usr/bin/env bash
set -euo pipefail

# Remote launcher for PPO finetuning on OpenVLA-OFT (CDPR), aligned with current script defaults.
# Usage:
#   bash vla-scripts/run_ppo_cdpr_remote.sh
# Optional overrides:
#   NPROC_PER_NODE=2 NUM_PARALLEL_ENVS=5 SCENE_POOL_SIZE=10 ACTION_STEP_XYZ=0.015 HOLD_STEPS=10 \
#   bash vla-scripts/run_ppo_cdpr_remote.sh

REPO_ROOT="${REPO_ROOT:-/root/repo/openvla-oft}"
SCRIPT_PATH="${SCRIPT_PATH:-$REPO_ROOT/vla-scripts/ppo_finetune_cdpr.py}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
NUM_PARALLEL_ENVS="${NUM_PARALLEL_ENVS:-5}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-16}"
MICROBATCH_SIZE="${MICROBATCH_SIZE:-16}"

CATALOG_PATH="${CATALOG_PATH:-/root/repo/CDPR-Dataset/cdpr_dataset/datasets/cdpr_scene_catalog.yaml}"
DESK_TEXTURES_DIR="${DESK_TEXTURES_DIR:-/root/repo/CDPR-Dataset/cdpr_dataset/wrappers/_desk_textures}"
ENV_TRACE_PATH="${ENV_TRACE_PATH:-$REPO_ROOT/runs_ppo/env_trace.jsonl}"

SCENE_POOL_SIZE="${SCENE_POOL_SIZE:-10}"
TEXTURE_POOL_SIZE="${TEXTURE_POOL_SIZE:-10}"
SCENE_SAMPLING="${SCENE_SAMPLING:-round_robin}"
SCENE_REFRESH_EVERY_STEPS="${SCENE_REFRESH_EVERY_STEPS:--1}"

VALIDATE_EVERY_UPDATES="${VALIDATE_EVERY_UPDATES:-10}"
VALIDATION_EPISODES="${VALIDATION_EPISODES:-1}"
VALIDATION_MAX_STEPS="${VALIDATION_MAX_STEPS:-40}"
ROLLOUT_TAP_EVERY_UPDATES="${ROLLOUT_TAP_EVERY_UPDATES:-10}"

ACTION_STEP_XYZ="${ACTION_STEP_XYZ:-0.012}"
ACTION_STEP_YAW="${ACTION_STEP_YAW:-0.08}"
HOLD_STEPS="${HOLD_STEPS:-8}"

cd "$REPO_ROOT"

torchrun --nproc_per_node="$NPROC_PER_NODE" "$SCRIPT_PATH" \
  --catalog_path "$CATALOG_PATH" \
  --desk_textures_dir "$DESK_TEXTURES_DIR" \
  --num_parallel_envs "$NUM_PARALLEL_ENVS" \
  --minibatch_size "$MINIBATCH_SIZE" \
  --microbatch_size "$MICROBATCH_SIZE" \
  --adam_eps 1e-5 \
  --weight_decay 0.0 \
  --normalize_advantage \
  --action_step_xyz "$ACTION_STEP_XYZ" \
  --action_step_yaw "$ACTION_STEP_YAW" \
  --hold_steps "$HOLD_STEPS" \
  --delta_closer_reward_coef 0 \
  --delta_farther_penalty_coef 0 \
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
  --save_validation_frames \
  --rollout_tap_every_updates "$ROLLOUT_TAP_EVERY_UPDATES" \
  --env_trace_path "$ENV_TRACE_PATH"
