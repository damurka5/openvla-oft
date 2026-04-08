#!/usr/bin/env python3
"""
grpo_finetune_cdpr.py

Group Relative Policy Optimization for OpenVLA-OFT on the CDPR language-conditioned environment.

Implementation notes:
- Reuses the PPO environment/model/loading utilities from sibling `ppo_finetune_cdpr.py`.
- Trains only LoRA adapters + continuous action head + actor log-std.
- Uses grouped rollouts from an identical simulator snapshot per observation.
- No learned value head / critic is optimized.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import random
import sys
import types
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW


def _load_ppo_module():
    script_path = Path(__file__).with_name("ppo_finetune_cdpr.py").resolve()
    spec = importlib.util.spec_from_file_location("openvla_oft_ppo_finetune_cdpr", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load PPO utilities from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ppo = _load_ppo_module()

SummaryWriter = ppo.SummaryWriter
make_run_dir = ppo.make_run_dir
save_run_config = ppo.save_run_config
run_validation_rollouts = ppo.run_validation_rollouts
_shape_reward_with_delta_progress = ppo._shape_reward_with_delta_progress
_extract_reward_components = ppo._extract_reward_components
_distance_ee_to_target_from_obs = ppo._distance_ee_to_target_from_obs
_sanitize_env_reward = ppo._sanitize_env_reward
_detect_unstable_transition = ppo._detect_unstable_transition
_motion_diagnostics = ppo._motion_diagnostics
_extract_target_object_fields = ppo._extract_target_object_fields
_float_or_none = ppo._float_or_none
_finite_float_or_none = ppo._finite_float_or_none
save_rollout_tap_npz = ppo.save_rollout_tap_npz
POLICY_ACTION_DIM = ppo.POLICY_ACTION_DIM
ACTION_DIM = ppo.ACTION_DIM
NUM_ACTIONS_CHUNK = ppo.NUM_ACTIONS_CHUNK
tqdm = ppo.tqdm


def parse_args() -> argparse.Namespace:
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--grpo_group_size", type=int, default=2)
    extra.add_argument(
        "--grpo_group_selection",
        type=str,
        choices=["uniform", "first"],
        default="uniform",
        help=(
            "Which sampled candidate continues the actual environment after GRPO scoring. "
            "`uniform` keeps on-policy continuation by choosing one sampled candidate uniformly; "
            "`first` always continues candidate 0."
        ),
    )
    extra.add_argument(
        "--grpo_advantage_eps",
        type=float,
        default=1.0e-6,
        help="Small epsilon added to the per-group std when normalizing relative rewards.",
    )
    extra.add_argument(
        "--grpo_normalize_group_advantage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize grouped rewards into zero-mean unit-std advantages within each sampled group.",
    )
    extra.add_argument(
        "--grpo_clip_advantage_abs",
        type=float,
        default=6.0,
        help="Optional absolute clip applied to normalized group-relative advantages. Use <=0 to disable.",
    )

    extras, remaining = extra.parse_known_args(sys.argv[1:])
    saved_argv = list(sys.argv)
    try:
        sys.argv = [saved_argv[0]] + remaining
        args = ppo.parse_args()
    finally:
        sys.argv = saved_argv

    for key, value in vars(extras).items():
        setattr(args, key, value)
    return args


class _ZeroValueHead(nn.Module):
    def forward(self, action_hidden_states: torch.Tensor) -> torch.Tensor:
        batch = int(action_hidden_states.shape[0])
        return torch.zeros((batch,), device=action_hidden_states.device, dtype=torch.float32)


class OpenVLAGRPOPolicy(ppo.OpenVLAPPOPolicy):
    def __init__(
        self,
        vla: nn.Module,
        processor: Any,
        action_head: nn.Module,
        device: torch.device,
        num_images_in_input: int,
        init_log_std: float,
    ):
        super().__init__(
            vla=vla,
            processor=processor,
            action_head=action_head,
            value_head=_ZeroValueHead(),
            device=device,
            num_images_in_input=num_images_in_input,
            init_log_std=init_log_std,
        )


class CDPRVisionLanguageEnv(ppo.CDPRVisionLanguageEnv):
    def capture_state(self) -> dict[str, Any]:
        if not hasattr(self.env, "capture_state"):
            raise RuntimeError("Underlying CDPR env does not support capture_state().")
        return {
            "env_state": self.env.capture_state(),
            "instruction": str(self._instruction),
        }

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        if not hasattr(self.env, "restore_state"):
            raise RuntimeError("Underlying CDPR env does not support restore_state().")
        self.env.restore_state(snapshot["env_state"])
        self._instruction = str(snapshot.get("instruction", self._instruction))

    def observe(self) -> Dict[str, Any]:
        raw_obs = self.env._get_obs() if hasattr(self.env, "_get_obs") else None
        obs = {
            "image_primary": ppo._latest_image_from_sim(self.env.sim, wrist=False),
            "image_wrist": ppo._latest_image_from_sim(self.env.sim, wrist=True),
            "instruction": self._instruction,
        }
        self._attach_state(obs, raw_obs)
        return obs


@dataclass
class Transition:
    img_primary: np.ndarray
    img_wrist: Optional[np.ndarray]
    instruction: str
    action: np.ndarray
    logprob: float
    env_reward: float
    reward: float
    advantage: float


@dataclass
class CandidateResult:
    reward: float
    env_reward: float
    env_reward_raw: Optional[float]
    env_reward_clipped: bool
    env_reward_non_finite: bool
    done: bool
    env_done: bool
    forced_scene_refresh: bool
    unstable: bool
    unstable_reason: str
    forced_unstable_reset: bool
    step_info: Dict[str, Any]
    reward_components: Dict[str, float]
    motion_diag: Dict[str, Any]
    closer_bonus: float
    farther_penalty: float
    raw_dist_delta: float
    dist_before: Optional[float]
    dist_after: Optional[float]
    target_object_catalog: str
    target_object_body: str
    target_object_name: str
    next_obs: Dict[str, Any]
    post_state: Optional[dict[str, Any]]


def _group_relative_advantages(
    rewards: np.ndarray,
    *,
    normalize: bool,
    eps: float,
    clip_abs: float,
) -> np.ndarray:
    group_rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    centered = group_rewards - float(group_rewards.mean())
    if normalize:
        scale = float(group_rewards.std(ddof=0))
        centered = centered / max(scale, float(eps))
    if clip_abs > 0:
        centered = np.clip(centered, -float(clip_abs), float(clip_abs))
    return centered.astype(np.float32)


def save_checkpoint(
    run_dir: Path,
    step: int,
    vla: nn.Module,
    action_head: nn.Module,
    log_std: nn.Parameter,
) -> None:
    ckpt_dir = run_dir / f"step_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(vla, "save_pretrained"):
        adapter_dir = ckpt_dir / "vla_cdpr_adapter"
        vla.save_pretrained(adapter_dir)

    torch.save(action_head.state_dict(), ckpt_dir / "action_head_cdpr.pt")
    actor_stats = {"log_std": log_std.detach().cpu()}
    torch.save(actor_stats, ckpt_dir / "grpo_actor_stats.pt")
    # Keep the familiar filename too so tooling can inspect log_std consistently.
    torch.save(actor_stats, ckpt_dir / "ppo_actor_stats.pt")


def _select_group_index(args: argparse.Namespace, rng: np.random.Generator) -> int:
    if args.grpo_group_selection == "first":
        return 0
    return int(rng.integers(0, int(args.grpo_group_size)))


def main() -> None:
    args = parse_args()
    if not args.capture_frames:
        raise ValueError(
            "Vision-language GRPO requires rendered images; please run with --capture_frames."
        )
    if args.minibatch_size < 1:
        raise ValueError("--minibatch_size must be >= 1.")
    if args.microbatch_size < 1:
        raise ValueError("--microbatch_size must be >= 1.")
    if args.num_parallel_envs < 1:
        raise ValueError("--num_parallel_envs must be >= 1.")
    if args.grpo_group_size < 2:
        raise ValueError("--grpo_group_size must be >= 2 for meaningful group-relative optimization.")
    if args.target_kl is not None and args.target_kl <= 0:
        raise ValueError("--target_kl must be > 0 when provided.")
    if args.adam_eps <= 0:
        raise ValueError("--adam_eps must be > 0.")
    if args.weight_decay < 0:
        raise ValueError("--weight_decay must be >= 0.")
    if args.validation_episodes < 1:
        raise ValueError("--validation_episodes must be >= 1.")
    if args.validation_max_steps < 1:
        raise ValueError("--validation_max_steps must be >= 1.")
    if args.hold_steps < 0:
        print("[WARN] --hold_steps must be >= 0; clamping to 0.", flush=True)
        args.hold_steps = 0
    if args.scene_refresh_every_steps == 0:
        args.scene_refresh_every_steps = -1
    if args.validate_every_updates == 0:
        args.validate_every_updates = -1
    if args.tensorboard_every_updates == 0:
        args.tensorboard_every_updates = -1
    if args.rollout_tap_every_updates == 0:
        args.rollout_tap_every_updates = -1
    if args.scene_pool_size == 0:
        args.scene_pool_size = -1
    if args.texture_pool_size == 0:
        args.texture_pool_size = -1
    if args.microbatch_size > args.minibatch_size:
        print(
            f"[WARN] --microbatch_size ({args.microbatch_size}) > --minibatch_size ({args.minibatch_size}); "
            "clamping microbatch_size to minibatch_size.",
            flush=True,
        )
        args.microbatch_size = args.minibatch_size

    rank, local_rank, world_size = ppo._init_distributed()
    is_main = ppo._is_main_process(rank)

    if torch.cuda.is_available():
        if world_size > 1:
            device = torch.device(f"cuda:{local_rank}")
        else:
            parsed_device = torch.device(args.device)
            if parsed_device.type == "cuda" and parsed_device.index is None:
                parsed_device = torch.device("cuda:0")
            device = parsed_device
        if device.type == "cuda":
            torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if is_main and torch.cuda.is_available() and world_size == 1 and torch.cuda.device_count() > 1:
        print(
            f"[INFO] Detected {torch.cuda.device_count()} GPUs. "
            "Use `torchrun --nproc_per_node=<num_gpus> ... grpo_finetune_cdpr.py` to train with DDP on all GPUs.",
            flush=True,
        )

    if is_main and args.use_wrapper_cache and args.wrapper_cleanup:
        print(
            "[WARN] `--use_wrapper_cache` is effectively disabled when `--wrapper_cleanup` is true in CDPR env reset logic. "
            "Use `--no-wrapper_cleanup` to keep cache active across episodes.",
            flush=True,
        )

    if world_size > 1 and not args.ddp_find_unused_parameters:
        if is_main:
            print(
                "[WARN] Multi-GPU GRPO may skip some trainable params on some iterations. "
                "Forcing --ddp_find_unused_parameters for DDP stability.",
                flush=True,
            )
        args.ddp_find_unused_parameters = True

    if world_size > 1 and args.microbatch_size != args.minibatch_size:
        if is_main:
            print(
                "[WARN] Multi-GPU mode uses one backward per optimizer step for stability on current PyTorch; "
                "forcing --microbatch_size == --minibatch_size. "
                "To reduce memory, lower --minibatch_size.",
                flush=True,
            )
        args.microbatch_size = args.minibatch_size

    total_parallel_envs = int(args.num_parallel_envs) * int(world_size)

    ppo.set_seed(args.seed + rank)
    branch_rng = np.random.default_rng(args.seed + 90_000 + rank)

    run_dir_local: Optional[Path] = None
    if is_main:
        run_dir_local = make_run_dir(args)
        save_run_config(args, run_dir_local)
        print(f"Run dir: {run_dir_local}", flush=True)
    run_dir = Path(ppo._broadcast_object(str(run_dir_local) if run_dir_local is not None else None, rank))
    tb_writer = None
    if is_main and SummaryWriter is not None:
        tb_logdir = run_dir / "tensorboard"
        tb_logdir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tb_logdir), flush_secs=10)
        print(f"[tensorboard] Logging to {tb_logdir}", flush=True)
        print(
            f"[tensorboard] train metrics: "
            f"{'disabled' if args.tensorboard_every_updates <= 0 else f'every {args.tensorboard_every_updates} updates'} "
            f"| validation metrics: "
            f"{'disabled' if args.validate_every_updates <= 0 else f'every {args.validate_every_updates} updates'}",
            flush=True,
        )
    elif is_main:
        print("[WARN] TensorBoard logging disabled: torch.utils.tensorboard is unavailable.", flush=True)

    cdpr_root = ppo._resolve_cdpr_dataset_root(Path(args.cdpr_dataset_root))
    ppo._prune_generated_wrapper_artifacts(
        cdpr_dataset_root=cdpr_root,
        is_main=is_main,
        rank=rank,
    )

    args.desk_textures_dir = ppo._prepare_desk_textures_dir(
        src_dir=args.desk_textures_dir,
        run_dir=run_dir,
        is_main=is_main,
        rank=rank,
        max_textures=args.texture_pool_size if args.texture_pool_size > 0 else 128,
    )

    vla, processor = ppo.load_vla_and_processor(args, device, rank=rank)
    ppo.maybe_enable_gradient_checkpointing(vla, enabled=bool(args.gradient_checkpointing))

    llm_dim = ppo._resolve_llm_dim(vla)
    if llm_dim is None:
        raise RuntimeError("Could not resolve llm_dim from OpenVLA model wrapper.")

    action_head = ppo.build_action_head(args=args, llm_dim=llm_dim, device=device)

    policy: nn.Module = OpenVLAGRPOPolicy(
        vla=vla,
        processor=processor,
        action_head=action_head,
        device=device,
        num_images_in_input=args.num_images_in_input,
        init_log_std=args.init_log_std,
    )
    if world_size > 1:
        if device.type != "cuda":
            raise RuntimeError("DDP multi-process GRPO currently requires CUDA devices.")
        ddp_kwargs = dict(
            device_ids=[device.index],
            find_unused_parameters=bool(args.ddp_find_unused_parameters),
            gradient_as_bucket_view=True,
        )
        ddp_params = inspect.signature(DDP.__init__).parameters
        if "static_graph" in ddp_params:
            ddp_kwargs["static_graph"] = bool(args.ddp_static_graph)
        policy = DDP(policy, **ddp_kwargs)
        if args.ddp_static_graph and hasattr(policy, "_set_static_graph"):
            try:
                policy._set_static_graph()
            except Exception:
                pass

    policy_core = ppo._unwrap_module(policy)
    lora_params = [p for p in policy_core.vla.parameters() if p.requires_grad]
    action_params = list(policy_core.action_head.parameters())
    trainable_params = lora_params + action_params + [policy_core.log_std]

    if is_main:
        print(
            "[trainable] "
            f"lora={sum(p.numel() for p in lora_params)} "
            f"action_head={sum(p.numel() for p in action_params)} "
            f"value_head=0 "
            f"world_size={world_size} "
            f"num_parallel_envs_per_rank={args.num_parallel_envs} "
            f"total_parallel_envs={total_parallel_envs}",
            flush=True,
        )

    param_groups = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": args.learning_rate})
    param_groups.append({"params": action_params, "lr": args.learning_rate})
    param_groups.append({"params": [policy_core.log_std], "lr": args.learning_rate})
    optimizer = AdamW(
        param_groups,
        eps=float(args.adam_eps),
        weight_decay=float(args.weight_decay),
    )
    use_tqdm = bool(args.status_bar and is_main and (tqdm is not None))
    if is_main and args.status_bar and tqdm is None:
        print("[WARN] tqdm is unavailable; falling back to periodic summary prints.", flush=True)

    envs: List[CDPRVisionLanguageEnv] = []
    val_env: Optional[CDPRVisionLanguageEnv] = None
    updates_pbar = None
    trace_fp = None
    global_step = 0
    episode_idx = 0

    if is_main and args.env_trace_path:
        trace_path = Path(args.env_trace_path).expanduser().resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_fp = trace_path.open("a", encoding="utf-8")

    try:
        base_env_seed = args.seed + rank * 10_000
        with ppo._silence_stdio(bool(args.quiet_env_logs)):
            for env_idx in range(int(args.num_parallel_envs)):
                env = CDPRVisionLanguageEnv(
                    cdpr_dataset_root=Path(args.cdpr_dataset_root),
                    cdpr_mujoco_root=args.cdpr_mujoco_root,
                    catalog_path=args.catalog_path,
                    max_steps=args.max_env_steps,
                    action_step_xyz=args.action_step_xyz,
                    action_step_yaw=args.action_step_yaw,
                    hold_steps=args.hold_steps,
                    capture_frames=args.capture_frames,
                    instruction_types=args.instruction_types,
                    desk_textures_dir=args.desk_textures_dir,
                    allowed_objects=args.allowed_objects,
                    desk_geom_regex=args.desk_geom_regex,
                    desk_texrepeat=args.desk_texrepeat,
                    wrapper_cleanup=args.wrapper_cleanup,
                    use_wrapper_cache=args.use_wrapper_cache,
                    invert_x_action=args.invert_x_action,
                    invert_y_action=args.invert_y_action,
                    seed=base_env_seed + env_idx,
                )
                envs.append(env)

        if not envs:
            raise RuntimeError("No training envs created. Check --num_parallel_envs.")

        env_main = envs[0]
        if args.prebuild_scene_cache:
            cache_info = env_main.enable_prebuilt_scene_cache(
                scene_pool_size=args.scene_pool_size,
                texture_pool_size=args.texture_pool_size,
                seed=args.seed + 20_000 + rank,
            )
            for env in envs[1:]:
                env.attach_prebuilt_scene_cache(
                    scene_wrapper_cache=env_main._scene_wrapper_cache,
                    texture_name_by_wrapper=env_main._texture_name_by_wrapper,
                )
            if is_main:
                print(
                    f"[env_cache] train scenes={cache_info['scenes']} "
                    f"variants={cache_info['variants']} textures={cache_info['textures']}",
                    flush=True,
                )

        scene_names, next_reset_options = ppo._make_scene_reset_sampler(
            scene_names=env_main.scene_names(),
            scene_sampling=args.scene_sampling,
            seed=args.seed + 10_000 + rank,
        )
        n_envs_local = len(envs)

        if is_main:
            print(
                f"[env] scene_sampling={args.scene_sampling} "
                f"scene_refresh_every_steps={args.scene_refresh_every_steps} "
                f"catalog_scenes={len(scene_names)} "
                f"prebuild_scene_cache={bool(args.prebuild_scene_cache)} "
                f"scene_pool_size={args.scene_pool_size} "
                f"texture_pool_size={args.texture_pool_size} "
                f"num_parallel_envs={args.num_parallel_envs} "
                f"hold_steps={args.hold_steps} "
                f"grpo_group_size={args.grpo_group_size} "
                f"delta_closer_reward_coef={args.delta_closer_reward_coef} "
                f"delta_farther_penalty_coef={args.delta_farther_penalty_coef} "
                f"reward_clip_abs={args.reward_clip_abs} "
                f"guard_unstable_transitions={bool(args.guard_unstable_transitions)} "
                f"unstable_gain_threshold={args.unstable_gain_threshold} "
                f"unstable_realized_xyz_norm_threshold={args.unstable_realized_xyz_norm_threshold} "
                f"unstable_env_reward_abs_threshold={args.unstable_env_reward_abs_threshold} "
                f"validate_every_updates={args.validate_every_updates} "
                f"rollout_tap_every_updates={args.rollout_tap_every_updates}",
                flush=True,
            )
            print(
                f"[grpo] selected_steps_per_update={args.rollout_steps * n_envs_local} "
                f"candidate_samples_per_update={args.rollout_steps * n_envs_local * args.grpo_group_size} "
                f"(rollout_steps={args.rollout_steps} x num_parallel_envs={n_envs_local} x group_size={args.grpo_group_size})",
                flush=True,
            )
            if len(scene_names) < max(2, int(args.num_parallel_envs)):
                print(
                    "[WARN] Catalog scene count is lower than desired parallel diversity. "
                    "Increase scene count or lower --num_parallel_envs.",
                    flush=True,
                )
            if args.scene_sampling == "env_random" and int(args.num_parallel_envs) > 1:
                print(
                    "[WARN] scene_sampling=env_random may pick duplicate scenes across parallel envs. "
                    "Use --scene_sampling round_robin for guaranteed per-reset scene diversity.",
                    flush=True,
                )

        obs_batch: List[Dict[str, Any]] = []
        with ppo._silence_stdio(bool(args.quiet_env_logs)):
            for env in envs:
                obs_batch.append(env.reset(options=next_reset_options()))

        steps_since_reset = [0 for _ in envs]
        ep_ret_running = [0.0 for _ in envs]
        ep_ret_env_running = [0.0 for _ in envs]

        next_val_reset_options = None
        if is_main and args.validate_every_updates > 0:
            with ppo._silence_stdio(bool(args.quiet_env_logs)):
                val_env = CDPRVisionLanguageEnv(
                    cdpr_dataset_root=Path(args.cdpr_dataset_root),
                    cdpr_mujoco_root=args.cdpr_mujoco_root,
                    catalog_path=args.catalog_path,
                    max_steps=args.max_env_steps,
                    action_step_xyz=args.action_step_xyz,
                    action_step_yaw=args.action_step_yaw,
                    hold_steps=args.hold_steps,
                    capture_frames=args.capture_frames,
                    instruction_types=args.instruction_types,
                    desk_textures_dir=args.desk_textures_dir,
                    allowed_objects=args.allowed_objects,
                    desk_geom_regex=args.desk_geom_regex,
                    desk_texrepeat=args.desk_texrepeat,
                    wrapper_cleanup=args.wrapper_cleanup,
                    use_wrapper_cache=args.use_wrapper_cache,
                    invert_x_action=args.invert_x_action,
                    invert_y_action=args.invert_y_action,
                    seed=args.seed + 50_000,
                )
            if args.prebuild_scene_cache:
                val_cache_info = val_env.attach_prebuilt_scene_cache(
                    scene_wrapper_cache=env_main._scene_wrapper_cache,
                    texture_name_by_wrapper=env_main._texture_name_by_wrapper,
                )
                print(
                    f"[env_cache] val scenes={val_cache_info['scenes']} "
                    f"variants={val_cache_info['variants']} textures={val_cache_info['textures']}",
                    flush=True,
                )
            _, next_val_reset_options = ppo._make_scene_reset_sampler(
                scene_names=val_env.scene_names(),
                scene_sampling=args.scene_sampling,
                seed=args.seed + 60_000,
            )

        if use_tqdm:
            updates_pbar = tqdm(total=args.total_updates, desc="updates", dynamic_ncols=True, leave=True)

        for update in range(1, args.total_updates + 1):
            policy.eval()
            transitions: List[Transition] = []
            rollout_records: List[Dict[str, Any]] = []
            loss_policy_values: List[float] = []
            loss_entropy_values: List[float] = []
            loss_total_values: List[float] = []
            approx_kl_values: List[float] = []
            clip_fraction_values: List[float] = []
            reward_xyz_values: List[float] = []
            reward_orient_values: List[float] = []
            reward_obj_values: List[float] = []
            reward_success_values: List[float] = []
            motion_gain_values: List[float] = []
            motion_cosine_values: List[float] = []
            episode_returns: List[float] = []
            episode_returns_env: List[float] = []
            unstable_transition_count = 0
            reward_clip_count = 0
            reward_non_finite_count = 0
            selected_rewards: List[float] = []
            selected_env_rewards: List[float] = []
            group_advantages_all: List[float] = []

            rollout_pbar = (
                tqdm(
                    total=args.rollout_steps,
                    desc=f"u{update:05d} rollout",
                    dynamic_ncols=True,
                    leave=False,
                )
                if use_tqdm
                else None
            )
            for rollout_step in range(args.rollout_steps):
                batch_images_primary = [obs_batch[i]["image_primary"] for i in range(n_envs_local)]
                batch_images_wrist = (
                    [obs_batch[i]["image_wrist"] for i in range(n_envs_local)]
                    if args.num_images_in_input > 1
                    else None
                )
                batch_instructions = [obs_batch[i]["instruction"] for i in range(n_envs_local)]

                with torch.no_grad():
                    mean_action, std_action, _, mean_pre_action = policy(
                        images_primary=batch_images_primary,
                        images_wrist=batch_images_wrist,
                        instructions=batch_instructions,
                    )

                sampled_actions_group: List[np.ndarray] = []
                env_actions_group: List[np.ndarray] = []
                logprob_group: List[np.ndarray] = []
                for _ in range(int(args.grpo_group_size)):
                    with torch.no_grad():
                        sampled_action_t, sampled_pre_tanh_t = ppo.squashed_gaussian_sample(mean_pre_action, std_action)
                        logprob_t = ppo.squashed_gaussian_log_prob(
                            sampled_action_t,
                            mean_pre_action,
                            std_action,
                            pre_tanh_action=sampled_pre_tanh_t,
                        ).sum(dim=-1)
                        action_env_t = torch.clamp(sampled_action_t, -1.0, 1.0)

                    sampled_actions_group.append(sampled_action_t.cpu().numpy().astype(np.float32))
                    env_actions_group.append(action_env_t.cpu().numpy().astype(np.float32))
                    logprob_group.append(logprob_t.cpu().numpy().astype(np.float32))

                for env_idx, env in enumerate(envs):
                    obs = obs_batch[env_idx]
                    base_state = env.capture_state()
                    next_step_count = steps_since_reset[env_idx] + 1
                    forced_scene_refresh = (
                        args.scene_refresh_every_steps > 0
                        and (next_step_count % args.scene_refresh_every_steps == 0)
                    )

                    candidate_results: List[CandidateResult] = []
                    candidate_rewards: List[float] = []

                    for group_idx in range(int(args.grpo_group_size)):
                        action_env = env_actions_group[group_idx][env_idx]
                        dist_before = _distance_ee_to_target_from_obs(obs)
                        next_obs, env_reward_raw_value, env_done, step_info = env.step(action_env)
                        step_info = dict(step_info) if isinstance(step_info, dict) else {}
                        motion_diag = _motion_diagnostics(
                            action_xyz=action_env,
                            ee_before=obs.get("ee_position"),
                            ee_after=next_obs.get("ee_position"),
                            action_step_xyz=args.action_step_xyz,
                        )
                        env_reward, env_reward_raw, env_reward_clipped, env_reward_non_finite = _sanitize_env_reward(
                            reward_raw=env_reward_raw_value,
                            clip_abs=args.reward_clip_abs,
                            fallback_reward=args.unstable_reward_penalty,
                        )
                        if env_reward_clipped:
                            reward_clip_count += 1
                        if env_reward_non_finite:
                            reward_non_finite_count += 1

                        unstable, unstable_reason = _detect_unstable_transition(
                            env_reward_raw=env_reward_raw,
                            motion_diag=motion_diag,
                            unstable_gain_threshold=args.unstable_gain_threshold,
                            unstable_realized_xyz_norm_threshold=args.unstable_realized_xyz_norm_threshold,
                            unstable_env_reward_abs_threshold=args.unstable_env_reward_abs_threshold,
                        )
                        forced_unstable_reset = bool(args.guard_unstable_transitions and unstable)
                        if forced_unstable_reset:
                            unstable_transition_count += 1
                            env_done = True
                            env_reward = float(args.unstable_reward_penalty)

                        step_info["unstable_transition"] = bool(unstable)
                        step_info["unstable_reason"] = str(unstable_reason)
                        step_info["forced_unstable_reset"] = bool(forced_unstable_reset)
                        step_info["reward_env_raw"] = _float_or_none(env_reward_raw)
                        step_info["reward_env_clipped"] = bool(env_reward_clipped)
                        step_info["reward_env_non_finite"] = bool(env_reward_non_finite)

                        dist_after = _distance_ee_to_target_from_obs(next_obs)
                        target_object_catalog, target_object_body, target_object_name = _extract_target_object_fields(step_info)
                        reward, closer_bonus, farther_penalty, raw_dist_delta = _shape_reward_with_delta_progress(
                            env_reward=env_reward,
                            distance_before=dist_before,
                            distance_after=dist_after,
                            delta_closer_reward_coef=args.delta_closer_reward_coef,
                            delta_farther_penalty_coef=args.delta_farther_penalty_coef,
                        )
                        reward_components = _extract_reward_components(step_info if isinstance(step_info, dict) else {})
                        done = bool(env_done or forced_scene_refresh or forced_unstable_reset)
                        post_state = None if done else env.capture_state()

                        candidate = CandidateResult(
                            reward=float(reward),
                            env_reward=float(env_reward),
                            env_reward_raw=_float_or_none(env_reward_raw),
                            env_reward_clipped=bool(env_reward_clipped),
                            env_reward_non_finite=bool(env_reward_non_finite),
                            done=bool(done),
                            env_done=bool(env_done),
                            forced_scene_refresh=bool(forced_scene_refresh),
                            unstable=bool(unstable),
                            unstable_reason=str(unstable_reason),
                            forced_unstable_reset=bool(forced_unstable_reset),
                            step_info=step_info,
                            reward_components=reward_components,
                            motion_diag=motion_diag,
                            closer_bonus=float(closer_bonus),
                            farther_penalty=float(farther_penalty),
                            raw_dist_delta=float(raw_dist_delta),
                            dist_before=_float_or_none(dist_before),
                            dist_after=_float_or_none(dist_after),
                            target_object_catalog=target_object_catalog,
                            target_object_body=target_object_body,
                            target_object_name=target_object_name,
                            next_obs=next_obs,
                            post_state=post_state,
                        )
                        candidate_results.append(candidate)
                        candidate_rewards.append(float(reward))
                        env.restore_state(base_state)

                    group_advantages = _group_relative_advantages(
                        np.asarray(candidate_rewards, dtype=np.float32),
                        normalize=bool(args.grpo_normalize_group_advantage),
                        eps=float(args.grpo_advantage_eps),
                        clip_abs=float(args.grpo_clip_advantage_abs),
                    )
                    group_advantages_all.extend(float(x) for x in group_advantages.tolist())

                    selected_idx = _select_group_index(args, branch_rng)
                    selected = candidate_results[selected_idx]

                    for group_idx, candidate in enumerate(candidate_results):
                        transitions.append(
                            Transition(
                                img_primary=obs["image_primary"],
                                img_wrist=obs["image_wrist"] if args.num_images_in_input > 1 else None,
                                instruction=obs["instruction"],
                                action=sampled_actions_group[group_idx][env_idx],
                                logprob=float(logprob_group[group_idx][env_idx]),
                                env_reward=float(candidate.env_reward),
                                reward=float(candidate.reward),
                                advantage=float(group_advantages[group_idx]),
                            )
                        )

                    selected_rewards.append(float(selected.reward))
                    selected_env_rewards.append(float(selected.env_reward))
                    reward_xyz_values.append(float(selected.reward_components["r_xyz"]))
                    reward_orient_values.append(float(selected.reward_components["r_orient"]))
                    reward_obj_values.append(float(selected.reward_components["r_obj"]))
                    reward_success_values.append(float(selected.reward_components["r_success"]))
                    if selected.motion_diag["realized_vs_command_gain"] is not None:
                        motion_gain_values.append(float(selected.motion_diag["realized_vs_command_gain"]))
                    if selected.motion_diag["realized_vs_command_cosine"] is not None:
                        motion_cosine_values.append(float(selected.motion_diag["realized_vs_command_cosine"]))

                    rollout_records.append(
                        {
                            "global_step": int(global_step + 1),
                            "step_in_update": int(len(rollout_records)),
                            "selected_group_index": int(selected_idx),
                            "group_size": int(args.grpo_group_size),
                            "group_advantages": [float(x) for x in group_advantages.tolist()],
                            "group_rewards": [float(x) for x in candidate_rewards],
                            "action_delta": env_actions_group[selected_idx][env_idx].astype(np.float32),
                            "action_delta_norm": float(np.linalg.norm(env_actions_group[selected_idx][env_idx])),
                            "reward_env": float(selected.env_reward),
                            "reward_env_raw": selected.env_reward_raw,
                            "reward_env_clipped": bool(selected.env_reward_clipped),
                            "reward_env_non_finite": bool(selected.env_reward_non_finite),
                            "reward_shaped": float(selected.reward),
                            "r_xyz": float(selected.reward_components["r_xyz"]),
                            "r_orient": float(selected.reward_components["r_orient"]),
                            "r_obj": float(selected.reward_components["r_obj"]),
                            "r_success": float(selected.reward_components["r_success"]),
                            "commanded_xyz_delta": selected.motion_diag["commanded_xyz_delta"],
                            "realized_xyz_delta": selected.motion_diag["realized_xyz_delta"],
                            "realized_vs_command_gain": selected.motion_diag["realized_vs_command_gain"],
                            "realized_vs_command_cosine": selected.motion_diag["realized_vs_command_cosine"],
                            "closer_bonus": float(selected.closer_bonus),
                            "farther_penalty": float(selected.farther_penalty),
                            "distance_before": selected.dist_before,
                            "distance_after": selected.dist_after,
                            "distance_delta_raw": float(selected.raw_dist_delta),
                            "ee_position_before": obs.get("ee_position"),
                            "ee_position_after": selected.next_obs.get("ee_position"),
                            "target_position_before": obs.get("target_object_position"),
                            "target_position_after": selected.next_obs.get("target_object_position"),
                            "all_object_positions_before": obs.get("all_object_positions"),
                            "all_object_positions_after": selected.next_obs.get("all_object_positions"),
                            "object_position_mask_before": obs.get("object_position_mask"),
                            "object_position_mask_after": selected.next_obs.get("object_position_mask"),
                            "logprob": float(logprob_group[selected_idx][env_idx]),
                            "scene": str(selected.step_info.get("scene", "")),
                            "instruction": str(obs.get("instruction", "")),
                            "instruction_type": str(selected.step_info.get("instruction_type", "")),
                            "target_object_catalog": selected.target_object_catalog,
                            "target_object_body": selected.target_object_body,
                            "target_object_name": selected.target_object_name,
                            "env_done": bool(selected.env_done),
                            "forced_scene_refresh": bool(selected.forced_scene_refresh),
                            "unstable_transition": bool(selected.unstable),
                            "unstable_reason": str(selected.unstable_reason),
                            "forced_unstable_reset": bool(selected.forced_unstable_reset),
                        }
                    )

                    steps_since_reset[env_idx] = next_step_count
                    global_step += 1
                    done = bool(selected.done)
                    ep_ret_running[env_idx] += float(selected.reward)
                    ep_ret_env_running[env_idx] += float(selected.env_reward)

                    if done:
                        episode_returns.append(float(ep_ret_running[env_idx]))
                        episode_returns_env.append(float(ep_ret_env_running[env_idx]))
                        ep_ret_running[env_idx] = 0.0
                        ep_ret_env_running[env_idx] = 0.0
                        steps_since_reset[env_idx] = 0
                        episode_idx += 1

                        if trace_fp is not None:
                            event = {
                                "update": int(update),
                                "episode": int(episode_idx),
                                "global_step": int(global_step),
                                "scene": str(selected.step_info.get("scene", "")),
                                "target_object_catalog": selected.target_object_catalog,
                                "target_object_body": selected.target_object_body,
                                "target_object_name": selected.target_object_name,
                                "instruction": str(selected.step_info.get("language_instruction", "")),
                                "instruction_type": str(selected.step_info.get("instruction_type", "")),
                                "success": bool(selected.step_info.get("success", False)),
                                "env_done": bool(selected.env_done),
                                "forced_scene_refresh": bool(selected.forced_scene_refresh),
                                "forced_unstable_reset": bool(selected.forced_unstable_reset),
                                "unstable_transition": bool(selected.unstable),
                                "unstable_reason": str(selected.unstable_reason),
                                "reward_env": float(selected.env_reward),
                                "reward_env_raw": selected.env_reward_raw,
                                "reward_env_clipped": bool(selected.env_reward_clipped),
                                "reward_env_non_finite": bool(selected.env_reward_non_finite),
                                "reward_shaped": float(selected.reward),
                                "r_xyz": float(selected.reward_components["r_xyz"]),
                                "r_orient": float(selected.reward_components["r_orient"]),
                                "r_obj": float(selected.reward_components["r_obj"]),
                                "r_success": float(selected.reward_components["r_success"]),
                                "commanded_xyz_delta": np.asarray(
                                    selected.motion_diag["commanded_xyz_delta"],
                                    dtype=np.float32,
                                ).tolist(),
                                "realized_xyz_delta": np.asarray(
                                    selected.motion_diag["realized_xyz_delta"],
                                    dtype=np.float32,
                                ).tolist(),
                                "realized_vs_command_gain": selected.motion_diag["realized_vs_command_gain"],
                                "realized_vs_command_cosine": selected.motion_diag["realized_vs_command_cosine"],
                                "closer_bonus": float(selected.closer_bonus),
                                "farther_penalty": float(selected.farther_penalty),
                                "distance_delta_raw": float(selected.raw_dist_delta),
                                "distance_before": selected.dist_before,
                                "distance_after": selected.dist_after,
                                "desk_texture": str(selected.step_info.get("desk_texture", "")),
                                "env_index": int(env_idx),
                            }
                            trace_fp.write(json.dumps(event, ensure_ascii=False) + "\n")
                            trace_fp.flush()

                        with ppo._silence_stdio(bool(args.quiet_env_logs)):
                            obs_batch[env_idx] = env.reset(options=next_reset_options())
                    else:
                        if selected.post_state is None:
                            raise RuntimeError("Expected a continuation state for non-terminal GRPO candidate.")
                        env.restore_state(selected.post_state)
                        obs_batch[env_idx] = env.observe()

                if rollout_pbar is not None:
                    rollout_pbar.update(1)

            if rollout_pbar is not None:
                rollout_pbar.close()

            rewards = np.asarray([transition.reward for transition in transitions], dtype=np.float32)
            env_rewards = np.asarray([transition.env_reward for transition in transitions], dtype=np.float32)
            actions = np.asarray([transition.action for transition in transitions], dtype=np.float32)
            old_logprobs = np.asarray([transition.logprob for transition in transitions], dtype=np.float32)
            advantages = np.asarray([transition.advantage for transition in transitions], dtype=np.float32)

            policy.train()
            idxs = np.arange(len(transitions))
            total_minibatches = args.ppo_epochs * max(1, math.ceil(len(idxs) / args.minibatch_size))
            train_pbar = (
                tqdm(
                    total=total_minibatches,
                    desc=f"u{update:05d} train",
                    dynamic_ncols=True,
                    leave=False,
                )
                if use_tqdm
                else None
            )

            stop_training = False
            for _ in range(args.ppo_epochs):
                np.random.shuffle(idxs)
                for start in range(0, len(idxs), args.minibatch_size):
                    mb_idx = idxs[start : start + args.minibatch_size]
                    if len(mb_idx) == 0:
                        continue

                    mb_adv_all = torch.tensor(advantages[mb_idx], dtype=torch.float32, device=device)
                    if args.normalize_advantage and mb_adv_all.numel() > 1:
                        mb_adv_all = (mb_adv_all - mb_adv_all.mean()) / (mb_adv_all.std(unbiased=False) + 1e-8)

                    optimizer.zero_grad(set_to_none=True)
                    micro_splits = max(1, math.ceil(len(mb_idx) / args.microbatch_size))
                    stop_on_target_kl = False

                    for micro_start in range(0, len(mb_idx), args.microbatch_size):
                        micro_end = micro_start + args.microbatch_size
                        micro_idx = mb_idx[micro_start:micro_end]
                        if len(micro_idx) == 0:
                            continue

                        is_last_micro = (micro_start + args.microbatch_size) >= len(mb_idx)
                        sync_ctx = (
                            nullcontext()
                            if (not isinstance(policy, DDP) or is_last_micro)
                            else policy.no_sync()
                        )
                        with sync_ctx:
                            mb_imgs_primary = [transitions[i].img_primary for i in micro_idx]
                            mb_imgs_wrist = (
                                [transitions[i].img_wrist for i in micro_idx] if args.num_images_in_input > 1 else None
                            )
                            mb_instr = [transitions[i].instruction for i in micro_idx]

                            mean_action, std_action, _, mean_pre_action = policy(
                                images_primary=mb_imgs_primary,
                                images_wrist=mb_imgs_wrist,
                                instructions=mb_instr,
                            )

                            mb_actions = torch.tensor(actions[micro_idx], dtype=torch.float32, device=device)
                            mb_old_logprobs = torch.tensor(old_logprobs[micro_idx], dtype=torch.float32, device=device)
                            mb_adv = mb_adv_all[micro_start:micro_end]

                            new_logprob = ppo.squashed_gaussian_log_prob(
                                mb_actions,
                                mean_pre_action,
                                std_action,
                            ).sum(dim=-1)
                            log_ratio = new_logprob - mb_old_logprobs
                            ratio = torch.exp(log_ratio)

                            pg_loss1 = -mb_adv * ratio
                            pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                            policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                            entropy = ppo.gaussian_entropy(std_action).sum(dim=-1).mean() / float(NUM_ACTIONS_CHUNK)
                            entropy_loss = -entropy
                            loss = policy_loss + args.ent_coef * entropy_loss

                            with torch.no_grad():
                                approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
                                clip_fraction = (torch.abs(ratio - 1.0) > args.clip_coef).float().mean()
                                approx_kl_values.append(float(approx_kl.item()))
                                clip_fraction_values.append(float(clip_fraction.item()))
                                loss_policy_values.append(float(policy_loss.item()))
                                loss_entropy_values.append(float(entropy.item()))
                                loss_total_values.append(float(loss.item()))

                                if args.target_kl is not None and float(approx_kl.item()) > (1.5 * float(args.target_kl)):
                                    stop_on_target_kl = True

                            if stop_on_target_kl:
                                break
                            (loss / micro_splits).backward()

                    if stop_on_target_kl:
                        optimizer.zero_grad(set_to_none=True)
                        stop_training = True
                        break

                    nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    optimizer.step()
                    if train_pbar is not None:
                        train_pbar.update(1)

                if stop_training:
                    break

            if train_pbar is not None:
                train_pbar.close()

            if stop_training and is_main and args.target_kl is not None:
                latest_kl = approx_kl_values[-1] if approx_kl_values else 0.0
                print(
                    f"[update {update:05d}] early-stop GRPO epochs by target_kl={args.target_kl:.6f} "
                    f"(approx_kl={latest_kl:.6f})",
                    flush=True,
                )

            avg_rollout_reward = float(np.mean(selected_rewards)) if selected_rewards else 0.0
            avg_rollout_reward_env = float(np.mean(selected_env_rewards)) if selected_env_rewards else 0.0
            avg_return = float(np.mean(episode_returns)) if episode_returns else float(np.mean(ep_ret_running))
            avg_return_env = float(np.mean(episode_returns_env)) if episode_returns_env else float(np.mean(ep_ret_env_running))
            avg_policy_loss = float(np.mean(loss_policy_values)) if loss_policy_values else 0.0
            avg_value_loss = 0.0
            avg_entropy = float(np.mean(loss_entropy_values)) if loss_entropy_values else 0.0
            avg_total_loss = float(np.mean(loss_total_values)) if loss_total_values else 0.0
            avg_approx_kl = float(np.mean(approx_kl_values)) if approx_kl_values else 0.0
            avg_clip_fraction = float(np.mean(clip_fraction_values)) if clip_fraction_values else 0.0
            avg_r_xyz = float(np.mean(reward_xyz_values)) if reward_xyz_values else 0.0
            avg_r_orient = float(np.mean(reward_orient_values)) if reward_orient_values else 0.0
            avg_r_obj = float(np.mean(reward_obj_values)) if reward_obj_values else 0.0
            avg_r_success = float(np.mean(reward_success_values)) if reward_success_values else 0.0
            avg_motion_gain = float(np.mean(motion_gain_values)) if motion_gain_values else 0.0
            avg_motion_cosine = float(np.mean(motion_cosine_values)) if motion_cosine_values else 0.0
            unstable_transition_rate = float(unstable_transition_count / max(1, len(transitions)))
            reward_clip_rate = float(reward_clip_count / max(1, len(transitions)))
            reward_non_finite_rate = float(reward_non_finite_count / max(1, len(transitions)))
            avg_group_advantage = float(np.mean(group_advantages_all)) if group_advantages_all else 0.0
            std_group_advantage = float(np.std(group_advantages_all)) if group_advantages_all else 0.0

            if updates_pbar is not None:
                updates_pbar.update(1)
                updates_pbar.set_postfix(
                    step=int(global_step),
                    r_env=f"{avg_rollout_reward_env:.3f}",
                    r_shape=f"{avg_rollout_reward:.3f}",
                    ep_env=f"{avg_return_env:.3f}",
                    ep_shape=f"{avg_return:.3f}",
                    l_pi=f"{avg_policy_loss:.3f}",
                    kl=f"{avg_approx_kl:.4f}",
                    unstable=f"{unstable_transition_rate:.3f}",
                    log_std=f"{float(policy_core.log_std.mean().item()):.3f}",
                )
            elif is_main:
                print(
                    f"[update {update:05d}] "
                    f"global_step={global_step} "
                    f"rollout_reward_env_mean={avg_rollout_reward_env:.4f} "
                    f"rollout_reward_shaped_mean={avg_rollout_reward:.4f} "
                    f"episode_return_env_mean={avg_return_env:.4f} "
                    f"episode_return_shaped_mean={avg_return:.4f} "
                    f"loss_policy_mean={avg_policy_loss:.4f} "
                    f"loss_value_mean={avg_value_loss:.4f} "
                    f"entropy_mean={avg_entropy:.4f} "
                    f"loss_total_mean={avg_total_loss:.4f} "
                    f"approx_kl_mean={avg_approx_kl:.6f} "
                    f"clip_fraction_mean={avg_clip_fraction:.4f} "
                    f"r_xyz_mean={avg_r_xyz:.4f} "
                    f"r_orient_mean={avg_r_orient:.4f} "
                    f"r_obj_mean={avg_r_obj:.4f} "
                    f"r_success_mean={avg_r_success:.4f} "
                    f"motion_gain_mean={avg_motion_gain:.4f} "
                    f"motion_cosine_mean={avg_motion_cosine:.4f} "
                    f"unstable_transition_rate={unstable_transition_rate:.4f} "
                    f"reward_clip_rate={reward_clip_rate:.4f} "
                    f"reward_non_finite_rate={reward_non_finite_rate:.4f} "
                    f"group_adv_mean={avg_group_advantage:.4f} "
                    f"group_adv_std={std_group_advantage:.4f} "
                    f"log_std_mean={float(policy_core.log_std.mean().item()):.4f}",
                    flush=True,
                )
            if (
                is_main
                and tb_writer is not None
                and args.tensorboard_every_updates > 0
                and (update % args.tensorboard_every_updates == 0)
            ):
                tb_writer.add_scalar("train/reward_env_mean", avg_rollout_reward_env, global_step)
                tb_writer.add_scalar("train/reward_shaped_mean", avg_rollout_reward, global_step)
                tb_writer.add_scalar("train/episode_return_env_mean", avg_return_env, global_step)
                tb_writer.add_scalar("train/episode_return_shaped_mean", avg_return, global_step)
                tb_writer.add_scalar("train/loss_policy_mean", avg_policy_loss, global_step)
                tb_writer.add_scalar("train/loss_value_mean", avg_value_loss, global_step)
                tb_writer.add_scalar("train/entropy_mean", avg_entropy, global_step)
                tb_writer.add_scalar("train/loss_total_mean", avg_total_loss, global_step)
                tb_writer.add_scalar("train/approx_kl_mean", avg_approx_kl, global_step)
                tb_writer.add_scalar("train/clip_fraction_mean", avg_clip_fraction, global_step)
                tb_writer.add_scalar("train/reward_component_r_xyz_mean", avg_r_xyz, global_step)
                tb_writer.add_scalar("train/reward_component_r_orient_mean", avg_r_orient, global_step)
                tb_writer.add_scalar("train/reward_component_r_obj_mean", avg_r_obj, global_step)
                tb_writer.add_scalar("train/reward_component_r_success_mean", avg_r_success, global_step)
                tb_writer.add_scalar("train/motion_realized_vs_command_gain_mean", avg_motion_gain, global_step)
                tb_writer.add_scalar("train/motion_realized_vs_command_cosine_mean", avg_motion_cosine, global_step)
                tb_writer.add_scalar("train/unstable_transition_rate", unstable_transition_rate, global_step)
                tb_writer.add_scalar("train/reward_clip_rate", reward_clip_rate, global_step)
                tb_writer.add_scalar("train/reward_non_finite_rate", reward_non_finite_rate, global_step)
                tb_writer.add_scalar("train/group_advantage_mean", avg_group_advantage, global_step)
                tb_writer.add_scalar("train/group_advantage_std", std_group_advantage, global_step)
                tb_writer.add_scalar("train/log_std_mean", float(policy_core.log_std.mean().item()), global_step)
                tb_writer.add_scalar("train/update_index", float(update), global_step)
                tb_writer.flush()

            if (
                is_main
                and args.rollout_tap_every_updates > 0
                and (update % args.rollout_tap_every_updates == 0)
            ):
                tap_path = save_rollout_tap_npz(
                    run_dir=run_dir,
                    update=update,
                    records=rollout_records,
                )
                if tap_path is not None:
                    print(
                        f"[rollout_tap u{update:05d}] path={tap_path} steps={len(rollout_records)}",
                        flush=True,
                    )

            if (
                is_main
                and val_env is not None
                and args.validate_every_updates > 0
                and (update % args.validate_every_updates == 0)
            ):
                val_summary = run_validation_rollouts(
                    policy_core=policy_core,
                    val_env=val_env,
                    run_dir=run_dir,
                    update=update,
                    num_episodes=args.validation_episodes,
                    max_steps=args.validation_max_steps,
                    num_images_in_input=args.num_images_in_input,
                    action_step_xyz=args.action_step_xyz,
                    delta_closer_reward_coef=args.delta_closer_reward_coef,
                    delta_farther_penalty_coef=args.delta_farther_penalty_coef,
                    reward_clip_abs=args.reward_clip_abs,
                    guard_unstable_transitions=bool(args.guard_unstable_transitions),
                    unstable_reward_penalty=args.unstable_reward_penalty,
                    unstable_gain_threshold=args.unstable_gain_threshold,
                    unstable_realized_xyz_norm_threshold=args.unstable_realized_xyz_norm_threshold,
                    unstable_env_reward_abs_threshold=args.unstable_env_reward_abs_threshold,
                    save_frames=bool(args.save_validation_frames),
                    quiet_env_logs=bool(args.quiet_env_logs),
                    next_reset_options=next_val_reset_options,
                )
                print(
                    f"[validation u{update:05d}] "
                    f"env_return_mean={val_summary['mean_env_return']:.4f} "
                    f"shaped_return_mean={val_summary['mean_shaped_return']:.4f} "
                    f"success_rate={val_summary['success_rate']:.3f} "
                    f"path={val_summary['path']}",
                    flush=True,
                )
                reward_means = val_summary.get("reward_component_means", {})
                if reward_means:
                    print(
                        f"[validation u{update:05d}] reward_components "
                        f"r_xyz={float(reward_means.get('r_xyz', 0.0)):.4f} "
                        f"r_orient={float(reward_means.get('r_orient', 0.0)):.4f} "
                        f"r_obj={float(reward_means.get('r_obj', 0.0)):.4f} "
                        f"r_success={float(reward_means.get('r_success', 0.0)):.4f}",
                        flush=True,
                    )
                motion_diag_val = val_summary.get("motion_diagnostics", {})
                if motion_diag_val:
                    print(
                        f"[validation u{update:05d}] motion "
                        f"gain={float(motion_diag_val.get('realized_vs_command_gain_mean', 0.0)):.4f} "
                        f"cosine={float(motion_diag_val.get('realized_vs_command_cosine_mean', 0.0)):.4f}",
                        flush=True,
                    )
                stability_val = val_summary.get("stability", {})
                if stability_val:
                    print(
                        f"[validation u{update:05d}] stability "
                        f"unstable={int(stability_val.get('unstable_transition_count', 0))} "
                        f"reward_clip={int(stability_val.get('reward_clip_count', 0))} "
                        f"reward_non_finite={int(stability_val.get('reward_non_finite_count', 0))}",
                        flush=True,
                    )
                xyz_stats = val_summary.get("action_xyz_stats", {})
                if xyz_stats:
                    print(
                        f"[validation u{update:05d}] action_xyz "
                        f"x(mean={xyz_stats['x']['mean']:.3f}, std={xyz_stats['x']['std']:.3f}) "
                        f"y(mean={xyz_stats['y']['mean']:.3f}, std={xyz_stats['y']['std']:.3f}) "
                        f"z(mean={xyz_stats['z']['mean']:.3f}, std={xyz_stats['z']['std']:.3f}) "
                        f"targets={','.join(val_summary.get('target_objects_seen', []))}",
                        flush=True,
                    )
                action_dim_stats = val_summary.get("action_dim_stats", {})
                if action_dim_stats:
                    sat_parts: List[str] = []
                    for axis in ("x", "z", "yaw", "gripper"):
                        axis_stats = action_dim_stats.get(axis)
                        if not isinstance(axis_stats, dict):
                            continue
                        sat_parts.append(
                            f"{axis}={100.0 * float(axis_stats.get('sat_frac_abs_ge_0_99', 0.0)):.1f}%"
                        )
                    if sat_parts:
                        print(
                            f"[validation u{update:05d}] action_saturation " + " ".join(sat_parts),
                            flush=True,
                        )
                if tb_writer is not None:
                    tb_writer.add_scalar("validation/env_return_mean", float(val_summary["mean_env_return"]), global_step)
                    tb_writer.add_scalar("validation/shaped_return_mean", float(val_summary["mean_shaped_return"]), global_step)
                    tb_writer.add_scalar("validation/success_rate", float(val_summary["success_rate"]), global_step)
                    tb_writer.add_scalar(
                        "validation/reward_component_r_xyz_mean",
                        float(reward_means.get("r_xyz", 0.0)),
                        global_step,
                    )
                    tb_writer.add_scalar(
                        "validation/reward_component_r_orient_mean",
                        float(reward_means.get("r_orient", 0.0)),
                        global_step,
                    )
                    tb_writer.add_scalar(
                        "validation/reward_component_r_obj_mean",
                        float(reward_means.get("r_obj", 0.0)),
                        global_step,
                    )
                    tb_writer.add_scalar(
                        "validation/reward_component_r_success_mean",
                        float(reward_means.get("r_success", 0.0)),
                        global_step,
                    )
                    tb_writer.add_scalar(
                        "validation/motion_realized_vs_command_gain_mean",
                        float(motion_diag_val.get("realized_vs_command_gain_mean", 0.0)),
                        global_step,
                    )
                    tb_writer.add_scalar(
                        "validation/motion_realized_vs_command_cosine_mean",
                        float(motion_diag_val.get("realized_vs_command_cosine_mean", 0.0)),
                        global_step,
                    )
                    axis_samples = val_summary.get("action_xyz_samples", {})
                    for axis in ("x", "y", "z"):
                        stats_axis = xyz_stats.get(axis, {})
                        tb_writer.add_scalar(
                            f"validation/action_{axis}_mean",
                            float(stats_axis.get("mean", 0.0)),
                            global_step,
                        )
                        tb_writer.add_scalar(
                            f"validation/action_{axis}_std",
                            float(stats_axis.get("std", 0.0)),
                            global_step,
                        )
                        vals = np.asarray(axis_samples.get(axis, []), dtype=np.float32)
                        if vals.size > 0:
                            tb_writer.add_histogram(f"validation/action_{axis}_hist", vals, global_step)
                    for axis, axis_stats in action_dim_stats.items():
                        if not isinstance(axis_stats, dict):
                            continue
                        tb_writer.add_scalar(
                            f"validation/action_{axis}_sat_frac_abs_ge_0_99",
                            float(axis_stats.get("sat_frac_abs_ge_0_99", 0.0)),
                            global_step,
                        )

            if is_main and update % args.save_every == 0:
                save_checkpoint(
                    run_dir=run_dir,
                    step=global_step,
                    vla=policy_core.vla,
                    action_head=policy_core.action_head,
                    log_std=policy_core.log_std,
                )

        if is_main:
            save_checkpoint(
                run_dir=run_dir,
                step=global_step,
                vla=policy_core.vla,
                action_head=policy_core.action_head,
                log_std=policy_core.log_std,
            )
    finally:
        if updates_pbar is not None:
            updates_pbar.close()
        if trace_fp is not None:
            trace_fp.close()
        if tb_writer is not None:
            tb_writer.close()
        if envs:
            with ppo._silence_stdio(bool(args.quiet_env_logs)):
                for env in envs:
                    env.close()
        if val_env is not None:
            with ppo._silence_stdio(bool(args.quiet_env_logs)):
                val_env.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
