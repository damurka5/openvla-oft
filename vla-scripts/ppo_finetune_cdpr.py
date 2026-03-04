#!/usr/bin/env python3
"""
ppo_finetune_cdpr.py

PPO fine-tuning for OpenVLA-OFT on the CDPR language-conditioned environment.

Key design:
- Vision + language conditioned policy (no state-only shortcut).
- Fine-tunes only LoRA adapters in OpenVLA + continuous action head + PPO value head.
- Uses PPO with GAE on continuous actions in [-1, 1]^ACTION_DIM.

This script intentionally avoids RLDS dataloader/shuffle-buffer logic from supervised finetuning;
PPO is on-policy and requires environment interaction rollouts.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import re
import shutil
import sys
import time
import json
import inspect
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.openvla_utils import check_model_logic_mismatch, model_is_on_hf_hub, update_auto_map
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="PPO finetune OpenVLA-OFT on CDPR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model / adapters
    ap.add_argument("--vla_path", type=str, default="openvla/openvla-7b")
    ap.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help=(
            "Optional path to an existing LoRA adapter to continue from. "
            "Can be either adapter dir itself or checkpoint dir containing `vla_cdpr_adapter/`."
        ),
    )
    ap.add_argument(
        "--train_loaded_adapter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When --adapter_path is provided, train LoRA params (default: freeze loaded adapter and "
            "train only action/value heads)."
        ),
    )
    ap.add_argument(
        "--action_head_path",
        type=str,
        default=None,
        help=(
            "Optional path to existing action-head checkpoint. Can be a .pt file or a checkpoint dir "
            "containing `action_head_cdpr.pt`."
        ),
    )
    ap.add_argument(
        "--strict_action_head_load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require exact key match when loading --action_head_path.",
    )

    # Environment
    ap.add_argument(
        "--cdpr_dataset_root",
        type=str,
        default=os.environ.get("CDPR_DATASET_ROOT", "/root/repo/CDPR-Dataset"),
        help=(
            "Path to CDPR-Dataset repo root (directory containing `cdpr_dataset/`). "
            "If `/root/repo` is given, script auto-resolves `/root/repo/CDPR-Dataset`."
        ),
    )
    ap.add_argument(
        "--cdpr_mujoco_root",
        type=str,
        default=os.environ.get("CDPR_MUJOCO_ROOT", None),
        help=(
            "Optional path to parent directory that contains `cdpr_mujoco/` "
            "(e.g. `/root/repo/VLA_CDPR`). Auto-detected if omitted."
        ),
    )
    ap.add_argument(
        "--catalog_path",
        type=str,
        default=None,
        help="Optional cdpr_scene_catalog.yaml path. If omitted, CDPR env default is used.",
    )
    ap.add_argument("--max_env_steps", type=int, default=150)
    ap.add_argument(
        "--action_step_xyz",
        type=float,
        default=0.005,
        help="World-frame XYZ delta scale per action step (teleop commonly uses 0.005-0.01).",
    )
    ap.add_argument(
        "--action_step_yaw",
        type=float,
        default=0.25,
        help="Yaw delta scale per action step.",
    )
    ap.add_argument("--capture_frames", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--desk_textures_dir",
        type=str,
        required=True,
        help="Path to desk textures directory used for per-episode randomization.",
    )
    ap.add_argument(
        "--allowed_objects",
        nargs="*",
        default=["bowl", "plate", "ycb_bowl", "ycb_apple", "ycb_peach", "ycb_pear"],
        help="Restrict target objects to this list.",
    )
    ap.add_argument(
        "--desk_geom_regex",
        type=str,
        default=r"(table|desk|workbench|counter|surface)",
        help="Regex to identify desk geoms for texture patching.",
    )
    ap.add_argument(
        "--desk_texrepeat",
        type=int,
        nargs=2,
        default=(20, 20),
        help="Desk texture repeat x y applied in generated wrappers.",
    )
    ap.add_argument(
        "--wrapper_cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete temporary generated wrappers/textures after each episode.",
    )
    ap.add_argument(
        "--use_wrapper_cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse wrapper cache instead of making per-episode temporary wrappers.",
    )
    ap.add_argument(
        "--instruction_types",
        nargs="*",
        default=None,
        help="Optional subset: pick_up move_left move_right move_top move_bottom",
    )
    ap.add_argument(
        "--scene_sampling",
        type=str,
        choices=["env_random", "round_robin", "random"],
        default="env_random",
        help=(
            "How to pick scene on reset. "
            "`env_random` lets CDPR env sample; `round_robin` cycles catalog scenes; "
            "`random` samples a scene uniformly each reset."
        ),
    )
    ap.add_argument(
        "--scene_refresh_every_steps",
        type=int,
        default=-1,
        help=(
            "Force reset every N interaction steps to resample scene/object positions/instruction. "
            "Use <=0 to disable forced refresh and reset only on env termination."
        ),
    )
    ap.add_argument(
        "--prebuild_scene_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Prebuild a fixed cache of scene wrappers/textured variants once at startup and reuse them on reset "
            "(prevents per-reset wrapper/texture file growth)."
        ),
    )
    ap.add_argument(
        "--scene_pool_size",
        type=int,
        default=5,
        help="How many catalog scenes to keep in the reusable scene pool. Use <=0 for all scenes.",
    )
    ap.add_argument(
        "--texture_pool_size",
        type=int,
        default=10,
        help="How many desk textures to prebuild per scene in scene-cache mode. Use <=0 for all.",
    )
    ap.add_argument(
        "--invert_x_action",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Invert applied X action before sending command to env (for coordinate debugging).",
    )
    ap.add_argument(
        "--invert_y_action",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Invert applied Y action before sending command to env (for coordinate debugging).",
    )

    # PPO
    ap.add_argument("--total_updates", type=int, default=70)
    ap.add_argument("--rollout_steps", type=int, default=256)
    ap.add_argument("--ppo_epochs", type=int, default=4)
    ap.add_argument("--minibatch_size", type=int, default=4)
    ap.add_argument(
        "--microbatch_size",
        type=int,
        default=4,
        help=(
            "Split each PPO minibatch into smaller forward/backward chunks to lower peak GPU memory. "
            "Effective optimizer batch remains --minibatch_size."
        ),
    )
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--clip_coef", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--value_lr", type=float, default=1e-4)
    ap.add_argument("--init_log_std", type=float, default=-1.2)
    ap.add_argument(
        "--delta_closer_reward_coef",
        type=float,
        default=5.0,
        help=(
            "Extra reward coefficient for positive reduction in EE-to-target distance "
            "(delta action moves closer to target)."
        ),
    )
    ap.add_argument(
        "--delta_farther_penalty_coef",
        type=float,
        default=0.0,
        help=(
            "Optional penalty coefficient for negative reduction in EE-to-target distance "
            "(delta action moves farther from target)."
        ),
    )

    # Runtime
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing on VLA to reduce activation memory.",
    )
    ap.add_argument(
        "--ddp_find_unused_parameters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable DDP unused-parameter detection for safety with optional branches.",
    )
    ap.add_argument(
        "--ddp_static_graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use DDP static-graph mode when available (recommended with gradient checkpointing). "
            "Can avoid 'Expected to mark a variable ready only once' errors."
        ),
    )
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument(
        "--status_bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bars for updates/rollout/train (rank 0 only).",
    )
    ap.add_argument(
        "--quiet_env_logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress simulator/wrapper stdout/stderr during env init/reset/close.",
    )
    ap.add_argument(
        "--env_trace_path",
        type=str,
        default=None,
        help="Optional JSONL file to log scene/object/instruction/success per completed episode (rank 0).",
    )
    ap.add_argument(
        "--validate_every_updates",
        type=int,
        default=-1,
        help="Run deterministic validation rollouts in a separate env every N PPO updates (rank 0). Use <=0 to disable.",
    )
    ap.add_argument(
        "--validation_episodes",
        type=int,
        default=1,
        help="Number of validation episodes per validation run.",
    )
    ap.add_argument(
        "--validation_max_steps",
        type=int,
        default=40,
        help="Maximum steps per validation episode.",
    )
    ap.add_argument(
        "--save_validation_frames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save overview/wrist PNG frames during periodic validation rollouts.",
    )
    ap.add_argument(
        "--rollout_tap_every_updates",
        type=int,
        default=10,
        help=(
            "Dump training-rollout diagnostics to .npz every N updates using already-computed policy calls "
            "(no extra env resets/renders). Use <=0 to disable."
        ),
    )
    ap.add_argument("--run_root_dir", type=str, default="runs_ppo")
    ap.add_argument("--run_id", type=str, default=None)
    ap.add_argument("--num_images_in_input", type=int, default=2, choices=[1, 2])

    return ap.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_distributed() -> Tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size


def _is_main_process(rank: int) -> bool:
    return rank == 0


def _broadcast_object(obj: Any, rank: int) -> Any:
    if not (dist.is_available() and dist.is_initialized()):
        return obj
    payload = [obj if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _unwrap_module(module: nn.Module) -> nn.Module:
    if isinstance(module, DDP):
        return module.module
    return module


@contextmanager
def _silence_stdio(enabled: bool):
    if not enabled:
        yield
        return

    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


def make_run_dir(args: argparse.Namespace) -> Path:
    run_root = Path(args.run_root_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    if args.run_id is not None:
        run_name = args.run_id
    else:
        run_name = f"ppo_cdpr_{int(time.time())}"
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_config(args: argparse.Namespace, run_dir: Path) -> None:
    out = run_dir / "run_config.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)


def _strip_generated_texture_prefix(name: str) -> str:
    # Generated texture names in CDPR wrappers commonly look like:
    # rl_<stamp>__rl_<stamp>__...__<base_name>.png
    out = str(name)
    for _ in range(32):
        new_out = re.sub(r"^rl_[^_]+__", "", out)
        if new_out == out:
            break
        out = new_out
    return out


def _prepare_desk_textures_dir(
    src_dir: str,
    run_dir: Path,
    is_main: bool,
    rank: int,
    max_textures: int,
) -> str:
    src = Path(src_dir).expanduser().resolve()
    if not src.exists() or not src.is_dir():
        return str(src)

    texture_files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not texture_files:
        return str(src)

    # Collapse generated chains by canonical tail name and pick shortest representative.
    dedup: Dict[str, Path] = {}
    for p in texture_files:
        name = _strip_generated_texture_prefix(p.name)
        canonical_name = name.split("__")[-1]
        cur = dedup.get(canonical_name)
        if cur is None or len(p.name) < len(cur.name):
            dedup[canonical_name] = p
    base_candidates = sorted(dedup.values(), key=lambda x: x.name)
    if int(max_textures) > 0 and len(base_candidates) > int(max_textures):
        base_candidates = base_candidates[: int(max_textures)]

    # If already a small clean directory, keep original path.
    if len(base_candidates) == len(texture_files) and len(texture_files) <= max(1, int(max_textures)):
        return str(src)

    cleaned_dir = run_dir / "desk_textures_clean"
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    for idx, p in enumerate(base_candidates):
        ext = p.suffix.lower() if p.suffix else ".png"
        short_hash = hashlib.sha1(p.name.encode("utf-8")).hexdigest()[:8]
        dst = cleaned_dir / f"tex_{idx:03d}_{short_hash}{ext}"
        if dst.exists():
            continue
        try:
            os.symlink(str(p), str(dst))
        except Exception:
            shutil.copy2(str(p), str(dst))
        kept += 1

    if is_main:
        print(
            f"[env] Sanitized desk textures dir: {src} -> {cleaned_dir} "
            f"(kept {kept}/{len(texture_files)} files; dropped generated variants and shortened names).",
            flush=True,
        )
        if "_desk_textures" in str(src):
            print(
                "[WARN] desk_textures_dir points to wrappers cache. Prefer original source textures dir to avoid "
                "recursive rl_* texture names and disk growth.",
                flush=True,
            )

    out_path = str(cleaned_dir)
    return _broadcast_object(out_path if rank == 0 else None, rank)


def _read_scene_names_from_catalog(catalog_path: Optional[str]) -> List[str]:
    if catalog_path is None:
        return []
    path = Path(catalog_path).expanduser().resolve()
    if not path.exists():
        return []
    try:
        import yaml  # type: ignore
    except Exception:
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return []
    scenes_raw = cfg.get("scenes", []) if isinstance(cfg, dict) else []
    out: List[str] = []
    for row in scenes_raw:
        if isinstance(row, dict):
            name = row.get("name")
            if name:
                out.append(str(name))
        elif row:
            out.append(str(row))
    return out


def _prune_generated_wrapper_artifacts(
    cdpr_dataset_root: Path,
    is_main: bool,
    rank: int,
) -> None:
    wrappers_dir = cdpr_dataset_root.expanduser().resolve() / "cdpr_dataset" / "wrappers"
    if not wrappers_dir.exists():
        return

    removed = 0
    bytes_freed = 0
    patterns = [
        "**/*__rltmp_*.xml",
        "**/*__desktex_rl_*.xml",
        "_desk_textures/rl_*",
    ]
    seen: set[Path] = set()
    for pattern in patterns:
        for p in wrappers_dir.glob(pattern):
            pp = p.resolve()
            if pp in seen or (not pp.exists()) or (not pp.is_file()):
                continue
            seen.add(pp)
            try:
                bytes_freed += int(pp.stat().st_size)
            except Exception:
                pass
            try:
                pp.unlink()
                removed += 1
            except Exception:
                pass

    if is_main and removed > 0:
        gb = bytes_freed / (1024.0 ** 3)
        print(
            f"[env] Pruned stale generated wrapper artifacts: removed={removed}, freed={gb:.2f} GiB",
            flush=True,
        )

    _broadcast_object({"removed": removed}, rank)


def _resolve_and_prepare_vla_path(vla_path: str) -> str:
    # Mirror `finetune.py` behavior:
    # - For HF Hub models, download snapshot and use local path.
    # - For local models, register OpenVLA auto-classes.
    # - In both cases, sync config auto_map + local modeling/configuration files.
    if model_is_on_hf_hub(vla_path):
        resolved_path = snapshot_download(repo_id=vla_path)
    else:
        resolved_path = vla_path
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    update_auto_map(resolved_path)
    check_model_logic_mismatch(resolved_path)
    return resolved_path


def _iter_model_candidates(model: Any) -> List[Any]:
    out: List[Any] = []
    queue: List[Any] = [model]
    seen: set[int] = set()

    while queue:
        cur = queue.pop(0)
        if cur is None:
            continue
        oid = id(cur)
        if oid in seen:
            continue
        seen.add(oid)
        out.append(cur)

        for attr in ("module", "model", "base_model"):
            child = getattr(cur, attr, None)
            if child is not None and id(child) not in seen:
                queue.append(child)

    return out


def _resolve_vision_backbone(vla: nn.Module) -> Any:
    for obj in _iter_model_candidates(vla):
        vb = getattr(obj, "vision_backbone", None)
        if vb is not None:
            return vb
    return None


def _set_num_images_in_input(vla: nn.Module, num_images: int) -> int:
    n = int(num_images)

    for obj in _iter_model_candidates(vla):
        if hasattr(obj, "set_num_images_in_input"):
            obj.set_num_images_in_input(n)
            return n
        if hasattr(obj, "num_images_in_input"):
            setattr(obj, "num_images_in_input", n)
            print(
                "[WARN] Model has no set_num_images_in_input(); set num_images_in_input directly.",
                flush=True,
            )
            return n

    vision_backbone = _resolve_vision_backbone(vla)
    if vision_backbone is not None:
        if hasattr(vision_backbone, "set_num_images_in_input"):
            vision_backbone.set_num_images_in_input(n)
            return n
        if hasattr(vision_backbone, "num_images_in_input"):
            setattr(vision_backbone, "num_images_in_input", n)
            print(
                "[WARN] Vision backbone has no set_num_images_in_input(); "
                "set num_images_in_input directly for compatibility.",
                flush=True,
            )
            return n
        # Some older backbones may store image count in config only.
        vb_cfg = getattr(vision_backbone, "config", None)
        if vb_cfg is not None and hasattr(vb_cfg, "num_images_in_input"):
            setattr(vb_cfg, "num_images_in_input", n)
            print(
                "[WARN] Set vision_backbone.config.num_images_in_input directly for compatibility.",
                flush=True,
            )
            return n

    # Last-resort fallback for older checkpoints/classes that are fixed to 1 image.
    if n != 1:
        print(
            "[WARN] This OpenVLA build does not expose multi-image controls. "
            "Falling back to --num_images_in_input=1.",
            flush=True,
        )
    else:
        print(
            "[WARN] OpenVLA image-count control API not found; continuing in single-image mode.",
            flush=True,
        )
    return 1


def _resolve_llm_dim(vla: nn.Module) -> Optional[int]:
    for obj in _iter_model_candidates(vla):
        llm_dim = getattr(obj, "llm_dim", None)
        if llm_dim is not None:
            return int(llm_dim)

        cfg = getattr(obj, "config", None)
        if cfg is not None:
            text_cfg = getattr(cfg, "text_config", None)
            hidden = getattr(text_cfg, "hidden_size", None) if text_cfg is not None else None
            if hidden is not None:
                return int(hidden)
            hidden = getattr(cfg, "hidden_size", None)
            if hidden is not None:
                return int(hidden)

        lm = getattr(obj, "language_model", None)
        lm_cfg = getattr(lm, "config", None) if lm is not None else None
        hidden = getattr(lm_cfg, "hidden_size", None) if lm_cfg is not None else None
        if hidden is not None:
            return int(hidden)

    return None


def _resolve_cdpr_dataset_root(path_like: str | Path) -> Path:
    raw = Path(path_like).expanduser()
    if not raw.is_absolute():
        raw = raw.resolve()

    script_root = Path(__file__).resolve().parents[1]  # .../openvla-oft
    repo_parent = script_root.parent                   # .../repo

    candidates = [
        raw,
        raw / "CDPR-Dataset",
        repo_parent / "CDPR-Dataset",
        Path("/root/repo/CDPR-Dataset"),
    ]

    for candidate in candidates:
        c = candidate.expanduser().resolve()
        if (c / "cdpr_dataset").is_dir():
            return c

    attempted = ", ".join(str(c.expanduser().resolve()) for c in candidates)
    raise FileNotFoundError(
        "Could not locate CDPR-Dataset root. Expected a directory containing `cdpr_dataset/`. "
        f"Checked: {attempted}"
    )


def _resolve_cdpr_mujoco_parent(
    cdpr_root: Path,
    explicit_root: str | Path | None,
) -> Path | None:
    candidates: List[Path] = []

    if explicit_root:
        p = Path(explicit_root).expanduser().resolve()
        candidates.append(p)
        if p.name == "cdpr_mujoco":
            candidates.append(p.parent)

    candidates.extend(
        [
            cdpr_root.parent / "VLA_CDPR",
            cdpr_root.parent,
            Path("/root/repo/VLA_CDPR"),
            Path("/root/repo"),
        ]
    )

    seen: set[Path] = set()
    for base in candidates:
        b = base.resolve()
        if b in seen:
            continue
        seen.add(b)
        if (b / "cdpr_mujoco").is_dir():
            return b
    return None


def _prepend_to_pythonpath(path: Path) -> None:
    p = str(path.resolve())
    current = os.environ.get("PYTHONPATH", "")
    parts = [x for x in current.split(os.pathsep) if x]
    if p in parts:
        return
    new_parts = [p] + parts
    os.environ["PYTHONPATH"] = os.pathsep.join(new_parts)


def _apply_cdpr_env_runtime_shims(CDPRLanguageRLEnv: Any, rl_env_module: Any) -> None:
    """
    Runtime compatibility layer for CDPR env variants across MuJoCo Python bindings.

    Some builds expose `data.body_xpos`; others expose `data.xpos` or named accessors.
    """
    if getattr(CDPRLanguageRLEnv, "_openvla_compat_patched", False):
        return

    mj_mod = getattr(rl_env_module, "mj", None)
    orig_get_body_position = CDPRLanguageRLEnv._get_body_position

    def _get_body_position_compat(self, body_name: str) -> np.ndarray:
        try:
            return orig_get_body_position(self, body_name)
        except AttributeError as exc:
            if "body_xpos" not in str(exc):
                raise
            if mj_mod is None:
                raise

            bid = mj_mod.mj_name2id(self.sim.model, mj_mod.mjtObj.mjOBJ_BODY, body_name)
            if bid == -1:
                raise RuntimeError(f"Body '{body_name}' not found in MuJoCo model.")

            data = self.sim.data
            if hasattr(data, "xpos"):
                return np.asarray(data.xpos[bid], dtype=np.float32).copy()

            if hasattr(data, "body"):
                try:
                    return np.asarray(data.body(body_name).xpos, dtype=np.float32).copy()
                except Exception:
                    pass

            raise

    CDPRLanguageRLEnv._get_body_position = _get_body_position_compat
    CDPRLanguageRLEnv._openvla_compat_patched = True
    print("[env] Applied runtime compatibility shim for MuJoCo body position access.", flush=True)


def _resolve_adapter_dir(path_like: str | Path) -> Path:
    base = Path(path_like).expanduser().resolve()
    candidates = [base, base / "vla_cdpr_adapter"]
    for c in candidates:
        if c.is_dir() and (c / "adapter_config.json").exists():
            return c
    attempted = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not resolve adapter directory from `{base}`. Checked: {attempted}"
    )


def _resolve_action_head_path(path_like: str | Path) -> Path:
    base = Path(path_like).expanduser().resolve()
    if base.is_file():
        return base
    candidates = [base / "action_head_cdpr.pt", base / "action_head.pt"]
    for c in candidates:
        if c.is_file():
            return c
    attempted = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not resolve action-head checkpoint from `{base}`. Checked: {attempted}"
    )


def _extract_state_dict(maybe_state: Any) -> Dict[str, torch.Tensor]:
    if isinstance(maybe_state, dict):
        for key in ("state_dict", "model_state_dict", "action_head_state_dict"):
            if key in maybe_state and isinstance(maybe_state[key], dict):
                maybe_state = maybe_state[key]
                break

    if not isinstance(maybe_state, dict):
        raise TypeError(f"Checkpoint must contain a state dict, got type={type(maybe_state)}")

    out: Dict[str, torch.Tensor] = {}
    for key, value in maybe_state.items():
        if not isinstance(value, torch.Tensor):
            continue
        k = key[7:] if key.startswith("module.") else key
        out[k] = value

    if not out:
        raise RuntimeError("No tensor parameters found in checkpoint state dict.")
    return out


def load_vla_and_processor(args: argparse.Namespace, device: torch.device):
    resolved_vla_path = _resolve_and_prepare_vla_path(args.vla_path)
    print(f"[model] Loading VLA from: {resolved_vla_path}", flush=True)

    processor = AutoProcessor.from_pretrained(resolved_vla_path, trust_remote_code=True)
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    vla = AutoModelForVision2Seq.from_pretrained(
        resolved_vla_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    args.num_images_in_input = _set_num_images_in_input(vla, args.num_images_in_input)

    adapter_loaded = False
    if args.adapter_path:
        adapter_dir = _resolve_adapter_dir(args.adapter_path)
        vla = PeftModel.from_pretrained(
            vla,
            str(adapter_dir),
            is_trainable=bool(args.train_loaded_adapter),
        ).to(device)
        adapter_loaded = True
        print(
            f"[adapter] Loaded from {adapter_dir} | train_loaded_adapter={bool(args.train_loaded_adapter)}",
            flush=True,
        )
    elif args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=min(args.lora_rank, 16),
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "qkv",
                "proj",
                "fc1",
                "fc2",
            ],
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config).to(device)
        print("[adapter] Initialized new LoRA adapter for PPO fine-tuning.", flush=True)

    if adapter_loaded and not args.train_loaded_adapter:
        for p in vla.parameters():
            p.requires_grad = False
    elif adapter_loaded or args.use_lora:
        for name, p in vla.named_parameters():
            p.requires_grad = "lora_" in name
    else:
        for p in vla.parameters():
            p.requires_grad = False

    return vla, processor


def maybe_enable_gradient_checkpointing(vla: nn.Module, enabled: bool) -> None:
    if not enabled:
        return
    try:
        enabled_non_reentrant = False
        if hasattr(vla, "gradient_checkpointing_enable"):
            try:
                vla.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                enabled_non_reentrant = True
            except TypeError:
                vla.gradient_checkpointing_enable()
        base_model = getattr(vla, "base_model", None)
        if base_model is not None and hasattr(base_model, "gradient_checkpointing_enable"):
            try:
                base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                enabled_non_reentrant = True
            except TypeError:
                base_model.gradient_checkpointing_enable()
        cfg = getattr(vla, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = False
        # Some wrappers keep their own config object.
        base_cfg = getattr(base_model, "config", None)
        if base_cfg is not None and hasattr(base_cfg, "use_cache"):
            base_cfg.use_cache = False
        msg = "[model] Enabled gradient checkpointing."
        if enabled_non_reentrant:
            msg += " (use_reentrant=False)"
        print(msg, flush=True)
    except Exception as exc:
        print(f"[WARN] Could not enable gradient checkpointing: {exc}", flush=True)


def build_action_head(
    args: argparse.Namespace, llm_dim: int, device: torch.device
) -> L1RegressionActionHead:
    action_head = L1RegressionActionHead(
        input_dim=llm_dim,
        hidden_dim=llm_dim,
        action_dim=ACTION_DIM,
    ).to(device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)

    if args.action_head_path:
        action_head_path = _resolve_action_head_path(args.action_head_path)
        state_raw = torch.load(action_head_path, map_location="cpu")
        state = _extract_state_dict(state_raw)
        missing, unexpected = action_head.load_state_dict(state, strict=bool(args.strict_action_head_load))
        if (missing or unexpected) and bool(args.strict_action_head_load):
            raise RuntimeError(
                "Strict action-head load requested, but checkpoint keys mismatch. "
                f"missing={missing}, unexpected={unexpected}"
            )
        if missing or unexpected:
            print(
                f"[WARN] Non-strict action-head load. missing={missing}, unexpected={unexpected}",
                flush=True,
            )
        print(f"[action_head] Loaded from {action_head_path}", flush=True)

    return action_head


class PPOValueHead(nn.Module):
    def __init__(self, llm_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(llm_dim * ACTION_DIM),
            nn.Linear(llm_dim * ACTION_DIM, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, 1),
        )

    def forward(self, action_hidden_states: torch.Tensor) -> torch.Tensor:
        # action_hidden_states: (B, ACTION_DIM, D) for current action token block
        target_dtype = self.mlp[1].weight.dtype
        x = action_hidden_states.reshape(action_hidden_states.shape[0], -1).to(dtype=target_dtype)
        return self.mlp(x).squeeze(-1)


class OpenVLAPPOPolicy(nn.Module):
    def __init__(
        self,
        vla: nn.Module,
        processor: Any,
        action_head: L1RegressionActionHead,
        value_head: PPOValueHead,
        device: torch.device,
        num_images_in_input: int,
        init_log_std: float,
    ):
        super().__init__()
        self.vla = vla
        self.processor = processor
        self.action_head = action_head
        self.value_head = value_head
        self.device = device
        self.num_images_in_input = int(num_images_in_input)
        self.log_std = nn.Parameter(
            torch.full(
                (ACTION_DIM,),
                float(init_log_std),
                dtype=torch.float32,
                device=self.device,
            )
        )

    def forward(
        self,
        images_primary: List[np.ndarray],
        instructions: List[str],
        images_wrist: Optional[List[np.ndarray]] = None,
    ):
        return self.distribution_and_value(
            images_primary=images_primary,
            instructions=instructions,
            images_wrist=images_wrist,
        )

    def _core_model(self):
        # PEFT wrappers forward unknown attrs, but we keep a robust fallback here.
        if hasattr(self.vla, "_prepare_input_for_action_prediction"):
            return self.vla
        base = getattr(self.vla, "base_model", None)
        if base is not None and hasattr(base, "model"):
            return base.model
        return self.vla

    def _prepare_inputs(
        self,
        images_primary: List[np.ndarray],
        images_wrist: Optional[List[np.ndarray]],
        instructions: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompts = [f"In: What action should the robot take to {t.lower()}?\nOut:" for t in instructions]

        input_ids_list = []
        attn_list = []
        pix_list = []

        for i, prompt in enumerate(prompts):
            img_primary = Image.fromarray(images_primary[i].astype(np.uint8)).convert("RGB")
            inputs = self.processor(prompt, img_primary, return_tensors="pt")
            pixel_values = inputs["pixel_values"]

            if self.num_images_in_input > 1 and images_wrist is not None:
                img_wrist = Image.fromarray(images_wrist[i].astype(np.uint8)).convert("RGB")
                wrist_inputs = self.processor(prompt, img_wrist, return_tensors="pt")
                pixel_values = torch.cat([pixel_values, wrist_inputs["pixel_values"]], dim=1)

            input_ids_list.append(inputs["input_ids"][0])
            attn_list.append(inputs["attention_mask"][0])
            pix_list.append(pixel_values[0])

        max_len = max(x.shape[0] for x in input_ids_list)
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id

        padded_ids = []
        padded_attn = []
        for ids, attn in zip(input_ids_list, attn_list):
            pad = max_len - ids.shape[0]
            if pad > 0:
                ids = torch.cat([ids, torch.full((pad,), pad_id, dtype=ids.dtype)], dim=0)
                attn = torch.cat([attn, torch.zeros((pad,), dtype=attn.dtype)], dim=0)
            padded_ids.append(ids)
            padded_attn.append(attn)

        input_ids = torch.stack(padded_ids, dim=0).to(self.device)
        attention_mask = torch.stack(padded_attn, dim=0).to(self.device)
        pixel_values = torch.stack(pix_list, dim=0).to(self.device, dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32)

        return input_ids, attention_mask, pixel_values

    def _extract_action_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        model = self._core_model()

        labels = torch.full_like(input_ids, fill_value=-100)
        prompt_len = input_ids.shape[1]

        input_ids_prep, attn_prep = model._prepare_input_for_action_prediction(input_ids, attention_mask)
        labels = model._prepare_labels_for_action_prediction(labels, input_ids_prep)

        input_embeddings = model.get_input_embeddings()(input_ids_prep)
        all_actions_mask = model._process_action_masks(labels)

        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )

        projected_patch_embeddings = model._process_vision_features(pixel_values, language_embeddings, use_film=False)

        all_actions_mask_expanded = all_actions_mask.unsqueeze(-1)
        input_embeddings = input_embeddings * ~all_actions_mask_expanded

        multimodal_embeddings, multimodal_attention_mask = model._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attn_prep
        )

        language_model_output = model.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden_states = language_model_output.hidden_states[-1]
        p_actual = projected_patch_embeddings.shape[1]

        text_hidden_states = torch.cat(
            [last_hidden_states[:, :1, :], last_hidden_states[:, 1 + p_actual :, :]], dim=1
        )

        a = ACTION_DIM * NUM_ACTIONS_CHUNK
        action_hidden_states = text_hidden_states[:, prompt_len : prompt_len + a, :]
        return action_hidden_states

    def distribution_and_value(
        self,
        images_primary: List[np.ndarray],
        instructions: List[str],
        images_wrist: Optional[List[np.ndarray]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids, attention_mask, pixel_values = self._prepare_inputs(images_primary, images_wrist, instructions)

        action_hidden_states = self._extract_action_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

        pred_pre = self.action_head.predict_action(action_hidden_states)
        action_chunk = torch.tanh(pred_pre)
        mean_action = action_chunk[:, 0, :].to(dtype=torch.float32)

        std = (
            torch.exp(self.log_std)
            .unsqueeze(0)
            .expand_as(mean_action)
            .to(device=mean_action.device, dtype=torch.float32)
        )
        std = torch.clamp(std, min=1e-6)

        current_action_hidden = action_hidden_states[:, :ACTION_DIM, :]
        value = self.value_head(current_action_hidden)
        return mean_action, std, value


@dataclass
class Transition:
    img_primary: np.ndarray
    img_wrist: Optional[np.ndarray]
    instruction: str
    action: np.ndarray
    logprob: float
    env_reward: float
    reward: float
    done: float
    value: float


def _latest_image_from_sim(sim, fallback_hw: Tuple[int, int] = (224, 224), wrist: bool = False) -> np.ndarray:
    frames_attr = "ee_camera_frames" if wrist else "overview_frames"
    frames = getattr(sim, frames_attr, None)
    if isinstance(frames, list) and len(frames) > 0:
        img = np.asarray(frames[-1])
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    # fallback black image if frame not available yet
    h, w = fallback_hw
    return np.zeros((h, w, 3), dtype=np.uint8)


class CDPRVisionLanguageEnv:
    def __init__(
        self,
        cdpr_dataset_root: Path,
        cdpr_mujoco_root: Optional[str],
        catalog_path: Optional[str],
        max_steps: int,
        action_step_xyz: float,
        action_step_yaw: float,
        capture_frames: bool,
        instruction_types: Optional[Sequence[str]],
        desk_textures_dir: str,
        allowed_objects: Sequence[str],
        desk_geom_regex: str,
        desk_texrepeat: Sequence[int],
        wrapper_cleanup: bool,
        use_wrapper_cache: bool,
        invert_x_action: bool,
        invert_y_action: bool,
        seed: int,
    ):
        cdpr_root = _resolve_cdpr_dataset_root(cdpr_dataset_root)
        if str(cdpr_root) not in sys.path:
            sys.path.insert(0, str(cdpr_root))
        cdpr_mj_parent = _resolve_cdpr_mujoco_parent(cdpr_root, cdpr_mujoco_root)
        if cdpr_mj_parent is not None and str(cdpr_mj_parent) not in sys.path:
            sys.path.insert(0, str(cdpr_mj_parent))
            print(f"[env] Added cdpr_mujoco parent to PYTHONPATH: {cdpr_mj_parent}", flush=True)
        if cdpr_mj_parent is not None:
            _prepend_to_pythonpath(cdpr_mj_parent)
            print(f"[env] Exported PYTHONPATH for subprocesses: {cdpr_mj_parent}", flush=True)
        elif cdpr_mj_parent is None:
            print(
                "[WARN] Could not auto-locate `cdpr_mujoco/` package parent. "
                "If env creation fails, pass --cdpr_mujoco_root /path/to/parent.",
                flush=True,
            )

        import cdpr_dataset.rl_cdpr_env as rl_env_module
        CDPRLanguageRLEnv = rl_env_module.CDPRLanguageRLEnv
        _apply_cdpr_env_runtime_shims(CDPRLanguageRLEnv, rl_env_module)
        self._rl_env_module = rl_env_module

        self.env = CDPRLanguageRLEnv(
            catalog_path=catalog_path,
            max_steps=max_steps,
            action_step_xyz=action_step_xyz,
            action_step_yaw=action_step_yaw,
            capture_frames=capture_frames,
            instruction_types=instruction_types,
            desk_textures_dir=desk_textures_dir,
            allowed_objects=allowed_objects,
            desk_geom_regex=desk_geom_regex,
            desk_texrepeat=desk_texrepeat,
            wrapper_cleanup=wrapper_cleanup,
            use_wrapper_cache=use_wrapper_cache,
            seed=seed,
        )
        self._instruction = ""
        self.invert_x_action = bool(invert_x_action)
        self.invert_y_action = bool(invert_y_action)
        self._scene_wrapper_cache: Dict[str, List[Path]] = {}
        self._texture_name_by_wrapper: Dict[str, str] = {}

    @staticmethod
    def _attach_state(obs_out: Dict[str, Any], raw_obs: Any) -> None:
        if not isinstance(raw_obs, dict):
            return
        ee = raw_obs.get("ee_position")
        tgt = raw_obs.get("target_object_position")
        all_obj = raw_obs.get("all_object_positions")
        obj_mask = raw_obs.get("object_position_mask")
        if ee is not None:
            obs_out["ee_position"] = np.asarray(ee, dtype=np.float32)
        if tgt is not None:
            obs_out["target_object_position"] = np.asarray(tgt, dtype=np.float32)
        if all_obj is not None:
            obs_out["all_object_positions"] = np.asarray(all_obj, dtype=np.float32)
        if obj_mask is not None:
            obs_out["object_position_mask"] = np.asarray(obj_mask, dtype=np.float32)

    def reset(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            raw_obs, info = self.env.reset(options=options)
        except TypeError:
            # Backward compatibility with env variants that don't accept `options`.
            raw_obs, info = self.env.reset()
        self._instruction = str(info.get("language_instruction", ""))

        # Ensure at least one captured frame exists after reset.
        try:
            self.env.sim.run_simulation_step(capture_frame=True)
        except Exception:
            pass

        obs = {
            "image_primary": _latest_image_from_sim(self.env.sim, wrist=False),
            "image_wrist": _latest_image_from_sim(self.env.sim, wrist=True),
            "instruction": self._instruction,
        }
        self._attach_state(obs, raw_obs)
        return obs

    def scene_names(self) -> List[str]:
        scenes = getattr(self.env, "scenes", None)
        if scenes is None:
            return []
        out: List[str] = []
        for scene in scenes:
            name = getattr(scene, "name", None)
            if name:
                out.append(str(name))
        return out

    @staticmethod
    def _safe_cache_tag(text: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_\\-]+", "_", str(text))[:96]

    def _activate_scene_wrapper_cache(
        self,
        scene_wrapper_cache: Dict[str, List[Path]],
        texture_name_by_wrapper: Dict[str, str],
    ) -> Dict[str, int]:
        rl = self.env
        cache = {str(k): [Path(p).resolve() for p in v] for k, v in scene_wrapper_cache.items() if v}
        texture_map = {str(Path(k).resolve()): str(v) for k, v in texture_name_by_wrapper.items()}
        if not cache:
            return {"scenes": 0, "variants": 0, "textures": 0}

        cached_scene_names = set(cache.keys())
        rl.scenes = [s for s in list(getattr(rl, "scenes", []) or []) if str(getattr(s, "name", "")) in cached_scene_names]
        self._scene_wrapper_cache = cache
        self._texture_name_by_wrapper = texture_map

        def _build_wrapper_cached(this, scene):
            scene_name = str(getattr(scene, "name", ""))
            variants_local = self._scene_wrapper_cache.get(scene_name)
            if variants_local:
                idx = int(this.np_random.integers(0, len(variants_local)))
                chosen = Path(variants_local[idx]).resolve()
                this._desk_texture_name = self._texture_name_by_wrapper.get(str(chosen), "")
                return chosen
            return this._build_wrapper_original(scene)

        if not hasattr(rl, "_build_wrapper_original"):
            rl._build_wrapper_original = rl._build_wrapper
        rl._build_wrapper = types.MethodType(_build_wrapper_cached, rl)
        rl.wrapper_cleanup = False
        rl.use_wrapper_cache = True

        variants_total = sum(len(v) for v in cache.values())
        unique_textures = len({v for v in texture_map.values() if v})
        return {"scenes": len(cache), "variants": variants_total, "textures": unique_textures}

    def enable_prebuilt_scene_cache(
        self,
        scene_pool_size: int,
        texture_pool_size: int,
        seed: int,
    ) -> Dict[str, int]:
        rl = self.env
        rl_mod = self._rl_env_module

        scenes_all = list(getattr(rl, "scenes", []) or [])
        if not scenes_all:
            return {"scenes": 0, "variants": 0, "textures": 0}

        rng = np.random.default_rng(int(seed))
        if int(scene_pool_size) > 0 and len(scenes_all) > int(scene_pool_size):
            chosen_idx = np.sort(rng.choice(len(scenes_all), size=int(scene_pool_size), replace=False))
            scenes = [scenes_all[int(i)] for i in chosen_idx]
        else:
            scenes = scenes_all

        try:
            build_wrapper_if_needed = rl_mod._import_wrapper_builder()
        except Exception:
            from cdpr_dataset.generate_cdpr_dataset import build_wrapper_if_needed

        texture_files = list(getattr(rl, "desk_texture_files", []) or [])
        if int(texture_pool_size) > 0 and len(texture_files) > int(texture_pool_size):
            tex_idx = np.sort(rng.choice(len(texture_files), size=int(texture_pool_size), replace=False))
            texture_files = [texture_files[int(i)] for i in tex_idx]

        cache: Dict[str, List[Path]] = {}
        texture_name_by_wrapper: Dict[str, str] = {}
        total_variants = 0

        for s_idx, scene in enumerate(scenes):
            scene_name = str(getattr(scene, "name", ""))
            scene_objects = list(getattr(scene, "objects", []))
            if not scene_name or not scene_objects:
                continue

            base_wrapper = Path(
                build_wrapper_if_needed(
                    scene_name=scene_name,
                    object_names=scene_objects,
                    scene_z=rl.defaults.get("scene_z", -0.85),
                    ee_start=rl.defaults.get("ee_start", (0.0, 0.0, 0.25)),
                    table_z=rl.defaults.get("table_z", 0.15),
                    settle_time=rl.defaults.get("settle_time", 1.0),
                    wrapper_out=None,
                    use_cache=True,
                )
            ).resolve()

            variants: List[Path] = []
            if texture_files:
                for tex_idx_local, tex in enumerate(texture_files):
                    tex_path = Path(tex).resolve()
                    tex_hash = hashlib.sha1(tex_path.name.encode("utf-8")).hexdigest()[:8]
                    tag = self._safe_cache_tag(f"c{s_idx:02d}_t{tex_idx_local:03d}_{tex_hash}")
                    patched = rl_mod._build_textured_wrapper_variant(
                        base_wrapper_xml=base_wrapper,
                        chosen_texture=tex_path,
                        variant_tag=tag,
                        desk_geom_regex=rl.desk_geom_regex,
                        desk_texrepeat=rl.desk_texrepeat,
                    )
                    wrapper_path = Path(patched.wrapper_xml).resolve()
                    variants.append(wrapper_path)
                    texture_name_by_wrapper[str(wrapper_path)] = tex_path.name
            else:
                variants.append(base_wrapper)
                texture_name_by_wrapper[str(base_wrapper)] = ""

            if variants:
                cache[scene_name] = variants
                total_variants += len(variants)

        if not cache:
            return {"scenes": 0, "variants": 0, "textures": 0}

        out = self._activate_scene_wrapper_cache(cache, texture_name_by_wrapper)
        # Keep this as explicit prebuild count for logs.
        out["variants"] = total_variants
        return out

    def attach_prebuilt_scene_cache(
        self,
        scene_wrapper_cache: Dict[str, List[Path]],
        texture_name_by_wrapper: Dict[str, str],
    ) -> Dict[str, int]:
        return self._activate_scene_wrapper_cache(scene_wrapper_cache, texture_name_by_wrapper)

    def step(self, action: np.ndarray):
        action_env = np.asarray(action, dtype=np.float32).copy()
        if self.invert_x_action:
            action_env[0] *= -1.0
        if self.invert_y_action:
            action_env[1] *= -1.0
        raw_obs, reward, terminated, truncated, info = self.env.step(action_env)
        self._instruction = str(info.get("language_instruction", self._instruction))

        obs = {
            "image_primary": _latest_image_from_sim(self.env.sim, wrist=False),
            "image_wrist": _latest_image_from_sim(self.env.sim, wrist=True),
            "instruction": self._instruction,
        }
        self._attach_state(obs, raw_obs)
        done = bool(terminated or truncated)
        return obs, float(reward), done, info

    def close(self):
        self.env.close()


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(rewards)
    advantages = np.zeros((n,), dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        nonterminal = 1.0 - dones[t]
        next_val = next_value if t == n - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def gaussian_sample(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return mean + torch.randn_like(mean) * std


def gaussian_log_prob(action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    var = std * std
    log_scale = torch.log(std)
    return -0.5 * (((action - mean) ** 2) / var + 2.0 * log_scale + math.log(2.0 * math.pi))


def gaussian_entropy(std: torch.Tensor) -> torch.Tensor:
    return 0.5 + 0.5 * math.log(2.0 * math.pi) + torch.log(std)


def _distance_ee_to_target_from_obs(obs: Dict[str, Any]) -> Optional[float]:
    ee = obs.get("ee_position")
    tgt = obs.get("target_object_position")
    if ee is None or tgt is None:
        return None

    ee_arr = np.asarray(ee, dtype=np.float32).reshape(-1)
    tgt_arr = np.asarray(tgt, dtype=np.float32).reshape(-1)
    if ee_arr.shape[0] < 3 or tgt_arr.shape[0] < 3:
        return None
    return float(np.linalg.norm(ee_arr[:3] - tgt_arr[:3]))


def _shape_reward_with_delta_progress(
    env_reward: float,
    distance_before: Optional[float],
    distance_after: Optional[float],
    delta_closer_reward_coef: float,
    delta_farther_penalty_coef: float,
) -> Tuple[float, float, float, float]:
    if distance_before is None or distance_after is None:
        return float(env_reward), 0.0, 0.0, 0.0
    raw_delta = float(distance_before - distance_after)
    closer_bonus = float(max(raw_delta, 0.0) * max(float(delta_closer_reward_coef), 0.0))
    farther_penalty = float(max(-raw_delta, 0.0) * max(float(delta_farther_penalty_coef), 0.0))
    shaped_reward = float(env_reward + closer_bonus - farther_penalty)
    return shaped_reward, closer_bonus, farther_penalty, raw_delta


def _make_scene_reset_sampler(
    scene_names: Sequence[str],
    scene_sampling: str,
    seed: int,
):
    names = [str(x) for x in scene_names if str(x)]
    rng = np.random.default_rng(seed)
    cursor = 0

    def _next_options() -> Optional[Dict[str, Any]]:
        nonlocal cursor
        if len(names) == 0 or scene_sampling == "env_random":
            return None
        if scene_sampling == "random":
            scene = names[int(rng.integers(0, len(names)))]
        else:
            scene = names[cursor % len(names)]
            cursor += 1
        return {"scene": scene}

    return names, _next_options


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], repeats=3, axis=2)
    return arr


def _float_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value_f = float(value)
    if not math.isfinite(value_f):
        return None
    return value_f


def _vec3_or_nan(value: Any) -> np.ndarray:
    if value is None:
        return np.full((3,), np.nan, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    out = np.full((3,), np.nan, dtype=np.float32)
    n = min(3, int(arr.shape[0]))
    if n > 0:
        out[:n] = arr[:n]
    return out


def _stack_or_object(arrays: List[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.asarray([], dtype=np.float32)
    shapes = {tuple(a.shape) for a in arrays}
    if len(shapes) == 1:
        return np.stack(arrays, axis=0)
    return np.asarray(arrays, dtype=object)


def save_rollout_tap_npz(
    run_dir: Path,
    update: int,
    records: List[Dict[str, Any]],
) -> Optional[Path]:
    if not records:
        return None

    out_dir = run_dir / "rollout_tap"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"update_{int(update):05d}.npz"

    all_obj_before = [
        np.asarray(r["all_object_positions_before"], dtype=np.float32)
        if r["all_object_positions_before"] is not None
        else np.full((0, 3), np.nan, dtype=np.float32)
        for r in records
    ]
    all_obj_after = [
        np.asarray(r["all_object_positions_after"], dtype=np.float32)
        if r["all_object_positions_after"] is not None
        else np.full((0, 3), np.nan, dtype=np.float32)
        for r in records
    ]
    all_obj_mask_before = [
        np.asarray(r["object_position_mask_before"], dtype=np.float32)
        if r["object_position_mask_before"] is not None
        else np.full((0,), np.nan, dtype=np.float32)
        for r in records
    ]
    all_obj_mask_after = [
        np.asarray(r["object_position_mask_after"], dtype=np.float32)
        if r["object_position_mask_after"] is not None
        else np.full((0,), np.nan, dtype=np.float32)
        for r in records
    ]

    data = {
        "update": np.asarray([int(update)] * len(records), dtype=np.int32),
        "global_step": np.asarray([int(r["global_step"]) for r in records], dtype=np.int64),
        "step_in_update": np.asarray([int(r["step_in_update"]) for r in records], dtype=np.int32),
        "action_delta": np.asarray([r["action_delta"] for r in records], dtype=np.float32),
        "action_delta_norm": np.asarray([float(r["action_delta_norm"]) for r in records], dtype=np.float32),
        "reward_env": np.asarray([float(r["reward_env"]) for r in records], dtype=np.float32),
        "reward_shaped": np.asarray([float(r["reward_shaped"]) for r in records], dtype=np.float32),
        "closer_bonus": np.asarray([float(r["closer_bonus"]) for r in records], dtype=np.float32),
        "farther_penalty": np.asarray([float(r["farther_penalty"]) for r in records], dtype=np.float32),
        "distance_before": np.asarray(
            [np.nan if r["distance_before"] is None else float(r["distance_before"]) for r in records],
            dtype=np.float32,
        ),
        "distance_after": np.asarray(
            [np.nan if r["distance_after"] is None else float(r["distance_after"]) for r in records],
            dtype=np.float32,
        ),
        "distance_delta_raw": np.asarray([float(r["distance_delta_raw"]) for r in records], dtype=np.float32),
        "ee_position_before": np.asarray([_vec3_or_nan(r["ee_position_before"]) for r in records], dtype=np.float32),
        "ee_position_after": np.asarray([_vec3_or_nan(r["ee_position_after"]) for r in records], dtype=np.float32),
        "target_position_before": np.asarray([_vec3_or_nan(r["target_position_before"]) for r in records], dtype=np.float32),
        "target_position_after": np.asarray([_vec3_or_nan(r["target_position_after"]) for r in records], dtype=np.float32),
        "all_object_positions_before": _stack_or_object(all_obj_before),
        "all_object_positions_after": _stack_or_object(all_obj_after),
        "object_position_mask_before": _stack_or_object(all_obj_mask_before),
        "object_position_mask_after": _stack_or_object(all_obj_mask_after),
        "value_pred": np.asarray([float(r["value_pred"]) for r in records], dtype=np.float32),
        "logprob": np.asarray([float(r["logprob"]) for r in records], dtype=np.float32),
        "scene": np.asarray([str(r["scene"]) for r in records], dtype=object),
        "instruction": np.asarray([str(r["instruction"]) for r in records], dtype=object),
        "instruction_type": np.asarray([str(r["instruction_type"]) for r in records], dtype=object),
        "target_object_catalog": np.asarray([str(r["target_object_catalog"]) for r in records], dtype=object),
        "env_done": np.asarray([bool(r["env_done"]) for r in records], dtype=bool),
        "forced_scene_refresh": np.asarray([bool(r["forced_scene_refresh"]) for r in records], dtype=bool),
    }

    np.savez_compressed(out_path, **data)
    return out_path


def run_validation_rollouts(
    *,
    policy_core: OpenVLAPPOPolicy,
    val_env: CDPRVisionLanguageEnv,
    run_dir: Path,
    update: int,
    num_episodes: int,
    max_steps: int,
    num_images_in_input: int,
    delta_closer_reward_coef: float,
    delta_farther_penalty_coef: float,
    save_frames: bool,
    quiet_env_logs: bool,
    next_reset_options,
) -> Dict[str, Any]:
    val_root = run_dir / "validation" / f"update_{update:05d}"
    val_root.mkdir(parents=True, exist_ok=True)

    episode_env_returns: List[float] = []
    episode_shaped_returns: List[float] = []
    episode_successes: List[float] = []

    for ep_idx in range(1, int(num_episodes) + 1):
        ep_dir = val_root / f"episode_{ep_idx:02d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        with _silence_stdio(bool(quiet_env_logs)):
            obs = val_env.reset(options=next_reset_options())

        env_return = 0.0
        shaped_return = 0.0
        success = False
        steps: List[Dict[str, Any]] = []

        for step_idx in range(int(max_steps)):
            if save_frames:
                Image.fromarray(_to_uint8_rgb(obs["image_primary"])).save(ep_dir / f"overview_{step_idx:03d}.png")
                wrist = obs.get("image_wrist")
                if wrist is not None:
                    Image.fromarray(_to_uint8_rgb(wrist)).save(ep_dir / f"wrist_{step_idx:03d}.png")

            with torch.no_grad():
                mean_action, std_action, value = policy_core(
                    images_primary=[obs["image_primary"]],
                    images_wrist=[obs["image_wrist"]] if num_images_in_input > 1 else None,
                    instructions=[obs["instruction"]],
                )
            action_np = torch.clamp(mean_action, -1.0, 1.0)[0].cpu().numpy().astype(np.float32)

            dist_before = _distance_ee_to_target_from_obs(obs)
            with _silence_stdio(bool(quiet_env_logs)):
                next_obs, env_reward, done, info = val_env.step(action_np)
            dist_after = _distance_ee_to_target_from_obs(next_obs)

            shaped_reward, closer_bonus, farther_penalty, raw_delta = _shape_reward_with_delta_progress(
                env_reward=env_reward,
                distance_before=dist_before,
                distance_after=dist_after,
                delta_closer_reward_coef=delta_closer_reward_coef,
                delta_farther_penalty_coef=delta_farther_penalty_coef,
            )

            env_return += float(env_reward)
            shaped_return += float(shaped_reward)
            success = bool(info.get("success", success))

            steps.append(
                {
                    "step": int(step_idx),
                    "instruction": str(obs.get("instruction", "")),
                    "scene": str(info.get("scene", "")),
                    "target_object_catalog": str(info.get("target_object_catalog", "")),
                    "action_delta": action_np.tolist(),
                    "action_delta_norm": float(np.linalg.norm(action_np)),
                    "value_pred": float(value.item()),
                    "std_mean": float(std_action.mean().item()),
                    "reward_env": float(env_reward),
                    "reward_shaped": float(shaped_reward),
                    "closer_bonus": float(closer_bonus),
                    "farther_penalty": float(farther_penalty),
                    "distance_delta_raw": float(raw_delta),
                    "distance_before": _float_or_none(dist_before),
                    "distance_after": _float_or_none(dist_after),
                    "success": bool(info.get("success", False)),
                    "done": bool(done),
                }
            )

            obs = next_obs
            if done:
                break

        steps_path = ep_dir / "steps.jsonl"
        with steps_path.open("w", encoding="utf-8") as f_steps:
            for row in steps:
                f_steps.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary_path = ep_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f_ep:
            json.dump(
                {
                    "update": int(update),
                    "episode": int(ep_idx),
                    "env_return": float(env_return),
                    "shaped_return": float(shaped_return),
                    "success": bool(success),
                    "num_steps": int(len(steps)),
                },
                f_ep,
                indent=2,
                sort_keys=True,
            )

        episode_env_returns.append(float(env_return))
        episode_shaped_returns.append(float(shaped_return))
        episode_successes.append(1.0 if success else 0.0)

    out = {
        "update": int(update),
        "episodes": int(num_episodes),
        "mean_env_return": float(np.mean(episode_env_returns)) if episode_env_returns else 0.0,
        "mean_shaped_return": float(np.mean(episode_shaped_returns)) if episode_shaped_returns else 0.0,
        "success_rate": float(np.mean(episode_successes)) if episode_successes else 0.0,
        "path": str(val_root),
    }
    with (val_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    return out


def save_checkpoint(
    run_dir: Path,
    step: int,
    vla: nn.Module,
    action_head: nn.Module,
    value_head: nn.Module,
    log_std: nn.Parameter,
) -> None:
    ckpt_dir = run_dir / f"step_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapters if present.
    if hasattr(vla, "save_pretrained"):
        adapter_dir = ckpt_dir / "vla_cdpr_adapter"
        vla.save_pretrained(adapter_dir)

    torch.save(action_head.state_dict(), ckpt_dir / "action_head_cdpr.pt")
    torch.save(value_head.state_dict(), ckpt_dir / "value_head.pt")
    torch.save({"log_std": log_std.detach().cpu()}, ckpt_dir / "ppo_actor_stats.pt")


def main() -> None:
    args = parse_args()
    if not args.capture_frames:
        raise ValueError(
            "Vision-language PPO requires rendered images; please run with --capture_frames (default: enabled)."
        )
    if args.minibatch_size < 1:
        raise ValueError("--minibatch_size must be >= 1.")
    if args.microbatch_size < 1:
        raise ValueError("--microbatch_size must be >= 1.")
    if args.validation_episodes < 1:
        raise ValueError("--validation_episodes must be >= 1.")
    if args.validation_max_steps < 1:
        raise ValueError("--validation_max_steps must be >= 1.")
    if args.scene_refresh_every_steps == 0:
        args.scene_refresh_every_steps = -1
    if args.validate_every_updates == 0:
        args.validate_every_updates = -1
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

    rank, local_rank, world_size = _init_distributed()
    is_main = _is_main_process(rank)

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
            "Use `torchrun --nproc_per_node=<num_gpus> ... ppo_finetune_cdpr.py` to train with DDP on all GPUs.",
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
                "[WARN] Multi-GPU PPO may skip some trainable params on some iterations. "
                "Forcing --ddp_find_unused_parameters for DDP stability.",
                flush=True,
            )
        args.ddp_find_unused_parameters = True

    # NOTE: Some PyTorch versions hit reducer/internal asserts when combining
    # DDP + gradient checkpointing + multiple backward() calls per optimizer step.
    # Keep one backward per step in multi-GPU mode for robustness.
    if world_size > 1 and args.microbatch_size != args.minibatch_size:
        if is_main:
            print(
                "[WARN] Multi-GPU mode uses one backward per optimizer step for stability on current PyTorch; "
                "forcing --microbatch_size == --minibatch_size. "
                "To reduce memory, lower --minibatch_size (e.g., 2 or 4).",
                flush=True,
            )
        args.microbatch_size = args.minibatch_size

    set_seed(args.seed + rank)

    run_dir_local: Optional[Path] = None
    if is_main:
        run_dir_local = make_run_dir(args)
        save_run_config(args, run_dir_local)
        print(f"Run dir: {run_dir_local}", flush=True)
    run_dir = Path(_broadcast_object(str(run_dir_local) if run_dir_local is not None else None, rank))

    cdpr_root = _resolve_cdpr_dataset_root(Path(args.cdpr_dataset_root))
    _prune_generated_wrapper_artifacts(
        cdpr_dataset_root=cdpr_root,
        is_main=is_main,
        rank=rank,
    )

    args.desk_textures_dir = _prepare_desk_textures_dir(
        src_dir=args.desk_textures_dir,
        run_dir=run_dir,
        is_main=is_main,
        rank=rank,
        max_textures=args.texture_pool_size if args.texture_pool_size > 0 else 128,
    )

    vla, processor = load_vla_and_processor(args, device)
    maybe_enable_gradient_checkpointing(vla, enabled=bool(args.gradient_checkpointing))

    llm_dim = _resolve_llm_dim(vla)
    if llm_dim is None:
        raise RuntimeError("Could not resolve llm_dim from OpenVLA model wrapper.")

    action_head = build_action_head(args=args, llm_dim=llm_dim, device=device)
    value_head = PPOValueHead(llm_dim=llm_dim).to(device)

    policy: nn.Module = OpenVLAPPOPolicy(
        vla=vla,
        processor=processor,
        action_head=action_head,
        value_head=value_head,
        device=device,
        num_images_in_input=args.num_images_in_input,
        init_log_std=args.init_log_std,
    )
    if world_size > 1:
        if device.type != "cuda":
            raise RuntimeError("DDP multi-process PPO currently requires CUDA devices.")
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

    policy_core = _unwrap_module(policy)
    lora_params = [p for p in policy_core.vla.parameters() if p.requires_grad]
    action_params = list(policy_core.action_head.parameters())
    value_params = list(policy_core.value_head.parameters())
    trainable_params = lora_params + action_params + value_params + [policy_core.log_std]

    if is_main:
        print(
            "[trainable] "
            f"lora={sum(p.numel() for p in lora_params)} "
            f"action_head={sum(p.numel() for p in action_params)} "
            f"value_head={sum(p.numel() for p in value_params)} "
            f"world_size={world_size}",
            flush=True,
        )

    param_groups = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": args.learning_rate})
    param_groups.append({"params": action_params, "lr": args.learning_rate})
    param_groups.append({"params": value_params, "lr": args.value_lr})
    param_groups.append({"params": [policy_core.log_std], "lr": args.value_lr})
    optimizer = AdamW(param_groups)
    use_tqdm = bool(args.status_bar and is_main and (tqdm is not None))
    if is_main and args.status_bar and tqdm is None:
        print("[WARN] tqdm is unavailable; falling back to periodic summary prints.", flush=True)

    env: Optional[CDPRVisionLanguageEnv] = None
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
        with _silence_stdio(bool(args.quiet_env_logs)):
            env = CDPRVisionLanguageEnv(
                cdpr_dataset_root=Path(args.cdpr_dataset_root),
                cdpr_mujoco_root=args.cdpr_mujoco_root,
                catalog_path=args.catalog_path,
                max_steps=args.max_env_steps,
                action_step_xyz=args.action_step_xyz,
                action_step_yaw=args.action_step_yaw,
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
                seed=args.seed + rank,
            )

        if args.prebuild_scene_cache:
            cache_info = env.enable_prebuilt_scene_cache(
                scene_pool_size=args.scene_pool_size,
                texture_pool_size=args.texture_pool_size,
                seed=args.seed + 20_000 + rank,
            )
            if is_main:
                print(
                    f"[env_cache] train scenes={cache_info['scenes']} "
                    f"variants={cache_info['variants']} textures={cache_info['textures']}",
                    flush=True,
                )

        scene_names, next_reset_options = _make_scene_reset_sampler(
            scene_names=env.scene_names(),
            scene_sampling=args.scene_sampling,
            seed=args.seed + 10_000 + rank,
        )

        if is_main:
            print(
                f"[env] scene_sampling={args.scene_sampling} "
                f"scene_refresh_every_steps={args.scene_refresh_every_steps} "
                f"catalog_scenes={len(scene_names)} "
                f"prebuild_scene_cache={bool(args.prebuild_scene_cache)} "
                f"scene_pool_size={args.scene_pool_size} "
                f"texture_pool_size={args.texture_pool_size} "
                f"delta_closer_reward_coef={args.delta_closer_reward_coef} "
                f"delta_farther_penalty_coef={args.delta_farther_penalty_coef} "
                f"validate_every_updates={args.validate_every_updates} "
                f"rollout_tap_every_updates={args.rollout_tap_every_updates}",
                flush=True,
            )
            if len(scene_names) < 2:
                print(
                    "[WARN] Catalog has fewer than 2 scenes. "
                    "For stronger visual diversity, add more scenes to catalog YAML.",
                    flush=True,
                )

        with _silence_stdio(bool(args.quiet_env_logs)):
            obs = env.reset(options=next_reset_options())

        next_val_reset_options = None
        if is_main and args.validate_every_updates > 0:
            with _silence_stdio(bool(args.quiet_env_logs)):
                val_env = CDPRVisionLanguageEnv(
                    cdpr_dataset_root=Path(args.cdpr_dataset_root),
                    cdpr_mujoco_root=args.cdpr_mujoco_root,
                    catalog_path=args.catalog_path,
                    max_steps=args.max_env_steps,
                    action_step_xyz=args.action_step_xyz,
                    action_step_yaw=args.action_step_yaw,
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
                    scene_wrapper_cache=env._scene_wrapper_cache,
                    texture_name_by_wrapper=env._texture_name_by_wrapper,
                )
                print(
                    f"[env_cache] val scenes={val_cache_info['scenes']} "
                    f"variants={val_cache_info['variants']} textures={val_cache_info['textures']}",
                    flush=True,
                )
            _, next_val_reset_options = _make_scene_reset_sampler(
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
            episode_returns = []
            episode_returns_env = []
            ep_ret = 0.0
            ep_ret_env = 0.0

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
            for _ in range(args.rollout_steps):
                with torch.no_grad():
                    mean_action, std_action, value = policy(
                        images_primary=[obs["image_primary"]],
                        images_wrist=[obs["image_wrist"]] if args.num_images_in_input > 1 else None,
                        instructions=[obs["instruction"]],
                    )
                    action_t = gaussian_sample(mean_action, std_action)
                    action_t = torch.clamp(action_t, -1.0, 1.0)
                    logprob_t = gaussian_log_prob(action_t, mean_action, std_action).sum(dim=-1)

                action_np = action_t[0].cpu().numpy().astype(np.float32)
                dist_before = _distance_ee_to_target_from_obs(obs)
                next_obs, env_reward, env_done, step_info = env.step(action_np)
                dist_after = _distance_ee_to_target_from_obs(next_obs)
                reward, closer_bonus, farther_penalty, raw_dist_delta = _shape_reward_with_delta_progress(
                    env_reward=env_reward,
                    distance_before=dist_before,
                    distance_after=dist_after,
                    delta_closer_reward_coef=args.delta_closer_reward_coef,
                    delta_farther_penalty_coef=args.delta_farther_penalty_coef,
                )
                global_step += 1
                forced_scene_refresh = (
                    args.scene_refresh_every_steps > 0
                    and (global_step % args.scene_refresh_every_steps == 0)
                )
                done = bool(env_done or forced_scene_refresh)

                transitions.append(
                    Transition(
                        img_primary=obs["image_primary"],
                        img_wrist=obs["image_wrist"] if args.num_images_in_input > 1 else None,
                        instruction=obs["instruction"],
                        action=action_np,
                        logprob=float(logprob_t.item()),
                        env_reward=float(env_reward),
                        reward=float(reward),
                        done=float(done),
                        value=float(value.item()),
                    )
                )

                rollout_records.append(
                    {
                        "global_step": int(global_step),
                        "step_in_update": int(len(rollout_records)),
                        "action_delta": action_np.astype(np.float32),
                        "action_delta_norm": float(np.linalg.norm(action_np)),
                        "reward_env": float(env_reward),
                        "reward_shaped": float(reward),
                        "closer_bonus": float(closer_bonus),
                        "farther_penalty": float(farther_penalty),
                        "distance_before": _float_or_none(dist_before),
                        "distance_after": _float_or_none(dist_after),
                        "distance_delta_raw": float(raw_dist_delta),
                        "ee_position_before": obs.get("ee_position"),
                        "ee_position_after": next_obs.get("ee_position"),
                        "target_position_before": obs.get("target_object_position"),
                        "target_position_after": next_obs.get("target_object_position"),
                        "all_object_positions_before": obs.get("all_object_positions"),
                        "all_object_positions_after": next_obs.get("all_object_positions"),
                        "object_position_mask_before": obs.get("object_position_mask"),
                        "object_position_mask_after": next_obs.get("object_position_mask"),
                        "value_pred": float(value.item()),
                        "logprob": float(logprob_t.item()),
                        "scene": str(step_info.get("scene", "")),
                        "instruction": str(obs.get("instruction", "")),
                        "instruction_type": str(step_info.get("instruction_type", "")),
                        "target_object_catalog": str(step_info.get("target_object_catalog", "")),
                        "env_done": bool(env_done),
                        "forced_scene_refresh": bool(forced_scene_refresh),
                    }
                )

                ep_ret += float(reward)
                ep_ret_env += float(env_reward)
                obs = next_obs

                if done:
                    episode_returns.append(ep_ret)
                    episode_returns_env.append(ep_ret_env)
                    ep_ret = 0.0
                    ep_ret_env = 0.0
                    episode_idx += 1

                    if trace_fp is not None:
                        event = {
                            "update": int(update),
                            "episode": int(episode_idx),
                            "global_step": int(global_step),
                            "scene": str(step_info.get("scene", "")),
                            "target_object_catalog": str(step_info.get("target_object_catalog", "")),
                            "target_object_body": str(step_info.get("target_object_body", "")),
                            "instruction": str(step_info.get("language_instruction", "")),
                            "instruction_type": str(step_info.get("instruction_type", "")),
                            "success": bool(step_info.get("success", False)),
                            "env_done": bool(env_done),
                            "forced_scene_refresh": bool(forced_scene_refresh),
                            "reward_env": float(env_reward),
                            "reward_shaped": float(reward),
                            "closer_bonus": float(closer_bonus),
                            "farther_penalty": float(farther_penalty),
                            "distance_delta_raw": float(raw_dist_delta),
                            "distance_before": _float_or_none(dist_before),
                            "distance_after": _float_or_none(dist_after),
                            "desk_texture": str(step_info.get("desk_texture", "")),
                        }
                        trace_fp.write(json.dumps(event, ensure_ascii=False) + "\n")
                        trace_fp.flush()

                    with _silence_stdio(bool(args.quiet_env_logs)):
                        obs = env.reset(options=next_reset_options())

                if rollout_pbar is not None:
                    rollout_pbar.update(1)

            if rollout_pbar is not None:
                rollout_pbar.close()

            with torch.no_grad():
                _, _, next_value_t = policy(
                    images_primary=[obs["image_primary"]],
                    images_wrist=[obs["image_wrist"]] if args.num_images_in_input > 1 else None,
                    instructions=[obs["instruction"]],
                )
                next_value = float(next_value_t.item())

            rewards = np.array([t.reward for t in transitions], dtype=np.float32)
            env_rewards = np.array([t.env_reward for t in transitions], dtype=np.float32)
            dones = np.array([t.done for t in transitions], dtype=np.float32)
            values = np.array([t.value for t in transitions], dtype=np.float32)
            actions = np.stack([t.action for t in transitions], axis=0)
            old_logprobs = np.array([t.logprob for t in transitions], dtype=np.float32)

            advantages, returns = compute_gae(
                rewards=rewards,
                dones=dones,
                values=values,
                next_value=next_value,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
            for _ in range(args.ppo_epochs):
                np.random.shuffle(idxs)
                for start in range(0, len(idxs), args.minibatch_size):
                    mb_idx = idxs[start : start + args.minibatch_size]
                    if len(mb_idx) == 0:
                        continue

                    optimizer.zero_grad(set_to_none=True)
                    micro_splits = max(1, math.ceil(len(mb_idx) / args.microbatch_size))

                    for micro_start in range(0, len(mb_idx), args.microbatch_size):
                        micro_idx = mb_idx[micro_start : micro_start + args.microbatch_size]
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

                            mean_action, std_action, value_pred = policy(
                                images_primary=mb_imgs_primary,
                                images_wrist=mb_imgs_wrist,
                                instructions=mb_instr,
                            )

                            mb_actions = torch.tensor(actions[micro_idx], dtype=torch.float32, device=device)
                            mb_old_logprobs = torch.tensor(old_logprobs[micro_idx], dtype=torch.float32, device=device)
                            mb_adv = torch.tensor(advantages[micro_idx], dtype=torch.float32, device=device)
                            mb_ret = torch.tensor(returns[micro_idx], dtype=torch.float32, device=device)
                            mb_old_values = torch.tensor(values[micro_idx], dtype=torch.float32, device=device)

                            new_logprob = gaussian_log_prob(mb_actions, mean_action, std_action).sum(dim=-1)
                            entropy = gaussian_entropy(std_action).sum(dim=-1).mean()
                            ratio = (new_logprob - mb_old_logprobs).exp()

                            pg_loss1 = -mb_adv * ratio
                            pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                            policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                            value_pred_clipped = mb_old_values + torch.clamp(
                                value_pred - mb_old_values,
                                -args.clip_coef,
                                args.clip_coef,
                            )
                            v_loss_unclipped = (value_pred - mb_ret) ** 2
                            v_loss_clipped = (value_pred_clipped - mb_ret) ** 2
                            value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy
                            (loss / micro_splits).backward()

                    nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    optimizer.step()
                    if train_pbar is not None:
                        train_pbar.update(1)

            if train_pbar is not None:
                train_pbar.close()

            avg_rollout_reward = float(rewards.mean())
            avg_rollout_reward_env = float(env_rewards.mean())
            avg_return = float(np.mean(episode_returns)) if episode_returns else ep_ret
            avg_return_env = float(np.mean(episode_returns_env)) if episode_returns_env else ep_ret_env

            if updates_pbar is not None:
                updates_pbar.update(1)
                updates_pbar.set_postfix(
                    step=int(global_step),
                    r_env=f"{avg_rollout_reward_env:.3f}",
                    r_shape=f"{avg_rollout_reward:.3f}",
                    ep_env=f"{avg_return_env:.3f}",
                    ep_shape=f"{avg_return:.3f}",
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
                    f"log_std_mean={float(policy_core.log_std.mean().item()):.4f}",
                    flush=True,
                )

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
                    delta_closer_reward_coef=args.delta_closer_reward_coef,
                    delta_farther_penalty_coef=args.delta_farther_penalty_coef,
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

            if is_main and update % args.save_every == 0:
                save_checkpoint(
                    run_dir=run_dir,
                    step=global_step,
                    vla=policy_core.vla,
                    action_head=policy_core.action_head,
                    value_head=policy_core.value_head,
                    log_std=policy_core.log_std,
                )

        if is_main:
            save_checkpoint(
                run_dir=run_dir,
                step=global_step,
                vla=policy_core.vla,
                action_head=policy_core.action_head,
                value_head=policy_core.value_head,
                log_std=policy_core.log_std,
            )
    finally:
        if updates_pbar is not None:
            updates_pbar.close()
        if trace_fp is not None:
            trace_fp.close()
        if env is not None:
            with _silence_stdio(bool(args.quiet_env_logs)):
                env.close()
        if val_env is not None:
            with _silence_stdio(bool(args.quiet_env_logs)):
                val_env.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
