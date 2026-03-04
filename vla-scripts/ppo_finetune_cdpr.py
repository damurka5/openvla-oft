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
import math
import os
import random
import sys
import time
import json
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.distributions import Normal
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
        default=0.01,
        help="World-frame XYZ delta scale per action step (teleop uses 0.01).",
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
        default=["ycb_apple", "ycb_pear", "ycb_peach"],
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
    ap.add_argument("--total_updates", type=int, default=3000)
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
            use_cache=None,
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
    ):
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
        dist = Normal(mean_action, std)

        current_action_hidden = action_hidden_states[:, :ACTION_DIM, :]
        value = self.value_head(current_action_hidden)
        return dist, value


@dataclass
class Transition:
    img_primary: np.ndarray
    img_wrist: Optional[np.ndarray]
    instruction: str
    action: np.ndarray
    logprob: float
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

    def reset(self) -> Dict[str, Any]:
        _, info = self.env.reset()
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
        return obs

    def step(self, action: np.ndarray):
        action_env = np.asarray(action, dtype=np.float32).copy()
        if self.invert_x_action:
            action_env[0] *= -1.0
        if self.invert_y_action:
            action_env[1] *= -1.0
        _, reward, terminated, truncated, info = self.env.step(action_env)
        self._instruction = str(info.get("language_instruction", self._instruction))

        obs = {
            "image_primary": _latest_image_from_sim(self.env.sim, wrist=False),
            "image_wrist": _latest_image_from_sim(self.env.sim, wrist=True),
            "instruction": self._instruction,
        }
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

    global_step = 0
    try:
        obs = env.reset()

        for update in range(1, args.total_updates + 1):
            policy.eval()
            transitions: List[Transition] = []
            episode_returns = []
            ep_ret = 0.0

            for _ in range(args.rollout_steps):
                with torch.no_grad():
                    policy_dist, value = policy(
                        images_primary=[obs["image_primary"]],
                        images_wrist=[obs["image_wrist"]] if args.num_images_in_input > 1 else None,
                        instructions=[obs["instruction"]],
                    )
                    action_t = policy_dist.sample()
                    action_t = torch.clamp(action_t, -1.0, 1.0)
                    logprob_t = policy_dist.log_prob(action_t).sum(dim=-1)

                action_np = action_t[0].cpu().numpy().astype(np.float32)
                next_obs, reward, done, _ = env.step(action_np)

                transitions.append(
                    Transition(
                        img_primary=obs["image_primary"],
                        img_wrist=obs["image_wrist"] if args.num_images_in_input > 1 else None,
                        instruction=obs["instruction"],
                        action=action_np,
                        logprob=float(logprob_t.item()),
                        reward=float(reward),
                        done=float(done),
                        value=float(value.item()),
                    )
                )

                ep_ret += float(reward)
                global_step += 1
                obs = next_obs

                if done:
                    episode_returns.append(ep_ret)
                    ep_ret = 0.0
                    obs = env.reset()

            with torch.no_grad():
                _, next_value_t = policy(
                    images_primary=[obs["image_primary"]],
                    images_wrist=[obs["image_wrist"]] if args.num_images_in_input > 1 else None,
                    instructions=[obs["instruction"]],
                )
                next_value = float(next_value_t.item())

            rewards = np.array([t.reward for t in transitions], dtype=np.float32)
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

                            policy_dist, value_pred = policy(
                                images_primary=mb_imgs_primary,
                                images_wrist=mb_imgs_wrist,
                                instructions=mb_instr,
                            )

                            mb_actions = torch.tensor(actions[micro_idx], dtype=torch.float32, device=device)
                            mb_old_logprobs = torch.tensor(old_logprobs[micro_idx], dtype=torch.float32, device=device)
                            mb_adv = torch.tensor(advantages[micro_idx], dtype=torch.float32, device=device)
                            mb_ret = torch.tensor(returns[micro_idx], dtype=torch.float32, device=device)
                            mb_old_values = torch.tensor(values[micro_idx], dtype=torch.float32, device=device)

                            new_logprob = policy_dist.log_prob(mb_actions).sum(dim=-1)
                            entropy = policy_dist.entropy().sum(dim=-1).mean()
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

            avg_rollout_reward = float(rewards.mean())
            avg_return = float(np.mean(episode_returns)) if episode_returns else ep_ret

            if is_main:
                print(
                    f"[update {update:05d}] "
                    f"global_step={global_step} "
                    f"rollout_reward_mean={avg_rollout_reward:.4f} "
                    f"episode_return_mean={avg_return:.4f} "
                    f"log_std_mean={float(policy_core.log_std.mean().item()):.4f}",
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
        env.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
