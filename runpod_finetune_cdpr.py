#!/usr/bin/env python3
import os
import sys
import glob
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

def find_tfrecord_dir(dataset_root: str) -> str:
    """
    Walk dataset_root and find a folder that contains .tfrecord files.
    Prefer one that has 'tfrecords_human_control' in its path, if available.
    """
    print(f"[RUNPOD] Scanning dataset root for .tfrecord files: {dataset_root}", flush=True)
    candidates = []
    for cur_root, _, files in os.walk(dataset_root):
        if any(f.endswith(".tfrecord") for f in files):
            candidates.append(cur_root)

    if not candidates:
        print("[RUNPOD] No .tfrecord files found. Root dir contents:", flush=True)
        try:
            print(os.listdir(dataset_root), flush=True)
        except Exception as e:
            print(f"[RUNPOD] Could not list root: {e}", flush=True)
        raise FileNotFoundError(f"No .tfrecord files found under: {dataset_root}")

    # Prefer "tfrecords_human_control"
    for c in candidates:
        if "tfrecords_human_control" in c:
            print(f"[RUNPOD] Selected TFRecord directory (human_control preferred): {c}", flush=True)
            return c

    # Prefer libero-like path if present
    for c in candidates:
        if "libero_spatial_no_noops" in c:
            print(f"[RUNPOD] Selected TFRecord directory (preferred match): {c}", flush=True)
            return c

    print(f"[RUNPOD] Selected TFRecord directory (first found): {candidates[0]}", flush=True)
    return candidates[0]


def safe_symlink(src: str, dest: str) -> None:
    dest_p = Path(dest)
    dest_p.parent.mkdir(parents=True, exist_ok=True)

    if dest_p.is_symlink() or dest_p.exists():
        if dest_p.is_dir() and not dest_p.is_symlink():
            shutil.rmtree(dest_p)
        else:
            dest_p.unlink()
    os.symlink(src, dest)
    print(f"[RUNPOD] Linked {dest} -> {os.path.realpath(dest)}", flush=True)


def start_tensorboard(logdir: str, port: int = 6006) -> subprocess.Popen:
    """
    Starts TensorBoard bound to 0.0.0.0 so it can be port-forwarded / exposed by Runpod.
    """
    cmd = ["tensorboard", f"--logdir={logdir}", "--host=0.0.0.0", f"--port={port}"]
    print("[RUNPOD] Starting TensorBoard:", " ".join(cmd), flush=True)
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def main() -> int:
    # ---------- User-configurable via env ----------
    # Absolute or relative path where your dataset lives INSIDE the pod.
    # Example: /workspace/data/cdpr_human_control_only
    data_root = os.environ.get("DATA_ROOT_DIR", "").strip()
    if not data_root:
        print("ERROR: Set DATA_ROOT_DIR to your dataset folder inside the pod.", flush=True)
        print("Example: export DATA_ROOT_DIR=/workspace/data/cdpr_human_control_only", flush=True)
        return 2

    # Repo root inside the pod (directory that contains vla-scripts/finetune.py)
    repo_root = Path(os.environ.get("REPO_ROOT", ".")).resolve()

    # Where you want run outputs (tensorboard + checkpoints)
    run_root_dir = os.environ.get("RUN_ROOT_DIR", str(repo_root / "VLA_CDPR" / "oft_cdpr_ckpts"))

    # What base model to start from
    vla_path = os.environ.get("VLA_PATH", "moojink/openvla-7b-oft-finetuned-libero-spatial")

    # Dataset name expected by your RLDS loader config
    dataset_name = os.environ.get("DATASET_NAME", "cdpr_local")

    # TensorBoard
    enable_tb = os.environ.get("ENABLE_TENSORBOARD", "1") == "1"
    tb_port = int(os.environ.get("TENSORBOARD_PORT", "6006"))

    # Training knobs
    max_steps = os.environ.get("MAX_STEPS", "100")
    batch_size = os.environ.get("BATCH_SIZE", "1")
    lr = os.environ.get("LEARNING_RATE", "1e-4")
    num_images = os.environ.get("NUM_IMAGES_IN_INPUT", "2")
    use_proprio = os.environ.get("USE_PROPRIO", "True")
    image_aug = os.environ.get("IMAGE_AUG", "False")

    # ---------- Environment ----------
    os.environ["VLA_ROBOT"] = "CDPR"

    # Hard-disable WandB (your finetune.py already uses DummyLogger logic, but keep these too)
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_API_KEY"] = "dummy_key_to_prevent_prompt"
    os.environ["WANDB_SILENT"] = "true"

    print(f"[RUNPOD] repo_root={repo_root}", flush=True)
    print(f"[RUNPOD] data_root={data_root}", flush=True)
    print(f"[RUNPOD] run_root_dir={run_root_dir}", flush=True)

    # ---------- Optional: recreate the same expected dataset layout via symlinks ----------
    # These destinations are what your CDPR config / code expects (from your ClearML script).
    meta_src = str(Path(data_root) / "meta_dataset.json")
    stats_src = str(Path(data_root) / "action_stats_libero_spatial_no_noops.json")

    meta_dest  = "/root/repo/CDPR_Dataset/cdpr_dataset/datasets/cdpr_synth/meta_dataset.json"
    stats_dest = "/root/repo/CDPR_Dataset/cdpr_dataset/datasets/cdpr_synth/action_stats_libero_spatial_no_noops.json"

    # Only link if the source exists; otherwise just warn (maybe you store them elsewhere now).
    if Path(meta_src).exists():
        safe_symlink(meta_src, meta_dest)
    else:
        print(f"[RUNPOD] WARN: meta_dataset.json not found at {meta_src}", flush=True)

    if Path(stats_src).exists():
        safe_symlink(stats_src, stats_dest)
    else:
        print(f"[RUNPOD] WARN: action_stats...json not found at {stats_src}", flush=True)

    # TFRecords symlink (so dataset config keeps working)
    tfrecord_src = find_tfrecord_dir(data_root)
    tfrecord_dest = (
        "/root/repo/CDPR_Dataset/cdpr_dataset/datasets/"
        "cdpr_synth/libero_spatial_no_noops/tfrecords"
    )
    safe_symlink(tfrecord_src, tfrecord_dest)

    # Quick sanity check
    real = os.path.realpath(tfrecord_dest)
    matches = glob.glob(os.path.join(real, "*.tfrecord"))
    print(f"[RUNPOD] TFRecord files found: {len(matches)}", flush=True)
    if matches:
        print("[RUNPOD] Example:", matches[:3], flush=True)

    # ---------- Launch TensorBoard ----------
    tb_proc = None
    if enable_tb:
        # Note: finetune.py logs to run_dir, which is run_root_dir / run_id.
        # We don't know run_id here, so we point TB to the root and it will pick up subdirs.
        tb_proc = start_tensorboard(run_root_dir, port=tb_port)
        print(f"[RUNPOD] TensorBoard should be reachable on port {tb_port} (expose/forward it in Runpod).", flush=True)
        print(f"[RUNPOD] If using SSH port-forward: ssh -L {tb_port}:localhost:{tb_port} <pod> then open http://localhost:{tb_port}", flush=True)

    # ---------- Run training ----------
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        str(repo_root / "vla-scripts" / "finetune.py"),
        "--vla_path", vla_path,
        "--data_root_dir", data_root,
        "--dataset_name", dataset_name,
        "--run_root_dir", run_root_dir,
        "--num_images_in_input", num_images,
        "--use_proprio", use_proprio,
        "--batch_size", batch_size,
        "--learning_rate", lr,
        "--max_steps", max_steps,
        "--image_aug", image_aug,

        # keep these since your old wrapper passed them (harmless if finetune.py expects them)
        "--wandb_entity", "dummy",
        "--wandb_project", "dummy",

        # IMPORTANT: you said you want the action head; that happens when use_l1_regression=True
        "--use_l1_regression", "True",
        "--use_lora", "True",
    ]

    print("[RUNPOD] Running command:\n", " ".join(cmd), flush=True)
    ret = subprocess.call(cmd)

    # ---------- Cleanup ----------
    if tb_proc is not None:
        print("[RUNPOD] Training finished; TensorBoard is still running (Ctrl+C the pod or kill tensorboard if desired).", flush=True)

    return int(ret)


if __name__ == "__main__":
    sys.exit(main())
