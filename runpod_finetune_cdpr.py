#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
import time
from pathlib import Path

def main() -> int:
    # ---------- User-configurable via env ----------
    # Root directory where your dataset lives - should be the PARENT directory
    data_root = os.environ.get("DATA_ROOT_DIR", "/root/repo/cdpr_synth_10hz").strip()
    
    # Repo root
    repo_root = Path(os.environ.get("REPO_ROOT", "/root/repo")).resolve()
    
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
    max_steps = os.environ.get("MAX_STEPS", "60_000")
    batch_size = os.environ.get("BATCH_SIZE", "4")
    lr = os.environ.get("LEARNING_RATE", "5e-4")
    num_images = os.environ.get("NUM_IMAGES_IN_INPUT", "2")
    use_proprio = os.environ.get("USE_PROPRIO", "False")
    image_aug = os.environ.get("IMAGE_AUG", "False")
    
    # ---------- Environment ----------
    os.environ["VLA_ROBOT"] = "CDPR"
    
    # Memory optimization settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # Hard-disable WandB
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_API_KEY"] = "dummy_key_to_prevent_prompt"
    os.environ["WANDB_SILENT"] = "true"
    
    print(f"[INFO] repo_root={repo_root}", flush=True)
    print(f"[INFO] data_root={data_root}", flush=True)
    print(f"[INFO] run_root_dir={run_root_dir}", flush=True)
    
    # Verify the data exists
    tfrecord_dir = Path(data_root) / "libero_spatial_no_noops" / "tfrecords_human_control"
    stats_file = Path(data_root) / "action_stats_libero_spatial_no_noops.json"
    dataset_stats_file = Path(data_root) / "dataset_statistics.json"
    
    print(f"[INFO] Checking TFRecord directory: {tfrecord_dir}", flush=True)
    if not tfrecord_dir.exists():
        print(f"[ERROR] TFRecord directory not found: {tfrecord_dir}", flush=True)
        print(f"[INFO] Please set DATA_ROOT_DIR to the parent directory containing 'libero_spatial_no_noops/'", flush=True)
        print(f"[INFO] Current data_root: {data_root}", flush=True)
        return 1
    
    tfrecord_files = list(tfrecord_dir.glob("*.tfrecord"))
    print(f"[INFO] Found {len(tfrecord_files)} .tfrecord files", flush=True)
    if tfrecord_files:
        print(f"[INFO] First few files: {[f.name for f in tfrecord_files[:3]]}", flush=True)
    
    if not stats_file.exists():
        print(f"[WARNING] Action stats file not found: {stats_file}", flush=True)
    else:
        print(f"[INFO] Found action stats file: {stats_file}", flush=True)
    
    if not dataset_stats_file.exists():
        print(f"[WARNING] Dataset statistics file not found: {dataset_stats_file}", flush=True)
    else:
        print(f"[INFO] Found dataset statistics file: {dataset_stats_file}", flush=True)
    
    # ---------- Run training ----------
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        str(repo_root / "openvla-oft" / "vla-scripts" / "finetune.py"),
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
        "--use_l1_regression", "True",
        "--use_lora", "True",
        "--wandb_entity", "dummy",
        "--wandb_project", "dummy",
        "--wandb_log_freq", "5",  # Log every 5 steps
        
        # Memory optimization flags (adjust as needed)
        # "--use_gradient_checkpointing",  # Uncomment if needed
        # "--mixed_precision", "bf16",     # Uncomment if needed
        # "--lora_r", "16",                # Uncomment if needed
        # "--gradient_accumulation_steps", "4",  # Uncomment if needed
    ]
    
    print("[INFO] Running command:\n", " ".join(cmd), flush=True)
    
    # Start TensorBoard in background AFTER training starts
    tb_proc = None
    if enable_tb:
        # Wait a bit for training to create the log directory
        # We'll start TensorBoard in a separate process after a delay
        import threading
        
        def start_tensorboard_later():
            # Wait 30 seconds for training to start and create logs
            time.sleep(30)
            
            # Check if run directory exists
            run_root_path = Path(run_root_dir)
            if run_root_path.exists():
                # Find the latest run directory
                run_dirs = list(run_root_path.glob("*"))
                if run_dirs:
                    latest_run = max(run_dirs, key=os.path.getmtime)
                    print(f"[INFO] Found run directory: {latest_run}", flush=True)
                    tb_cmd = ["tensorboard", f"--logdir={run_root_dir}", "--host=0.0.0.0", f"--port={tb_port}"]
                else:
                    print(f"[INFO] No run directories yet, will monitor: {run_root_dir}", flush=True)
                    tb_cmd = ["tensorboard", f"--logdir={run_root_dir}", "--host=0.0.0.0", f"--port={tb_port}"]
            else:
                print(f"[INFO] Run root doesn't exist yet, will monitor: {run_root_dir}", flush=True)
                tb_cmd = ["tensorboard", f"--logdir={run_root_dir}", "--host=0.0.0.0", f"--port={tb_port}"]
            
            print(f"[INFO] Starting TensorBoard: {' '.join(tb_cmd)}", flush=True)
            return subprocess.Popen(tb_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Start TensorBoard in a separate thread
        tb_thread = threading.Thread(target=lambda: globals().update(tb_proc=start_tensorboard_later()))
        tb_thread.daemon = True
        tb_thread.start()
        
        print(f"[INFO] TensorBoard will start in 30 seconds on port {tb_port}", flush=True)
        print(f"[INFO] If using SSH port-forward: ssh -L {tb_port}:localhost:{tb_port} <server>", flush=True)
    
    # Run training
    ret = subprocess.call(cmd)
    
    # ---------- Cleanup ----------
    if tb_proc is not None:
        print("[INFO] Training finished; TensorBoard still running.", flush=True)
        print("[INFO] To stop TensorBoard: kill the process or Ctrl+C", flush=True)
    
    return int(ret)

if __name__ == "__main__":
    sys.exit(main())