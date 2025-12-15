import os
import sys
import subprocess
from clearml import Task, Dataset
from pathlib import Path

PROJECT = "CDPR"
TASK_NAME = "openvla-7b-oft-cdpr"


def find_tfrecord_dir(dataset_root: str) -> str:
    """
    Walk the ClearML dataset directory and find a folder that contains .tfrecord files.
    Prefer one that has 'libero_spatial_no_noops' in its path, if available.
    """
    print(f"[CDPR] Scanning dataset root for .tfrecord files: {dataset_root}", flush=True)
    candidates = []

    for cur_root, dirs, files in os.walk(dataset_root):
        rel = os.path.relpath(cur_root, dataset_root)
        print(f"[CDPR]   inspecting: {rel}", flush=True)

        if any(f.endswith(".tfrecord") for f in files):
            print(f"[CDPR]     found .tfrecord files in: {cur_root}", flush=True)
            candidates.append(cur_root)

    if not candidates:
        # Extra debug: show what's actually in the root
        print("[CDPR] No .tfrecord files found. Root dir contents:", flush=True)
        print(os.listdir(dataset_root), flush=True)
        raise FileNotFoundError(
            f"[CDPR] No .tfrecord files found under dataset root: {dataset_root}. "
            f"Check what you added to the ClearML dataset."
        )

    # prefer human_control
    for c in candidates:
        if "tfrecords_human_control" in c:
            print(f"[CDPR] Selected TFRecord directory (human_control preferred): {c}", flush=True)
            return c

    # Prefer a directory whose path looks like your original
    for c in candidates:
        if "libero_spatial_no_noops" in c:
            print(f"[CDPR] Selected TFRecord directory (preferred match): {c}", flush=True)
            return c

    # Fallback: first candidate
    print(f"[CDPR] Selected TFRecord directory (first found): {candidates[0]}", flush=True)
    return candidates[0]


def main():
    # 1) Init ClearML task
    # task = Task.init(project_name=PROJECT, task_name=TASK_NAME)
    task = Task.current_task() or Task.init(project_name=PROJECT, task_name=TASK_NAME)
    
    # 2) Get dataset from ClearML
    ds = Dataset.get(dataset_name="cdpr_human_control_only", dataset_project=PROJECT)
    data_root = ds.get_local_copy()
    
    meta_src  = os.path.join(data_root, "meta_dataset.json")
    stats_src = os.path.join(data_root, "action_stats_libero_spatial_no_noops.json")

    meta_dest  = "/root/repo/CDPR_Dataset/cdpr_dataset/datasets/cdpr_synth/meta_dataset.json"
    stats_dest = "/root/repo/CDPR_Dataset/cdpr_dataset/datasets/cdpr_synth/action_stats_libero_spatial_no_noops.json"

    for src, dest in [(meta_src, meta_dest), (stats_src, stats_dest)]:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.islink(dest) or os.path.exists(dest):
            os.remove(dest)
        os.symlink(src, dest)
        print("[CDPR] Linked", dest, "->", os.path.realpath(dest), flush=True)


    print(f"[CDPR] ClearML dataset root: {data_root}", flush=True)

    # 3) Auto-detect where TFRecords actually live inside the dataset
    tfrecord_src = find_tfrecord_dir(data_root)

    # The path that your finetune config still expects
    tfrecord_dest = (
        "/root/repo/CDPR_Dataset/cdpr_dataset/datasets/"
        "cdpr_synth/libero_spatial_no_noops/tfrecords"
    )

    print(f"[CDPR] Expecting TFRecords at: {tfrecord_dest}", flush=True)
    print(f"[CDPR] Creating symlink -> actual ClearML path: {tfrecord_src}", flush=True)

    os.makedirs(os.path.dirname(tfrecord_dest), exist_ok=True)

    # If dest already exists, clean it up (just in case)
    if os.path.islink(tfrecord_dest) or os.path.exists(tfrecord_dest):
        try:
            os.remove(tfrecord_dest)
        except IsADirectoryError:
            import shutil
            shutil.rmtree(tfrecord_dest)

    os.symlink(tfrecord_src, tfrecord_dest)
    
    real = os.path.realpath(tfrecord_dest)
    print("[CDPR] Symlink points to:", real, flush=True)

    import glob
    matches = glob.glob(os.path.join(real, "libero_spatial_no_noops-train-*.tfrecord"))
    print(f"[CDPR] Matched TFRecords: {len(matches)}", flush=True)
    print("[CDPR] Example:", matches[:3], flush=True)

    # 4) Env vars - Disable WandB completely
    os.environ["VLA_ROBOT"] = "CDPR"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "offline"
    
    # Also set to prevent interactive prompt
    os.environ["WANDB_API_KEY"] = "dummy_key_to_prevent_prompt"
    os.environ["WANDB_SILENT"] = "true"

    # 5) Build torchrun command with TensorBoard setup
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        "vla-scripts/finetune.py",
        "--vla_path",
        "moojink/openvla-7b-oft-finetuned-libero-spatial",
        "--data_root_dir",
        data_root,
        "--dataset_name",
        "cdpr_local",
        "--run_root_dir",
        "./VLA_CDPR/oft_cdpr_ckpts",
        "--num_images_in_input",
        "2",
        "--use_proprio",
        "True",
        "--batch_size",
        "1",
        "--learning_rate",
        "1e-4",
        "--max_steps",
        "100", # 1000
        "--image_aug",
        "False",
        # Disable WandB - but the script doesn't have this argument!
        # Let's check what arguments finetune.py actually accepts
    ]
    
    # First, let's see what arguments finetune.py accepts
    # Looking at the code, it uses draccus.wrap() which means all FinetuneConfig fields
    # are available as command line arguments
    
    # Add arguments to disable wandb (based on FinetuneConfig in the script)
    cmd.extend([
        "--wandb_entity", "dummy",
        "--wandb_project", "dummy",
        # The script doesn't have use_wandb argument, so we need to prevent wandb.init()
        # by setting environment variables instead
    ])
    
    print(f"[CDPR] CWD: {os.getcwd()}", flush=True)
    print(f"[CDPR] Script dir: {Path(__file__).resolve().parent}", flush=True)

    print("Running command:", " ".join(cmd), flush=True)
    
    # Run training
    ret = subprocess.call(cmd)

    # Upload the finetuned checkpoint folder as an artifact
    repo_root = Path(__file__).resolve().parent
    ckpt_root = repo_root / "VLA_CDPR" / "oft_cdpr_ckpts"

    if ckpt_root.exists():
        print(f"[CDPR] Uploading checkpoint folder as ClearML artifact: {ckpt_root}", flush=True)
        task.upload_artifact(
            name="openvla_cdpr_finetuned",
            artifact_object=str(ckpt_root),
        )
        
        # ClearML automatically logs console output, no need for special TensorBoard setup
    else:
        print(f"[CDPR] WARNING: checkpoint root not found: {ckpt_root}", flush=True)

    return ret


if __name__ == "__main__":
    sys.exit(main())