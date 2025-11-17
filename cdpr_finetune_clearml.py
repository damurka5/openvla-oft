import os
import sys
import subprocess
from clearml import Task, Dataset

PROJECT = "CDPR"
TASK_NAME = "openvla-7b-oft-cdpr"

def main():
    # 1) Init ClearML task
    task = Task.init(project_name=PROJECT, task_name=TASK_NAME)

    # 2) Get dataset from ClearML
    ds = Dataset.get(dataset_name="cdpr_synth_v1", dataset_project=PROJECT)
    data_root = ds.get_local_copy()  # e.g. /clearml_agent_cache/storage_manager/datasets/ds_xxx

    print(f"[CDPR] ClearML dataset root: {data_root}", flush=True)

    # === NEW: recreate your old local path inside the container via symlink ===

    # Where ClearML actually put your tfrecords *inside* the dataset
    # (because you did: clearml-data add --files CDPR_Dataset/cdpr_dataset/.../tfrecords)
    tfrecord_src = os.path.join(
        data_root,
        "CDPR_Dataset/cdpr_dataset/datasets/cdpr_synth/libero_spatial_no_noops/tfrecords",
    )

    # The hard-coded path that finetune.py is still looking for
    tfrecord_dest = "/root/repo/CDPR_Dataset/cdpr_dataset/datasets/cdpr_synth/libero_spatial_no_noops/tfrecords"

    print(f"[CDPR] Expecting TFRecords at: {tfrecord_dest}", flush=True)
    print(f"[CDPR] Creating symlink -> actual ClearML path: {tfrecord_src}", flush=True)

    # Make parent dirs and create symlink if needed
    os.makedirs(os.path.dirname(tfrecord_dest), exist_ok=True)

    if not os.path.exists(tfrecord_dest):
        if not os.path.exists(tfrecord_src):
            raise FileNotFoundError(
                f"[CDPR] TFRecord source directory not found at {tfrecord_src}. "
                f"Check dataset contents or adjust path in cdpr_finetune_clearml.py."
            )
        os.symlink(tfrecord_src, tfrecord_dest)

    # 3) Environment variables (same as before)
    os.environ["VLA_ROBOT"] = "CDPR"
    os.environ["WANDB_MODE"] = "offline"

    # 4) Build your torchrun command
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        "vla-scripts/finetune.py",
        "--vla_path", "moojink/openvla-7b-oft-finetuned-libero-spatial",
        "--data_root_dir", data_root,   # still pass this; may be used for other bits
        "--dataset_name", "cdpr_local",
        "--run_root_dir", "./VLA_CDPR/oft_cdpr_ckpts",
        "--num_images_in_input", "2",
        "--use_proprio", "True",
        "--batch_size", "2",
        "--learning_rate", "1e-4",
        "--max_steps", "3000",
    ]

    print("Running command:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)

if __name__ == "__main__":
    sys.exit(main())
