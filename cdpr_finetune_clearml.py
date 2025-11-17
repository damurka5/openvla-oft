import os
import sys
import subprocess
from clearml import Task, Dataset

PROJECT = "CDPR"
TASK_NAME = "openvla-7b-oft-cdpr"


def find_tfrecord_dir(root: str) -> str:
    """
    Recursively search for a directory literally named 'tfrecords'
    somewhere under `root` and return its full path.
    """
    print(f"[CDPR] Scanning dataset root for 'tfrecords' dirs: {root}", flush=True)
    for cur_root, dirs, files in os.walk(root):
        # quick log so you see some structure in ClearML logs
        rel = os.path.relpath(cur_root, root)
        print(f"[CDPR]   inspecting: {rel}", flush=True)

        if os.path.basename(cur_root) == "tfrecords":
            print(f"[CDPR] Found 'tfrecords' directory at: {cur_root}", flush=True)
            return cur_root

    raise FileNotFoundError(
        f"[CDPR] Could not find any directory named 'tfrecords' under {root}. "
        f"Check what you added to the ClearML dataset."
    )


def main():
    # 1) Init ClearML task
    task = Task.init(project_name=PROJECT, task_name=TASK_NAME)

    # 2) Get dataset from ClearML
    ds = Dataset.get(dataset_name="cdpr_synth_v1", dataset_project=PROJECT)
    data_root = ds.get_local_copy()  # e.g. /clearml_agent_cache/storage_manager/datasets/ds_xxx

    print(f"[CDPR] ClearML dataset root: {data_root}", flush=True)

    # 3) Auto-detect where the 'tfrecords' dir actually lives inside the dataset
    tfrecord_src = find_tfrecord_dir(data_root)

    # The path that your finetune config still expects
    tfrecord_dest = (
        "/root/repo/CDPR_Dataset/cdpr_dataset/datasets/"
        "cdpr_synth/libero_spatial_no_noops/tfrecords"
    )

    print(f"[CDPR] Expecting TFRecords at: {tfrecord_dest}", flush=True)
    print(f"[CDPR] Creating symlink -> actual ClearML path: {tfrecord_src}", flush=True)

    os.makedirs(os.path.dirname(tfrecord_dest), exist_ok=True)

    # If destination exists but isn't already a symlink, you *could* clean it up,
    # but in this scenario it shouldn't exist at all.
    if not os.path.exists(tfrecord_dest):
        os.symlink(tfrecord_src, tfrecord_dest)
    else:
        print(f"[CDPR] Destination already exists: {tfrecord_dest}", flush=True)

    # 4) Env vars
    os.environ["VLA_ROBOT"] = "CDPR"
    os.environ["WANDB_MODE"] = "offline"

    # 5) Build torchrun command
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        "vla-scripts/finetune.py",
        "--vla_path",
        "moojink/openvla-7b-oft-finetuned-libero-spatial",
        "--data_root_dir",
        data_root,  # still pass this in case other bits use it
        "--dataset_name",
        "cdpr_local",
        "--run_root_dir",
        "./VLA_CDPR/oft_cdpr_ckpts",
        "--num_images_in_input",
        "2",
        "--use_proprio",
        "True",
        "--batch_size",
        "2",
        "--learning_rate",
        "1e-4",
        "--max_steps",
        "3000",
    ]

    print("Running command:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
