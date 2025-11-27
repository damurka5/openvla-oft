# debug_cdpr_pipeline.py

import os
import tensorflow as tf

from clearml import Task, Dataset

# Make sure we use CDPR constants
os.environ["VLA_ROBOT"] = "CDPR"

# Force tf.function to run eagerly so we get Python stack traces
tf.config.run_functions_eagerly(True)

PROJECT = "CDPR"
DATASET_NAME = "cdpr_synth_v1"

def main():
    task = Task.init(project_name=PROJECT, task_name="dataset_debug")

    # 1) Get the same dataset ClearML mounts for training
    ds = Dataset.get(dataset_name=DATASET_NAME, dataset_project=PROJECT)
    data_root = ds.get_local_copy()
    print("[DEBUG] data_root:", data_root, flush=True)

    # 2) Import RLDSDataset + OXE configs
    from prismatic.vla.datasets.datasets import RLDSDataset
    from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS

    oxe_cfg = OXE_DATASET_CONFIGS["cdpr_local"]

    # Build a minimal rlds_config similar to what finetune.py uses.
    # You might need to tweak traj/frame kwargs to exactly match finetune.py,
    # but this is a good starting point.
    rlds_config = {
        "dataset_kwargs_list": [
            {
                **oxe_cfg,
                "data_root_dir": data_root,
            }
        ],
        "sample_weights": [1.0],
        "train": True,
        "shuffle_buffer_size": 128,
        "traj_transform_kwargs": {
            # If you can, copy these from finetune.py's call into RLDSDataset.
            # To start debugging, you can even try empty dicts, then add things back.
        },
        "frame_transform_kwargs": {
            # Same comment as above.
        },
        "batch_size": None,
        "balance_weights": False,
        "traj_transform_threads": None,
        "traj_read_threads": None,
    }

    # 3) Build RLDSDataset
    ds_obj = RLDSDataset(rlds_config)
    ds_tf = ds_obj.dataset  # this is the DLataset wrapping TF

    print("[DEBUG] Created RLDSDataset, about to iterate...", flush=True)

    # 4) Iterate a couple of elements to trigger the error under eager
    for i, traj in zip(range(3), ds_tf.as_numpy_iterator()):
        print(f"[DEBUG] Got traj {i}")
        print("  keys:", traj.keys())
        print("  obs keys:", traj["observation"].keys())
        print("  action shape:", traj["action"].shape, flush=True)

if __name__ == "__main__":
    main()
