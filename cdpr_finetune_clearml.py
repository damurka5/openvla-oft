from clearml import Task, Dataset
import os, subprocess, sys

PROJECT = "CDPR"
TASK_NAME = "openvla-7b-oft-cdpr"

def main():
    task = Task.init(project_name=PROJECT, task_name=TASK_NAME)

    ds = Dataset.get(dataset_name="cdpr_synth_v1", dataset_project=PROJECT)
    data_root = ds.get_local_copy()

    os.environ["VLA_ROBOT"] = "CDPR"
    os.environ["WANDB_MODE"] = "offline"

    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        "vla-scripts/finetune.py",
        "--vla_path", "moojink/openvla-7b-oft-finetuned-libero-spatial",
        "--data_root_dir", data_root,
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
