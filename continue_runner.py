#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

def main() -> int:
    # Set up environment variables
    os.environ["VLA_ROBOT"] = "CDPR"
    
    # Memory optimization settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # Hard-disable WandB
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_API_KEY"] = "dummy_key_to_prevent_prompt"
    os.environ["WANDB_SILENT"] = "true"
    
    # Paths to your existing checkpoints
    ADAPTER_PATH = "/root/repo/VLA_CDPR/oft_cdpr_ckpts/cdpr_finetune_step10000_20260220-013512_sbs700/vla_cdpr_adapter"
    ACTION_HEAD_PATH = "/root/repo/VLA_CDPR/oft_cdpr_ckpts/cdpr_finetune_step10000_20260220-013512_sbs700/action_head_cdpr.pt"
        
    print("=" * 60)
    print("CONTINUING TRAINING FROM EXISTING CHECKPOINT")
    print(f"Adapter path: {ADAPTER_PATH}")
    print(f"Action head path: {ACTION_HEAD_PATH}")
    print(f"New learning rate: 5e-4")
    print("=" * 60)
    
    # Build the command to run with torchrun
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).parent / "continue_finetune.py"),
        "--vla_path=moojink/openvla-7b-oft-finetuned-libero-spatial",
        "--learning_rate=5e-4",
        "--max_steps=60000",
        "--data_root_dir=/root/repo/cdpr_synth_10hz",
        "--dataset_name=cdpr_local",
        "--run_root_dir=/root/repo/VLA_CDPR/oft_cdpr_ckpts",
        "--use_l1_regression=True",
        "--use_lora=True",
        "--batch_size=4",
        "--num_images_in_input=2",
        "--use_proprio=True",
        "--image_aug=True",
        "--lora_rank=32",
        "--lora_dropout=0.0",
        "--wandb_entity=dummy",
        "--wandb_project=dummy",
        "--wandb_log_freq=5",
    ]
    
    print("[INFO] Running command:")
    print(" ".join(cmd))
    print("=" * 60)
    
    # Run the command
    ret = subprocess.call(cmd)
    
    return ret

if __name__ == "__main__":
    sys.exit(main())