# debug_cdpr_dataset.py

import tensorflow as tf
from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS
from prismatic.vla.datasets.rlds.dataset import make_interleaved_dataset

cfg = OXE_DATASET_CONFIGS["cdpr_local"]

dataset = make_interleaved_dataset(
    dataset_name="cdpr_local",
    data_root_dir="/root/repo/CDPR_Dataset",  # or the root you're using
    dataset_configs={"cdpr_local": cfg},
    mixture_weights={"cdpr_local": 1.0},
    train=False,  # or True; doesn't matter for one batch
)

for batch in dataset.take(1):
    print("Keys:", batch.keys())
    print("observation keys:", batch["observation"].keys())
    print("action shape:", batch["action"].shape)
    for k, v in batch["observation"].items():
        try:
            print(k, v.shape)
        except Exception:
            print(k, type(v))
