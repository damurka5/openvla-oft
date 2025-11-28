"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights

def debug_actions(actions):
    print(f"[DEBUG ACTIONS] Type: {type(actions)}", flush=True)
    if hasattr(actions, 'shape'):
        print(f"[DEBUG ACTIONS] Shape: {actions.shape}", flush=True)
    if hasattr(actions, 'dtype'):
        print(f"[DEBUG ACTIONS] Dtype: {actions.dtype}", flush=True)
    print(f"[DEBUG ACTIONS] Sample values: {actions[:2] if len(actions) > 2 else actions}", flush=True)
    return actions

# Debug what the action tokenizer expects
def debug_action_tokenizer(action_tokenizer, sample_action):
    """Debug what the action tokenizer can handle"""
    print(f"[DEBUG ACTION TOKENIZER] Sample action shape: {sample_action.shape}, dtype: {sample_action.dtype}", flush=True)
    
    # Try different input formats
    try:
        # Try as numpy array
        result1 = action_tokenizer(sample_action)
        print(f"[DEBUG ACTION TOKENIZER] Numpy array works: {result1}", flush=True)
    except Exception as e:
        print(f"[DEBUG ACTION TOKENIZER] Numpy array failed: {e}", flush=True)
    
    try:
        # Try as torch tensor
        result2 = action_tokenizer(torch.tensor(sample_action))
        print(f"[DEBUG ACTION TOKENIZER] Torch tensor works: {result2}", flush=True)
    except Exception as e:
        print(f"[DEBUG ACTION TOKENIZER] Torch tensor failed: {e}", flush=True)
    
    try:
        # Try as list
        result3 = action_tokenizer(sample_action.tolist())
        print(f"[DEBUG ACTION TOKENIZER] List works: {result3}", flush=True)
    except Exception as e:
        print(f"[DEBUG ACTION TOKENIZER] List failed: {e}", flush=True)

def debug_proprio_structure(proprio):
    print(f"[DEEP DEBUG PROPRIO] Full proprio structure:", flush=True)
    print(f"  Shape: {proprio.shape}", flush=True)
    print(f"  Dtype: {proprio.dtype}", flush=True)
    print(f"  Total elements: {proprio.size}", flush=True)
    
    if len(proprio.shape) >= 1:
        print(f"  First element shape: {proprio[0].shape}", flush=True)
        print(f"  First element: {proprio[0]}", flush=True)
    
    if len(proprio.shape) >= 2:
        print(f"  Second dimension size: {proprio.shape[1]}", flush=True)
        
    if len(proprio.shape) >= 3:
        print(f"  Third dimension size: {proprio.shape[2]}", flush=True)

@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False

#     def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
#         """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
#         dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][0]
#         img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
#         lang = rlds_batch["task"]["language_instruction"].decode().lower()
#         actions = rlds_batch["action"]

#         # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
#         prompt_builder = self.prompt_builder_fn("openvla")

#         # Get future action chunk
#         future_actions = rlds_batch["action"][1:]
#         future_actions_string = ''.join(self.action_tokenizer(future_actions))

#         # Get action chunk string
#         current_action_string = self.action_tokenizer(current_action)
#         action_chunk_string = current_action_string + future_actions_string
#         action_chunk_len = len(action_chunk_string)

#         conversation = [
#             {"from": "human", "value": f"What action should the robot take to {lang}?"},
#             {"from": "gpt", "value": action_chunk_string},
#         ]
#         for turn in conversation:
#             prompt_builder.add_turn(turn["from"], turn["value"])

#         # Tokenize (w/ `base_tokenizer`)
#         input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
#         labels = list(input_ids)

#         # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
#         #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
#         input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
#         pixel_values = self.image_transform(img)

#         # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
#         labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
#         if not self.predict_stop_token:
#             labels[-1] = IGNORE_INDEX

#         return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=actions)

#         # Add additional inputs
#         if self.use_wrist_image:
#             all_wrist_pixels = []
#             for k in rlds_batch["observation"].keys():
#                 if "wrist" in k:
#                     img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
#                     pixel_values_wrist = self.image_transform(img_wrist)
#                     all_wrist_pixels.append(pixel_values_wrist)
#             return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
#         if self.use_proprio and "proprio" in rlds_batch["observation"]:
#             proprio = rlds_batch["observation"]["proprio"]
#             return_dict["proprio"] = proprio

#         return return_dict

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name = rlds_batch["dataset_name"]
        
        # FIX: Convert float32 images to uint8 for PIL
        image_primary_data = rlds_batch["observation"]["image_primary"][0]
        if image_primary_data.dtype == np.float32:
            image_primary_data = (image_primary_data * 255).astype(np.uint8)
        img = Image.fromarray(image_primary_data)
        
        # FIX: Handle language instruction
        lang_instruction = rlds_batch["task"]["language_instruction"]
        if hasattr(lang_instruction, 'decode'):
            lang = lang_instruction.decode().lower()
        else:
            lang = lang_instruction[0].decode().lower() if isinstance(lang_instruction, np.ndarray) else str(lang_instruction).lower()
        
        actions = rlds_batch["action"]
        print(f"[DEBUG] Actions shape: {actions.shape}, dtype: {actions.dtype}", flush=True)
        test_action = actions[0, 0]
        tokenized_test = self.action_tokenizer(test_action)
        print(f"[DEBUG ACTION TOKENIZER] Single action tokens: '{tokenized_test}'", flush=True)
        print(f"[DEBUG ACTION TOKENIZER] Single action token length: {len(tokenized_test)}", flush=True)

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")

        # FIX: Use only the first action chunk from the first timestep
        current_action = actions[0, 0]  # Shape: (5,) - single action
        
        # FIX: Limit future actions to prevent sequence from being too long
        max_future_actions = 10  # Limit to prevent sequence overflow
        future_actions = actions[1:1+max_future_actions, 0]  # Shape: (max_future_actions, 5)
        print(f"[DEBUG] Limited future actions shape: {future_actions.shape}", flush=True)

        # FIX: Tokenize actions properly
        try:
            current_action_string = self.action_tokenizer(current_action)
            future_actions_string = ''.join([self.action_tokenizer(action) for action in future_actions])
        except Exception as e:
            print(f"[ERROR] Action tokenizer failed: {e}", flush=True)
            current_action_string = ''
            future_actions_string = ''

        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        print(f"[DEBUG TOKENS] Action chunk string length: {action_chunk_len}", flush=True)
        print(f"[DEBUG TOKENS] Action chunk preview: {action_chunk_string[:100]}...", flush=True)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        print(f"[DEBUG TOKENS] Total input IDs length: {len(input_ids)}", flush=True)
        print(f"[DEBUG TOKENS] Base tokenizer vocab size: {self.base_tokenizer.vocab_size}", flush=True)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=actions)

        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    wrist_image_data = rlds_batch["observation"][k][0]
                    if wrist_image_data.dtype == np.float32:
                        wrist_image_data = (wrist_image_data * 255).astype(np.uint8)
                    img_wrist = Image.fromarray(wrist_image_data)
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            print(f"[DEBUG] Proprio shape: {proprio.shape}, dtype: {proprio.dtype}", flush=True)
            
            # Take the first timestep and first chunk
            if len(proprio.shape) == 3:
                proprio_processed = proprio[0, 0]
            else:
                proprio_processed = proprio[0]
            
            proprio_tensor = torch.tensor(proprio_processed, dtype=torch.float32).unsqueeze(0)
            print(f"[DEBUG] Final proprio shape for model: {proprio_tensor.shape}", flush=True)
            return_dict["proprio"] = proprio_tensor

        print(f"[DEBUG TOKENS] Input IDs length: {len(input_ids)}", flush=True)
        print(f"[DEBUG TOKENS] Action chunk string length: {len(action_chunk_string)}", flush=True)
        print(f"[DEBUG TOKENS] Action chunk tokens: {action_chunk_string}", flush=True)
        return return_dict

class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            dataset_or_mixture_name=self.data_mix,
            data_root_dir=self.data_root_dir,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            balance_weights=False,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )

        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=NUM_ACTIONS_CHUNK-1,      # For action chunking
                skip_unlabeled=False,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
