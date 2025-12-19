"""
openvla.py

PyTorch Module defining OpenVLA as a lightweight wrapper around a PrismaticVLM; defines custom logic around
discretizing actions with the ActionTokenizer.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import LlamaTokenizerFast

from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.action_tokenizer import ActionTokenizer

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# prismatic/models/openvla.py (or prismatic/models/vla.py / model builder file)

import torch
from prismatic.vla.constants import ACTION_DIM
from prismatic.models.action_heads import L1RegressionActionHead

def build_l1_action_head_for_model(vla_model) -> L1RegressionActionHead:
    """
    Build an L1RegressionActionHead using the *actual* LLM hidden size of the loaded model.
    This prevents the 5632 vs 4096 mismatch and also ensures action_dim matches ACTION_DIM.
    """
    # Try common attribute names; adjust if your model uses different names
    if hasattr(vla_model, "language_model") and hasattr(vla_model.language_model, "config"):
        hidden_size = int(vla_model.language_model.config.hidden_size)
    elif hasattr(vla_model, "llm") and hasattr(vla_model.llm, "config"):
        hidden_size = int(vla_model.llm.config.hidden_size)
    else:
        raise RuntimeError("Cannot find language model hidden size on vla_model.")

    head = L1RegressionActionHead(
        input_dim=hidden_size,        # must match actions_hidden_states.shape[-1] (4096 in your logs)
        hidden_dim=hidden_size,
        action_dim=ACTION_DIM,        # 5 for CDPR
        mlp_input_dim=None,           # your head will compute expected automatically
    )
    print(f"[ACTION_HEAD BUILD] hidden_size={hidden_size}, action_dim={ACTION_DIM}", flush=True)
    return head


def replace_action_head_if_shape_mismatch(vla_model):
    """
    If the current head was loaded from a checkpoint with incompatible shapes,
    replace it with a fresh head that matches the current model + ACTION_DIM.
    """
    # Find the head (adjust attribute name if different)
    head = getattr(vla_model, "action_head", None)
    if head is None:
        vla_model.action_head = build_l1_action_head_for_model(vla_model)
        return

    # Determine expected hidden size
    if hasattr(vla_model, "language_model") and hasattr(vla_model.language_model, "config"):
        expected_hidden = int(vla_model.language_model.config.hidden_size)
    elif hasattr(vla_model, "llm") and hasattr(vla_model.llm, "config"):
        expected_hidden = int(vla_model.llm.config.hidden_size)
    else:
        raise RuntimeError("Cannot find language model hidden size on vla_model.")

    # Replace if mismatch
    head_input_dim = int(getattr(head, "input_dim", -1))
    head_action_dim = int(getattr(head, "action_dim", -1))
    if head_input_dim != expected_hidden or head_action_dim != ACTION_DIM:
        print(
            f"[ACTION_HEAD REPLACE] old(input_dim={head_input_dim}, action_dim={head_action_dim}) "
            f"!= expected(input_dim={expected_hidden}, action_dim={ACTION_DIM}). Reinitializing head.",
            flush=True,
        )
        vla_model.action_head = build_l1_action_head_for_model(vla_model)
    else:
        print("[ACTION_HEAD KEEP] head dims already match model.", flush=True)


class OpenVLA(PrismaticVLM):
    def __init__(
        self,
        *args,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
        action_tokenizer: ActionTokenizer,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.norm_stats = norm_stats
        self.action_tokenizer = action_tokenizer

    @torch.inference_mode()
    def predict_action(
        self, image: Image, instruction: str, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action (de-tokenizes).

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        # Build VLA Prompt
        prompt_builder = self.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            # If the special empty token ('') does not already appear after the colon (':') token in the prompt
            # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # Preprocess Image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            generated_ids = super(PrismaticVLM, self).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
                max_new_tokens=self.get_action_dim(unnorm_key),
                **kwargs
            )
            # fmt: on

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :]
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict, unnorm_key: Optional[str]) -> str:
        """
        Resolve the unnormalization key for action statistics.

        For multi-dataset models, if `unnorm_key` is None we now *default* to the first
        available key instead of raising. This is important for custom continuous-action
        heads (e.g. CDPR) that do not actually use these statistics.
        """
        keys = list(norm_stats.keys())

        if unnorm_key is None:
            if len(keys) > 1:
                print(
                    "⚠️ Model has multiple dataset statistics; no `unnorm_key` was given. "
                    f"Defaulting to the first key: {keys[0]}"
                )
            unnorm_key = keys[0]

        if unnorm_key not in norm_stats:
            print(
                f"⚠️ Requested `unnorm_key`='{unnorm_key}' not found in statistics. "
                f"Falling back to first key: {keys[0]}"
            )
            unnorm_key = keys[0]

        return unnorm_key


    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return self.norm_stats[unnorm_key]["action"]
