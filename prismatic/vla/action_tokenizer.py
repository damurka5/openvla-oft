"""
action_tokenizer.py

Wraps a base tokenizer with logic to discretize and tokenize continuous robot actions.

Key fix:
- Provide a token-id API so training does NOT round-trip through text.
- This eliminates the "sometimes 41 tokens" issue entirely.
"""

from typing import List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase


class ActionTokenizer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        bins: int = 256,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.n_bins = int(bins)
        self.min_action = float(min_action)
        self.max_action = float(max_action)

        # Uniform bins + centers
        self.bins = np.linspace(self.min_action, self.max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Keep legacy field for compatibility (not used by new encode_to_token_ids path)
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

        # NEW: pick a stable pool of token ids to represent bins.
        # We prefer the tail of vocab (least used), but require round-trip safety:
        # encode(decode([tid])) must equal [tid].
        self.action_token_ids: List[int] = self._select_stable_token_ids(self.n_bins)

        # Map bin index [0..n_bins-1] -> token id
        self.bin_to_token_id = np.array(self.action_token_ids, dtype=np.int64)

    def _select_stable_token_ids(self, needed: int) -> List[int]:
        tok = self.tokenizer
        stable: List[int] = []
        # Search from end of vocab backwards (least used tokens)
        for tid in range(tok.vocab_size - 1, -1, -1):
            s = tok.decode([tid], clean_up_tokenization_spaces=False)
            # Must round-trip to itself as a single token when encoded alone
            ids = tok(s, add_special_tokens=False).input_ids
            if len(ids) == 1 and ids[0] == tid:
                stable.append(tid)
                if len(stable) >= needed:
                    break

        if len(stable) < needed:
            raise RuntimeError(
                f"Could only find {len(stable)} stable tokens, need {needed}. "
                f"Try smaller bins or add special tokens to the tokenizer."
            )

        stable.reverse()  # keep increasing order (optional)
        return stable

    def _discretize(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, a_min=self.min_action, a_max=self.max_action)
        # digitize returns in [1..n_bins] for bins length n_bins
        disc = np.digitize(action, self.bins)
        # convert to [0..n_bins-1] and clip to valid interval centers
        disc = np.clip(disc - 1, a_min=0, a_max=self.n_bins - 1)
        return disc

    def encode_to_token_ids(self, action: np.ndarray) -> List[int]:
        """
        Encode a single action vector (shape [ACTION_DIM]) to token IDs (length ACTION_DIM).
        This is the stable API you should use for training/inference prompt building.
        """
        disc = self._discretize(np.asarray(action, dtype=np.float32))
        if disc.ndim != 1:
            raise ValueError(f"encode_to_token_ids expects 1D action, got shape {disc.shape}")
        return self.bin_to_token_id[disc].tolist()

    def encode_chunk_to_token_ids(self, action_chunk: np.ndarray) -> List[int]:
        """
        Encode action chunk (shape [NUM_ACTIONS_CHUNK, ACTION_DIM]) to token IDs.
        Returns a flat list of length NUM_ACTIONS_CHUNK * ACTION_DIM.
        """
        arr = np.asarray(action_chunk, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"encode_chunk_to_token_ids expects 2D chunk, got shape {arr.shape}")
        ids: List[int] = []
        for row in arr:
            ids.extend(self.encode_to_token_ids(row))
        return ids

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """
        Backward compatible: returns decoded string(s), but this is NOT recommended for training
        because re-tokenization can change lengths.
        """
        arr = np.asarray(action, dtype=np.float32)
        if arr.ndim == 1:
            ids = self.encode_to_token_ids(arr)
            return self.tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        elif arr.ndim == 2:
            # batch of actions (B, ACTION_DIM)
            outs = []
            for row in arr:
                ids = self.encode_to_token_ids(row)
                outs.append(self.tokenizer.decode(ids, clean_up_tokenization_spaces=False))
            return outs
        else:
            raise ValueError(f"Unsupported action shape {arr.shape}")

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Decode token IDs back to continuous action values using bin centers.
        Works with ids produced by encode_to_token_ids/encode_chunk_to_token_ids.
        """
        ids = np.asarray(action_token_ids, dtype=np.int64)

        # token id -> bin index
        # Build inverse map once (fast enough here)
        inv = {tid: i for i, tid in enumerate(self.action_token_ids)}
        flat = ids.reshape(-1)
        bins = np.array([inv.get(int(t), 0) for t in flat], dtype=np.int64)
        bins = np.clip(bins, 0, self.bin_centers.shape[0] - 1)

        cont = self.bin_centers[bins]
        return cont.reshape(ids.shape)

    @property
    def vocab_size(self) -> int:
        return self.n_bins
