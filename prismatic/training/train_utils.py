"""Utils for training/fine-tuning scripts."""

import torch

from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK

def get_current_action_mask(labels: torch.Tensor) -> torch.Tensor:
    """
    Returns a boolean mask over `labels` (shape [B, L]) selecting ONLY the
    first ACTION_DIM supervised action tokens per example.

    Works with any tokenizer and any action-token id scheme.
    """
    if labels is None:
        return None

    B, L = labels.shape
    mask = torch.zeros((B, L), dtype=torch.bool, device=labels.device)

    expected_total = NUM_ACTIONS_CHUNK * ACTION_DIM  # e.g. 40

    for b in range(B):
        pos = torch.nonzero(labels[b] != IGNORE_INDEX, as_tuple=False).squeeze(1)

        # If EOS is also supervised, pos might be 41. Keep only the first expected_total.
        if pos.numel() > expected_total:
            pos = pos[:expected_total]

        # Current action = first ACTION_DIM tokens (if available)
        if pos.numel() > 0:
            cur = pos[: min(ACTION_DIM, pos.numel())]
            mask[b, cur] = True

    return mask


def get_next_actions_mask(labels: torch.Tensor) -> torch.Tensor:
    """
    Returns a boolean mask over `labels` (shape [B, L]) selecting the
    supervised tokens AFTER the first ACTION_DIM action tokens per example,
    up to NUM_ACTIONS_CHUNK*ACTION_DIM total tokens.

    Works with any tokenizer and any action-token id scheme.
    """
    if labels is None:
        return None

    B, L = labels.shape
    mask = torch.zeros((B, L), dtype=torch.bool, device=labels.device)

    expected_total = NUM_ACTIONS_CHUNK * ACTION_DIM  # e.g. 40

    for b in range(B):
        pos = torch.nonzero(labels[b] != IGNORE_INDEX, as_tuple=False).squeeze(1)

        # If EOS is supervised, pos might be 41. Keep only first expected_total.
        if pos.numel() > expected_total:
            pos = pos[:expected_total]

        # Next actions = everything after the first ACTION_DIM tokens
        if pos.numel() > ACTION_DIM:
            nxt = pos[ACTION_DIM:]
            mask[b, nxt] = True

    return mask

def compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask):
    correct_preds = (predicted_token_ids == ground_truth_token_ids) & mask
    accuracy = correct_preds.sum().float() / mask.sum().float()
    return accuracy


def compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask):
    pred_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(predicted_token_ids[mask].cpu().numpy())
    )
    true_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(ground_truth_token_ids[mask].cpu().numpy())
    )
    l1_loss = torch.nn.functional.l1_loss(pred_continuous_actions, true_continuous_actions)
    return l1_loss
