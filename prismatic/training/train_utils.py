"""Utils for training/fine-tuning scripts."""

import torch

from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX


# def get_current_action_mask(token_ids):
#     # Create a tensor marking positions of IGNORE_INDEX
#     newline_positions = token_ids != IGNORE_INDEX

#     # Calculate cumulative sum to identify regions between newlines
#     cumsum = torch.cumsum(newline_positions, dim=1)

#     # Create the mask
#     mask = (1 <= cumsum) & (cumsum <= ACTION_DIM)

#     # Extract the action part only
#     action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
#     mask = action_tokens_only_mask * mask

#     return mask
def get_current_action_mask(labels):
    # Handle None case for inference
    if labels is None:
        # Return a dummy mask - the shape doesn't matter since it won't be used
        return None
    
    batch_size, seq_len = labels.shape
    
    # Find newline positions (token_id = 13)
    newline_positions = (labels == 13)
    
    # Convert bool to int before cumsum
    if newline_positions.dtype == torch.bool:
        newline_positions = newline_positions.to(torch.int64)
    
    # Compute cumulative sum
    cumsum = torch.cumsum(newline_positions, dim=1)
    
    # Create mask where cumulative sum is even (after each pair of newlines)
    current_action_mask = (cumsum % 2 == 0) & (labels != -100)
    
    # Ensure mask is boolean
    return current_action_mask.bool()

def get_next_actions_mask(token_ids):
    # Create a tensor marking positions of IGNORE_INDEX
    newline_positions = token_ids != IGNORE_INDEX

    # Calculate cumulative sum to identify regions between newlines
    cumsum = torch.cumsum(newline_positions, dim=1)

    # Create the mask
    mask = cumsum > ACTION_DIM

    # Extract the action part only
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask

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
