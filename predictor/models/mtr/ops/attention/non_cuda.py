import torch


def _compute_key_global_indices(
    query_batch_cnt: torch.Tensor,
    key_batch_cnt: torch.Tensor,
    index_pair_batch: torch.Tensor,
    index_pair: torch.Tensor,
):
    """
    Build a tensor of global key indices for each (query, local_slot).

    Args:
        query_batch_cnt: [bs] int
        key_batch_cnt: [bs] int
        index_pair_batch: [total_query_num] int, batch id for each query
        index_pair: [total_query_num, local_size] int, local key idx in that batch
                    (-1 means "no neighbor")

    Returns:
        safe_idx: [total_query_num, local_size] long, global key indices;
                  invalid positions (where index_pair == -1) are clamped to 0.
        mask_invalid: [total_query_num, local_size] bool, True where index_pair == -1.
    """
    # Make sure we have long for indexing
    if query_batch_cnt.dtype != torch.long:
        query_batch_cnt = query_batch_cnt.long()
    if key_batch_cnt.dtype != torch.long:
        key_batch_cnt = key_batch_cnt.long()
    if index_pair_batch.dtype != torch.long:
        index_pair_batch = index_pair_batch.long()
    if index_pair.dtype != torch.long:
        index_pair = index_pair.long()

    device = key_batch_cnt.device
    bs = query_batch_cnt.numel()

    # key_batch_cnt: [bs] -> key_offsets: [bs]
    # key_offsets[b] = sum_{i<b} key_batch_cnt[i]
    key_offsets = torch.cumsum(key_batch_cnt, dim=0) - key_batch_cnt  # [bs]

    # For each query q, get its batch id, then broadcast to local_size
    total_query_num, local_size = index_pair.shape
    batch_ids = index_pair_batch.view(-1, 1).expand(-1, local_size)  # [Q, L]

    # Global indices: offset_of_batch + local_index
    global_idx = key_offsets[batch_ids] + index_pair  # [Q, L]

    # Handle invalid entries (index_pair == -1)
    mask_invalid = index_pair < 0
    safe_idx = global_idx.clone()
    safe_idx[mask_invalid] = 0  # any valid index; we'll mask them out later

    return safe_idx.to(device=device), mask_invalid.to(device=device)


def attention_weight_computation(
    query_batch_cnt: torch.Tensor,
    key_batch_cnt: torch.Tensor,
    index_pair_batch: torch.Tensor,
    index_pair: torch.Tensor,
    query_features: torch.Tensor,
    key_features: torch.Tensor,
):
    """
    Generate the attention weight matrix.

    Args:
        query_batch_cnt: [bs] int, number of queries per batch
        key_batch_cnt:   [bs] int, number of keys   per batch
        index_pair_batch: [total_query_num] int, batch id of each query
        index_pair: [total_query_num, local_size] int, local key index per query
                    (-1 means no neighbor)
        query_features: [total_query_num, nhead, hdim] float
        key_features:   [total_key_num, nhead, hdim] float

    Returns:
        output: [total_query_num, local_size, nhead] float
    """
    # Ensure contiguity (like the CUDA version did)
    query_batch_cnt = query_batch_cnt.contiguous()
    key_batch_cnt = key_batch_cnt.contiguous()
    index_pair_batch = index_pair_batch.contiguous()
    index_pair = index_pair.contiguous()
    query_features = query_features.contiguous()
    key_features = key_features.contiguous()

    device = query_features.device
    total_query_num, nhead, hdim = query_features.shape
    total_key_num, nhead_k, hdim_k = key_features.shape
    assert nhead == nhead_k and hdim == hdim_k, "Q/K head_dim mismatch"

    # Compute global key indices per (query, local_slot)
    safe_idx, mask_invalid = _compute_key_global_indices(
        query_batch_cnt, key_batch_cnt, index_pair_batch, index_pair
    )
    # shape: safe_idx, mask_invalid -> [Q, L]

    # Gather key features for each (q, l)
    # key_features: [K, H, D]; safe_idx: [Q, L]
    # -> [Q, L, H, D]
    keys_selected = key_features[safe_idx]  # advanced indexing

    # Expand query features to [Q, 1, H, D]
    q_expanded = query_features.unsqueeze(1)  # [Q, 1, H, D]

    # Dot product along D: (Q,L,H,D) * (Q,1,H,D) -> sum_D -> (Q,L,H)
    weights = (q_expanded * keys_selected).sum(dim=-1)  # [Q, L, H]

    # Zero out contributions from invalid neighbors
    weights = weights.masked_fill(mask_invalid.unsqueeze(-1), 0.0)

    return weights.to(device=device)


def attention_value_computation(
    query_batch_cnt: torch.Tensor,
    key_batch_cnt: torch.Tensor,
    index_pair_batch: torch.Tensor,
    index_pair: torch.Tensor,
    attn_weight: torch.Tensor,
    value_features: torch.Tensor,
):
    """
    Generate the attention result.

    Args:
        query_batch_cnt: [bs] int
        key_batch_cnt: [bs] int
        index_pair_batch: [total_query_num] int, batch id of each query
        index_pair: [total_query_num, local_size] int, local key index per query
                    (-1 means no neighbor)
        attn_weight: [total_query_num, local_size, nhead] float
        value_features: [total_key_num, nhead, hdim] float

    Returns:
        output: [total_query_num, nhead, hdim] float
    """
    query_batch_cnt = query_batch_cnt.contiguous()
    key_batch_cnt = key_batch_cnt.contiguous()
    index_pair_batch = index_pair_batch.contiguous()
    index_pair = index_pair.contiguous()
    attn_weight = attn_weight.contiguous()
    value_features = value_features.contiguous()

    device = attn_weight.device
    total_query_num, local_size, nhead = attn_weight.shape
    total_key_num, nhead_v, hdim = value_features.shape
    assert nhead == nhead_v, "attn_weight/value_features head mismatch"

    # Compute global key indices per (q, l)
    safe_idx, mask_invalid = _compute_key_global_indices(
        query_batch_cnt, key_batch_cnt, index_pair_batch, index_pair
    )

    # Gather value features: [Q, L, H, D]
    values_selected = value_features[safe_idx]

    # Mask out invalid neighbors in the weights
    attn_w = attn_weight.masked_fill(mask_invalid.unsqueeze(-1), 0.0)  # [Q,L,H]
    attn_w = attn_w.unsqueeze(-1)  # [Q,L,H,1]

    # Weighted sum over local_size -> [Q,H,D]
    output = (attn_w * values_selected).sum(dim=1)

    return output.to(device=device)
