import torch
from torch.autograd import Function

def _knn_batch_pytorch(xyz, query_xyz, batch_idxs, query_batch_offsets, k: int):
    """
    Pure PyTorch implementation of the original knn_batch kernel.

    Args:
        xyz: (n, 3) float32, CUDA, contiguous
        query_xyz: (m, 3) float32, CUDA, contiguous
        batch_idxs: (n,) int32/int64, CUDA, contiguous
        query_batch_offsets: (B+1,) int32/int64, CUDA, contiguous
        k: int

    Returns:
        idx: (n, k) int32, CUDA
            For each point i, idx[i, :] are indices in [0, mb-1]
            relative to that batch's query slice (same as i - start).
    """
    assert xyz.is_cuda and query_xyz.is_cuda
    assert batch_idxs.is_cuda and query_batch_offsets.is_cuda
    assert xyz.is_contiguous()
    assert query_xyz.is_contiguous()
    assert batch_idxs.is_contiguous()
    assert query_batch_offsets.is_contiguous()

    device = xyz.device
    n = xyz.size(0)
    m = query_xyz.size(0)
    assert k <= m, "k must be <= total number of query points"

    # make sure integer tensors are long for indexing
    if batch_idxs.dtype != torch.long:
        batch_idxs = batch_idxs.long()
    if query_batch_offsets.dtype != torch.long:
        query_batch_offsets = query_batch_offsets.long()

    B = query_batch_offsets.numel() - 1
    idx_out = torch.empty((n, k), dtype=torch.int32, device=device)

    for b in range(B):
        start = int(query_batch_offsets[b].item())
        end = int(query_batch_offsets[b + 1].item())
        if end <= start:
            continue

        # mask for xyz belonging to this batch
        mask = (batch_idxs == b)
        if not mask.any():
            continue

        xyz_b = xyz[mask]               # (nb, 3)
        query_b = query_xyz[start:end]  # (mb, 3)

        # pairwise distances: (nb, mb)
        dists = torch.cdist(xyz_b, query_b, p=2)  # uses CUDA kernels under the hood

        # smallest k distances per point
        _, topk_idx = torch.topk(dists, k, dim=1, largest=False)

        # topk_idx is in [0, mb-1], which matches original besti[i] = i - start
        idx_out[mask] = topk_idx.to(torch.int32)

    return idx_out


def _knn_batch(xyz, query_xyz, batch_idxs, query_batch_offsets, idx, n, m, k):
    """
    Drop-in replacement for C++ knn_batch:

    void knn_batch(at::Tensor xyz_tensor, at::Tensor query_xyz_tensor,
                   at::Tensor batch_idxs_tensor, at::Tensor query_batch_offsets_tensor,
                   at::Tensor idx_tensor, int n, int m, int k);
    """
    # Basic consistency checks (like the C++ version)
    assert xyz.size(0) == n
    assert query_xyz.size(0) == m
    assert idx.size(0) == n and idx.size(1) == k

    out = _knn_batch_pytorch(
        xyz=xyz,
        query_xyz=query_xyz,
        batch_idxs=batch_idxs,
        query_batch_offsets=query_batch_offsets,
        k=k,
    )

    # write into the provided tensor, in-place
    idx.copy_(out)


def _knn_batch_mlogk(xyz, query_xyz, batch_idxs, query_batch_offsets, idx, n, m, k):
    """
    Drop-in replacement for C++ knn_batch_mlogk with same signature.
    Original CUDA implementation was O(m log k); here we just reuse
    the dense version for simplicity.
    """
    assert k <= 128, "original kernel assumed k <= 128"
    _knn_batch(xyz, query_xyz, batch_idxs, query_batch_offsets, idx, n, m, k)



def knn_batch(xyz, query_xyz, batch_idxs, query_batch_offsets, k):
    """
    Pure PyTorch KNN (drop-in replacement for old KNNBatch.apply)

    Args:
        xyz: (n,3) float32 cuda
        query_xyz: (m,3)
        batch_idxs: (n,)
        query_batch_offsets: (B+1,)
        k: int

    Returns:
        idx: (n,k) int32
    """
    n = xyz.size(0)
    m = query_xyz.size(0)

    idx = torch.zeros((n, k), dtype=torch.int32, device=xyz.device)

    _knn_batch(xyz, query_xyz, batch_idxs, query_batch_offsets, idx, n, m, k)

    return idx


def knn_batch_mlogk(xyz, query_xyz, batch_idxs, query_batch_offsets, k):
    """
    Pure PyTorch mlogk version (drop-in replacement for KNNBatchMlogK.apply)
    """
    n = xyz.size(0)
    m = query_xyz.size(0)

    idx = torch.zeros((n, k), dtype=torch.int32, device=xyz.device)

    _knn_batch_mlogk(xyz, query_xyz, batch_idxs, query_batch_offsets, idx, n, m, k)

    return idx