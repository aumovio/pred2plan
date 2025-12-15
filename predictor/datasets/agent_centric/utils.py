"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
import random
import numpy as np
import torch
from torch.utils.data import Sampler


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z_tensor(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    if points.shape[-1] == 2:
        rot_matrix = torch.stack((
            cosa, sina,
            -sina, cosa
        ), dim=1).view(-1, 2, 2).float()
        points_rot = torch.matmul(points, rot_matrix)
    else:
        ones = angle.new_ones(points.shape[0])
        rot_matrix = torch.stack((
            cosa, sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(-1, 3, 3).float()
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def rotate_points_along_z(points, angle):
    """
    Rotate points around the Z-axis using the given angle.

    Args:
        points: ndarray of shape (B, N, 3 + C) - B batches, N points per batch, 3 coordinates (x, y, z) + C extra channels
        angle: ndarray of shape (B,) - angles for each batch in radians

    Returns:
        Rotated points as an ndarray.
    """

    # Checking if the input is 2D or 3D points
    is_2d = points.shape[-1] == 2

    # Cosine and sine of the angles
    cosa = np.cos(angle)
    sina = np.sin(angle)

    if is_2d:
        # Rotation matrix for 2D
        rot_matrix = np.stack((
            cosa, sina,
            -sina, cosa
        ), axis=1).reshape(-1, 2, 2)

        # Apply rotation
        points_rot = np.matmul(points, rot_matrix)
    else:
        # Rotation matrix for 3D
        rot_matrix = np.stack((
            cosa, sina, np.zeros_like(angle),
            -sina, cosa, np.zeros_like(angle),
            np.zeros_like(angle), np.zeros_like(angle), np.ones_like(angle)
        ), axis=1).reshape(-1, 3, 3)

        # Apply rotation to the first 3 dimensions
        points_rot = np.matmul(points[:, :, :3], rot_matrix)

        # Concatenate any additional dimensions back
        if points.shape[-1] > 3:
            points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)

    return points_rot

def find_true_segments(mask):
    # Find the indices where `True` changes to `False` and vice versa
    change_points = np.where(np.diff(mask))[0] + 1

    # Add the start and end indices
    indices = np.concatenate(([0], change_points, [len(mask)]))

    # Extract the segments of continuous `True`
    segments = [list(range(indices[i], indices[i + 1])) for i in range(len(indices) - 1) if mask[indices[i]]]

    return segments

# def find_true_segments(mask):
#     m = np.asarray(mask, dtype=np.bool_)
#     n = m.size
#     if n == 0:
#         return []

#     dm = np.diff(m.astype(np.int8))
#     starts = np.flatnonzero(dm == 1) + 1
#     ends   = np.flatnonzero(dm == -1) + 1
#     if m[0]:
#         starts = np.r_[0, starts]
#     if m[-1]:
#         ends = np.r_[ends, n]

#     # materialize indices into lists
#     return [list(range(a, b)) for a, b in zip(starts, ends)]


def merge_batch_by_padding_2nd_dim(tensor_list, return_pad_mask=False):
    assert len(tensor_list[0].shape) in [3, 4]
    only_3d_tensor = False
    if len(tensor_list[0].shape) == 3:
        tensor_list = [x.unsqueeze(dim=-1) for x in tensor_list]
        only_3d_tensor = True
    maxt_feat0 = max([x.shape[1] for x in tensor_list])

    _, _, num_feat1, num_feat2 = tensor_list[0].shape

    ret_tensor_list = []
    ret_mask_list = []
    for k in range(len(tensor_list)):
        cur_tensor = tensor_list[k]
        assert cur_tensor.shape[2] == num_feat1 and cur_tensor.shape[3] == num_feat2, print(cur_tensor.shape)

        new_tensor = cur_tensor.new_zeros(cur_tensor.shape[0], maxt_feat0, num_feat1, num_feat2)
        new_tensor[:, :cur_tensor.shape[1], :, :] = cur_tensor
        ret_tensor_list.append(new_tensor)

        new_mask_tensor = cur_tensor.new_zeros(cur_tensor.shape[0], maxt_feat0)
        new_mask_tensor[:, :cur_tensor.shape[1]] = 1
        ret_mask_list.append(new_mask_tensor.bool())

    ret_tensor = torch.cat(ret_tensor_list, dim=0)  # (num_stacked_samples, num_feat0_maxt, num_feat1, num_feat2)
    ret_mask = torch.cat(ret_mask_list, dim=0)

    if only_3d_tensor:
        ret_tensor = ret_tensor.squeeze(dim=-1)

    if return_pad_mask:
        return ret_tensor, ret_mask
    return ret_tensor


def get_batch_offsets(batch_idxs, bs):
    '''
    :param batch_idxs: (N), int
    :param bs: int
    :return: batch_offsets: (bs + 1)
    '''
    batch_offsets = torch.zeros(bs + 1).int()
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DynamicSampler(Sampler):
    def __init__(self, datasets):
        """
        datasets: Dictionary of datasets.
        epoch_to_datasets: A dict where keys are epoch numbers and values are lists of dataset names to be used in that epoch.
        """
        self.datasets = datasets
        self.config = datasets.config
        all_dataset = self.datasets.dataset_idx.keys()
        self.sample_num = self.config.data['sample_num']
        self.sample_mode = self.config.data['sample_mode']

        data_usage_dict = {}
        max_data_num = self.config.data['max_data_num']
        for k, num in zip(all_dataset, max_data_num):
            data_usage_dict[k] = num
        # self.selected_idx = self.datasets.dataset_idx
        # self.reset()
        self.set_sampling_strategy(data_usage_dict)

    def set_sampling_strategy(self, sampleing_dict):
        all_idx = []
        selected_idx = {}
        for k, v in sampleing_dict.items():
            assert k in self.datasets.dataset_idx.keys()
            data_idx = self.datasets.dataset_idx[k]
            if v <= 1.0:
                data_num = int(len(data_idx) * v)
            else:
                data_num = int(v)
            if data_num == 0:
                continue
            data_num = min(data_num, len(data_idx))
            # randomly select data_idx by data_num
            sampled_data_idx = np.random.choice(data_idx, data_num, replace=False).tolist()
            all_idx.extend(sampled_data_idx)
            selected_idx[k] = sampled_data_idx

        self.idx = all_idx[:self.sample_num]
        self.selected_idx = selected_idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)

    def reset(self):
        all_index = []
        for k, v in self.selected_idx.items():
            all_index.extend(v)
        self.idx = all_index

    def set_idx(self, idx):
        self.idx = idx

