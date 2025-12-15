"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

import glob
import os
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from predictor.datasets.types import object_type, polyline_type, track_type

track_type_reversed = {v: k for k, v in track_type.items()}
object_type_reversed = {v: k for k, v in object_type.items()}


def find_latest_checkpoint(base_path):
    # Pattern to match all .ckpt files in the base_path recursively
    search_pattern = Path(base_path, 'epoch*', '*.ckpt')
    # List all files matching the pattern
    list_of_files = glob.glob(search_pattern, recursive=True)
    # Find the file with the latest modification time
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file

def set_seed(seed_value=42):
    """
    Set seed for reproducibility in PyTorch Lightning based training.

    Args:
    seed_value (int): The seed value to be set for random number generators.
    """
    # Set the random seed for PyTorch
    torch.manual_seed(seed_value)

    # If using CUDA (PyTorch with GPU)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU

    # Set the random seed for numpy (if using numpy in the project)
    np.random.seed(seed_value)

    # Set the random seed for Python's `random`
    random.seed(seed_value)

    # Set the seed for PyTorch Lightning's internal operations
    pl.seed_everything(seed_value, workers=True)

# def predictions_to_dataframe(predictions):
    # inputs = [prediction["inputs"] for prediction in predictions]
    # df_inputs = inputs_to_dataframe(inputs)
    # selected_keys = [
    #     'predictive_uncertainty',
    #     'predictive_aleatoric',
    #     'predictive_epistemic',
    #     'scene_decoding',
    #     'scene_context',
    #     'agent_embedding',
    #     'map_embedding',
    #     'minADE',
    #     'minFDE',
    #     'minADE6',
    #     'minFDE6',
    #     'miss_rate',
    #     'brier_minFDE6',
    #     'minFDE_hat',
    #     'minFDE6_hat',
    # ]
    # Iterate through the dataloader and extract the data
    # dfs = []
    # for d in predictions:
    #     selected_d = {k: d[k] for k in selected_keys if k in d}
    #     dfs.append(pd.DataFrame(selected_d))
    # df_outputs = pd.concat(dfs, ignore_index=True)
    # df = pd.concat([df_inputs, df_outputs], axis=1)
    # df = pd.DataFrame(predictions)

    # return df

# def inputs_to_dataframe(inputs):
#     selected_keys = [
#         "scenario_id",
#         "dataset_split",
#         "center_objects_type", 
#         "center_objects_id",
#         "center_track_difficulty", 
#         "center_track_type", 
#         "ego_track_difficulty", 
#         "ego_track_type"
#     ]
#     dfs = []
#     for d in inputs:
#         selected_d = {k: d[k] for k in selected_keys if k in d}
#         dfs.append(pd.DataFrame(selected_d))
#     df = pd.concat(dfs, ignore_index=True)

#     df["center_track_type"] = df["center_track_type"].map(track_type_reversed)
#     df["ego_track_type"] = df["ego_track_type"].map(track_type_reversed)
#     df["center_objects_type"] = df["center_objects_type"].map(object_type_reversed)
#     return df

# import pandas as pd
# from pathlib import Path
# import torch
# import dask
# import dask.dataframe as dd

# other_dimensions = ["loss", "predicted_trajectory", "predicted_probability", "scene_decoding", "agent_embedding", "map_embedding", "scene_context"]

# def load_predictions(directory):
#     def load_file_to_df(file_path):
#         # Load the dictionary from the file
#         d = torch.load(file_path, weights_only=False)
#         # Convert any tensors to lists
#         #converted = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in d.items()}
#         converted = {k: v for k, v in d.items() if k not in other_dimensions}
#         converted = {k: v.numpy() for k, v in converted.items() if hasattr(v, "numpy")}
#         # Create a pandas DataFrame
#         return pd.DataFrame(converted)
#     # Create a list of delayed DataFrames
#     delayed_dfs = [dask.delayed(load_file_to_df)(file_path) 
#                 for file_path in Path(directory).glob("*.pt")]
#     # return delayed_dfs
#     # Build the Dask DataFrame from the delayed objects
#     ddf = dd.from_delayed(delayed_dfs)
#     return ddf

# ddf = load_predictions(".")
# subset_ddf = ddf[ddf["center_objects_type"] == 1]
# subset_ddf["diff"] = subset_ddf["center_track_difficulty"] - subset_ddf["minFDE_hat"]
# df = subset_ddf[['center_track_difficulty', 'diff', "center_track_type", "mixtureNLL"]].sample(frac=0.1).compute()

# df = ddf[['center_track_difficulty', "minFDE_hat", 'diff']].sample(frac=0.1).compute()
# df.describe()
# sc = plt.scatter(
#     df["center_track_difficulty"],
#     df["diff"],
#     c=df["mixtureNLL"],  # continuous values here
#     cmap="viridis",  # choose a colormap; viridis is a good default
#     s=2           # fine markers
# )
# plt.xlabel("center_track_difficulty")
# plt.ylabel("diff")
# plt.title("Scatter Plot: center_track_difficulty vs. diff\nColored by continuous variable")
# # Add a colorbar to indicate the continuous scale
# plt.colorbar(sc, label="Continuous Hue Value")
# # Save the figure
# plt.savefig("scatter_continuous.png", dpi=300, bbox_inches="tight")
# plt.close()

# scatter= sns.scatterplot(
#     data=df, 
#     x="center_track_difficulty", 
#     y="diff", 
#     hue="mixtureNLL", 
#     s=5  # marker size
# )
# plt.title("Scatter Plot: center_track_difficulty vs. diff (Colored by center_track_type)")
# plt.savefig("colored_scatter_plot_seaborn.png", dpi=300, bbox_inches="tight")
# plt.close()

# ## colored scatterplots
# plt.figure(figsize=(8, 6))
# for track_type, group_data in df.groupby("center_track_type"):
#     plt.scatter(
#         group_data["center_track_difficulty"], 
#         group_data["diff"], 
#         label=str(track_type),
#         s=2  # make points small/fine
#     )
# plt.xlabel("center_track_difficulty")
# plt.ylabel("diff")
# plt.title("Scatter Plot: center_track_difficulty vs. diff (Colored by center_track_type)")
# plt.legend(title="center_track_type")
# plt.savefig("colored_scatter_plot.png", dpi=300, bbox_inches="tight")
# plt.close()


# ddf[ddf["center_track_difficulty"]>10]["minADE"].mean().compute()


# # We'll create 20 thresholds between min_val and max_val
# min_val = ddf['center_track_difficulty'].min().compute()
# max_val = ddf['center_track_difficulty'].max().compute()
# thresholds = np.linspace(min_val, max_val-10, 20)
# # 2. For each threshold, select rows and compute the mean of minFDE_hat
# mean_values = []
# for i, t in enumerate(thresholds):
#     print("Threshold: ", i," ", t)
#     # Filter
#     subset = ddf[ddf['center_track_difficulty'] >= t].sample(frac=t/20+0.01)
#     # Compute mean
#     mean_minFDE_hat = subset['minFDE_hat'].mean().compute()
#     mean_values.append(mean_minFDE_hat)

# plt.figure(figsize=(8, 6))
# plt.plot(thresholds, mean_values, marker='o')
# plt.xlabel("Threshold for center_track_difficulty")
# plt.ylabel("Mean of minFDE_hat (above threshold)")
# plt.title("Mean of minFDE_hat vs. center_track_difficulty Threshold")
# plt.grid(True)

# # 4. Save the figure
# plt.savefig("mean_minFDE_hat_vs_threshold.png", dpi=300, bbox_inches="tight")
# plt.close()



# def load_file_to_df(file_path):
#     d = torch.load(file_path, weights_only=False)
#     # Convert tensors to numpy arrays (even if multidimensional)
#     data = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in d.items() if k not in other_dimensions}
#     data["file_path"] = str(file_path)
#     # Return a single-row DataFrame
#     return pd.DataFrame(data)
# def load_file_to_decoding(file_path):
#     d = torch.load(file_path, weights_only=False)
#     data = d["scene_context"]
#     return data
# # Get the first 50 .pt files
# files = list(Path(".").glob("*.pt"))[:10]
# del files[4]
# # Load each file into a DataFrame and concatenate them
# df = pd.concat([load_file_to_df(fp) for fp in files], ignore_index=True)
# decodings = torch.stack([load_file_to_decoding(fp) for fp in files], dim=0)
# flattened = decodings.view(decodings.size(0), -1)  # shape: (128, 49152)
# # Normalize each vector to unit length
# norms = flattened.norm(dim=1, keepdim=True)
# normalized = flattened / norms
# # Compute the cosine similarity matrix (128 x 128)
# cosine_similarity_matrix = normalized @ normalized.t()
# print(cosine_similarity_matrix)
# eigenvalues, eigenvectors = torch.linalg.eigh(cosine_similarity_matrix)
# threshold = torch.quantile(eigenvalues, 0.2)
# minor_mask = eigenvalues < threshold
# V_minor = eigenvectors[:, minor_mask]
# lambda_minor = eigenvalues[minor_mask]
# uniqueness_contributions = (V_minor ** 2) * lambda_minor
# uniqueness_scores = uniqueness_contributions.sum(dim=1) 
# uniqueness_scores = (normalized @ V_minor).pow(2).sum(dim=1)
# p = eigenvalues / eigenvalues.sum()
# epsilon = 1e-12
# nonzero_mask = p > epsilon
# entropy_exp = torch.exp(-(p[nonzero_mask] * torch.log(p[nonzero_mask])).sum())

# from sklearn.cluster import SpectralClustering
# spectral = SpectralClustering(n_clusters=20, affinity="precomputed", random_state=42)  
# cluster_labels = spectral.fit_predict(cosine_similarity_matrix)  
# df["clusters_spectral"]=cluster_labels
# palette = sns.color_palette("Set1", n_colors=df['clusters_affinity'].nunique())
# scatter= sns.scatterplot(
#     data=df[df["center_objects_type"]==1], 
#     x="center_track_difficulty", 
#     y="minFDE_hat", 
#     hue="clusters_spectral", 
#     palette=palette,
#     s=5,  # marker size
#     legend="full"
# )
# plt.title("Scatter Plot: center_track_difficulty vs. diff (Colored by cluster_label)")
# plt.savefig("clusters_scatter.png", dpi=300, bbox_inches="tight")
# plt.close()