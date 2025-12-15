import time
from pathlib import Path

import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import unbatch

from .av2_multiagent_submission_protocol.submission import ChallengeSubmission


class SubmissionAv2MultiAgent:
    def __init__(self, save_dir: str = "") -> None:
        stamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.submission_file = Path(save_dir) / f"multi_agent_{stamp}.parquet"
        self.challenge_submission = ChallengeSubmission(predictions={})

    @torch.no_grad()
    def format_data(
        self,
        data: Batch,
        prediction: dict,
        normalized_probability=False,
    ) -> None:
        """
        prediction['loc_traj']: (A, K, 60, 2) # prediction are in agent local frame
        prediction['pi']: (B, K)
        normalized_probability: if the input probability is normalized,
        """
        B = data.batch_size
        scenario_ids_list = data["scenario_id"]
        track_ids_list = data["agent"]["agent_ids"]
        pred_traj_local_list = unbatch(prediction["loc_traj"].transpose(0,1), batch=data['agent']['batch'])
        agent_origin_list = unbatch(data["agent"]["agent_origin"], batch=data['agent']['batch'])
        agent_R_list = unbatch(data["agent"]["agent_R"], batch=data['agent']['batch'])
        score_agent_mask_list = unbatch(data["agent"]["track_category"] >= 2, batch=data['agent']['batch'])
        
        probability = prediction["pi"]
        
        global_origin_list = data["origin"]
        global_R_list = data["ro_mat"]


        if not normalized_probability:
            probability = torch.softmax(probability.double(), dim=-1)

        probability = probability.cpu().numpy()

        for batch_idx in range(B):
            scored_mask = score_agent_mask_list[batch_idx]
            (scene_id, scored_track_ids, scored_traj_local) = (scenario_ids_list[batch_idx], track_ids_list[batch_idx][scored_mask.cpu()], pred_traj_local_list[batch_idx][scored_mask])
            agent_origin, agent_R = agent_origin_list[batch_idx][scored_mask], agent_R_list[batch_idx][scored_mask]
            global_origin, global_R = global_origin_list[batch_idx], global_R_list[batch_idx]
            scored_traj_focal_frame = ((scored_traj_local@agent_R.unsqueeze(1).mT) + agent_origin.unsqueeze(1).unsqueeze(2)).cpu().numpy()
            scored_traj_global = scored_traj_focal_frame@global_R.T + global_origin
            scenario_predictions = {
                track_id: trajectory
                for track_id, trajectory in zip(scored_track_ids, scored_traj_global)
            }

            self.challenge_submission.predictions[scene_id] = (
                probability[batch_idx],
                scenario_predictions,
            )

    def generate_submission_file(self):
        print(
            "generating submission file for argoverse 2.0 motion forecasting challenge"
        )
        self.challenge_submission.to_parquet(self.submission_file)
        print(f"file saved to {self.submission_file}")
