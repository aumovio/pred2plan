"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, OneCycleLR

from predictor.models.base_model.base_model import BaseModel
from .wayformer_utils import PerceiverEncoder, PerceiverDecoder, TrainableQueryProvider
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, OneCycleLR
from predictor.models.uncertainty import HeteroscedasticGaussianProcess, RandomFeatureGaussianProcess, spectral_norm_bound
from torch.amp import autocast

def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Wayformer(BaseModel):
    '''
    Wayformer Class.
    '''

    def __init__(self, config):
        super(Wayformer, self).__init__(config)
        self.config = config
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.learning_rate = self.config.trainer["learning_rate"]

        # model-specific configurations
        self.map_attr = self.config.model['num_map_feature']
        self.k_attr = self.config.model['num_agent_feature']
        self.d_k = self.config.model['hidden_size']
        self.c = self.config.model['num_modes']
        self.L_enc = self.config.model['num_encoder_layers']
        self.num_heads = self.config.model['tx_num_heads']
        self.L_dec = self.config.model['num_decoder_layers']
        self.num_queries_enc = self.config.model['num_queries_enc']
        self.num_queries_dec = self.config.model['num_queries_dec']

        # data- and training-dependent configurations
        self.learning_rate = self.config.trainer['learning_rate']
        self.epsilon = self.config.trainer["epsilon"]
        self.past_T = self.config.data['past_len']
        self.T = self.config.data['future_len']
        self.max_points_per_lane = self.config.data['max_points_per_lane']
        self._M = self.config.data['max_num_agents']  # num agents without the ego-agent
        self.max_num_roads = self.config.data['max_num_roads']


        self.road_pts_lin = nn.Sequential(init_(nn.Linear(self.map_attr, self.d_k)))
        # INPUT ENCODERS
        self.agents_dynamic_encoder = nn.Sequential(init_(nn.Linear(self.k_attr, self.d_k)))
        self.perceiver_encoder = PerceiverEncoder(self.num_queries_enc, self.d_k,
                                                  num_self_attention_blocks = self.L_enc,
                                                  num_cross_attention_qk_channels=self.d_k,
                                                  num_cross_attention_v_channels=self.d_k,
                                                  num_self_attention_qk_channels=self.d_k,
                                                  num_self_attention_v_channels=self.d_k)

        output_query_provider = TrainableQueryProvider(
            num_queries=self.num_queries_dec,
            num_query_channels=self.d_k,
            init_scale=0.1,
        )

        self.agents_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, 1, (self._M + 1), self.d_k)),
            requires_grad=True
        )

        self.temporal_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, self.past_T, 1, self.d_k)),
            requires_grad=True
        )

        # self.map_positional_embedding = nn.parameter.Parameter(
        #     torch.zeros((1, self.max_points_per_lane * self.max_num_roads, self.d_k)), requires_grad=True
        # )

        self.perceiver_decoder = PerceiverDecoder(output_query_provider, 
                                                  self.d_k, 
                                                  num_cross_attention_layers=self.L_dec,
                                                  num_cross_attention_heads=self.num_heads,
                                                )

        if self.uncertainty:
            self.prob_predictor = HeteroscedasticGaussianProcess(
                num_features = self.c * self.d_k,
                num_random_features = self.config.model.gp_prob_num_rffs,
                num_factors = self.c,
                num_classes = self.c,
                test_mc_samples = self.config.model.gp_test_mc_samples,
                train_mc_samples = self.config.model.gp_train_mc_samples,
                likelihood = "softmax",
            )
            # self.output_model = RandomFeatureGaussianProcess(
            #     num_features = self.d_k,
            #     num_random_features = self.config.model.gp_traj_num_rffs,
            #     num_classes = 5 * self.T,
            #     num_mc_samples = self.config.model.gp_test_mc_samples,
            #     likelihood="softmax",
            # )
        else:
            self.prob_predictor = nn.Sequential(init_(nn.Linear(self.d_k, 1)))
            # self.output_model = nn.Sequential(init_(nn.Linear(self.d_k, 5 * self.T)))

        self.output_model = nn.Sequential(init_(nn.Linear(self.d_k, 5 * self.T)))
        self.selu = nn.SELU(inplace=True)

        self.criterion = Criterion(self.config)

        if config.model.get("spectral_norm", None) is not None:
            for submodule in self.modules():
                if isinstance(submodule, torch.nn.Linear):
                    spectral_norm_bound(
                        submodule,
                        name="weight",
                        n_power_iterations=1,
                        bound=config.model["spectral_norm"],
                    )

    def process_observations(self, ego, agents):
        '''
        :param observations: (B, T, N+2, A+1) where N+2 is [ego, other_agents, env]
        :return: a tensor of only the agent dynamic states, active_agent masks and env masks.
        '''
        # ego stuff
        ego_tensor = ego[:, :, :self.k_attr]
        env_masks_orig = ego[:, :, -1]
        env_masks = (1.0 - env_masks_orig).to(torch.bool)
        env_masks = env_masks.unsqueeze(1).repeat(1, self.num_queries_dec, 1).view(ego.shape[0] * self.num_queries_dec,
                                                                                   -1)

        # Agents stuff
        temp_masks = torch.cat((torch.ones_like(env_masks_orig.unsqueeze(-1)), agents[:, :, :, -1]), dim=-1)
        opps_masks = (1.0 - temp_masks).to(torch.bool)  # only for agents.
        opps_tensor = agents[:, :, :, :self.k_attr]  # only opponent states

        return ego_tensor, opps_tensor, opps_masks, env_masks

    def _forward(self, inputs):
        '''
        :param ego_in: [B, T_obs, k_attr+1] with last values being the existence mask.
        :param agents_in: [B, T_obs, M-1, k_attr+1] with last values being the existence mask.
        :param roads: [B, S, P, map_attr+1] representing the road network if self.use_map_lanes or
                      [B, 3, 128, 128] image representing the road network if self.use_map_img or
                      [B, 1, 1] if self.use_map_lanes and self.use_map_img are False.
        :return:
            pred_obs: shape [c, T, B, 5] c trajectories for the ego agents with every point being the params of
                                        Bivariate Gaussian distribution.
            mode_probs: shape [B, c] mode probability predictions P(z|X_{1:T_obs})
        '''
        ego_in, agents_in, roads = inputs['ego_in'], inputs['agents_in'], inputs['roads']

        B = ego_in.size(0)
        num_agents = agents_in.shape[2] + 1
        # Encode all input observations (k_attr --> d_k)
        ego_tensor, _agents_tensor, opps_masks_agents, env_masks = self.process_observations(ego_in, agents_in)

        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), _agents_tensor), dim=2)

        agents_emb = self.selu(self.agents_dynamic_encoder(agents_tensor))
        agents_emb = (agents_emb + self.agents_positional_embedding[:, :,
                                   :num_agents] + self.temporal_positional_embedding).view(B, -1, self.d_k)
        road_pts_feats = self.selu(self.road_pts_lin(roads[:, :self.max_num_roads, :, :self.map_attr]).view(B, -1,
                                                                                                            self.d_k))# + self.map_positional_embedding
        mixed_input_features = torch.concat([agents_emb, road_pts_feats], dim=1)
        opps_masks_roads = (1.0 - roads[:, :self.max_num_roads, :, -1]).to(torch.bool)
        mixed_input_masks = torch.concat([opps_masks_agents.view(B, -1), opps_masks_roads.view(B, -1)], dim=1)
        # Process through Wayformer's encoder
        context = self.perceiver_encoder(mixed_input_features, mixed_input_masks)

        # Wayformer-Ego Decoding
        # e.g. c = 6, d_k = 256
        out_seq = self.perceiver_decoder(context) # [B, 64, d_k]
        # Mode prediction
        # output = {}
        # if not self.uncertainty:
        #     out_dists = self.output_model(out_seq[:, :self.c]).reshape(B, self.c, self.T, -1) # [B, c, 300] reshaped into [B, c, T, 5] 
        #     mode_probs = self.prob_predictor(out_seq[:, :self.c]).reshape(B, self.c) # [B, c, 1] into [B, c]
        #     mode_log_probs = F.log_softmax(mode_probs, dim=-1)
        #     mode_probs = F.softmax(mode_probs, dim=-1)
        # else:
        #     with autocast(device_type="cuda" if out_seq.is_cuda else "cpu", enabled=False):
        #         out_dists, ep_var = self.output_model(out_seq[:, :self.c].reshape(B*self.c, self.d_k).to(torch.float32))
        #         out_dists = out_dists.reshape(B, self.c, self.T, -1).to(out_seq.dtype)
        #         ep_var = ep_var.reshape(B, self.c, self.T, -1).to(out_seq.dtype)
        #         combined_mode_params_emb = out_seq[:,:self.c].reshape(B, -1)
        #         logits, pred_mean, pred_variance, gp_pred_variance = self.prob_predictor(combined_mode_params_emb.to(torch.float32))
        #         mode_probs = pred_mean.to(out_seq.dtype)
        #         mode_log_probs = logits.to(out_seq.dtype)
        #         output['predictive_uncertainty'] = pred_variance.to(out_seq.dtype)
        #         output['epistemic_uncertainty'] = gp_pred_variance.to(out_seq.dtype) 
        
        output = {}
        out_dists = self.output_model(out_seq[:, :self.c]).reshape(B, self.c, self.T, -1) # [B, c, 300] reshaped into [B, c, T, 5] 
        if not self.uncertainty:
            mode_probs = self.prob_predictor(out_seq[:, :self.c]).reshape(B, self.c) # [B, c, 1] into [B, c]
            mode_log_probs = F.log_softmax(mode_probs, dim=-1)
            mode_probs = F.softmax(mode_probs, dim=-1)
        else:
            with autocast(device_type="cuda" if out_seq.is_cuda else "cpu", enabled=False):
                combined_mode_params_emb = out_seq[:,:self.c].reshape(B, -1)
                logits, pred_mean, pred_variance, gp_pred_variance = self.prob_predictor(combined_mode_params_emb.to(torch.float32))
                mode_probs = pred_mean.to(out_seq.dtype)
                mode_log_probs = logits.to(out_seq.dtype)
                output['predictive_uncertainty'] = pred_variance.to(out_seq.dtype)
                output['epistemic_uncertainty'] = gp_pred_variance.to(out_seq.dtype) 
        # return  [c, T, B, 5], [B, c]
        output['predicted_log_probability'] = mode_log_probs
        output['predicted_probability'] = mode_probs  # #[B, c]
        output['predicted_trajectory'] = out_dists  # [B, c, T, 5] to be able to parallelize code
        if not self.training:
            output['agent_embedding'] = agents_emb#out_seq[:, :self.num_queries_dec].reshape(B, -1)
            output['map_embedding'] = road_pts_feats
            output['scene_context'] = context
            output['scene_decoding'] = out_seq[:, :self.num_queries_dec].reshape(B, -1)
        # if len(np.argwhere(np.isnan(out_dists.detach().cpu().to(torch.float32).numpy()))) > 1:
        #     breakpoint()
        return output

    def forward(self, batch):
        model_input = {}
        inputs = batch['input_dict']
        agents_in, agents_mask, roads = inputs['obj_trajs'], inputs['obj_trajs_mask'], inputs['map_polylines']
        ego_in = torch.gather(agents_in, 1, inputs['track_index_to_predict'].view(-1, 1, 1, 1).repeat(1, 1,
                                                                                                      *agents_in.shape[
                                                                                                       -2:])).squeeze(1)
        ego_mask = torch.gather(agents_mask, 1, inputs['track_index_to_predict'].view(-1, 1, 1).repeat(1, 1,
                                                                                                       agents_mask.shape[
                                                                                                           -1])).squeeze(
            1)
        agents_in = torch.cat([agents_in, agents_mask.unsqueeze(-1)], dim=-1)
        agents_in = agents_in.transpose(1, 2)
        ego_in = torch.cat([ego_in, ego_mask.unsqueeze(-1)], dim=-1)
        roads = torch.cat([inputs['map_polylines'], inputs['map_polylines_mask'].unsqueeze(-1)], dim=-1)
        model_input['ego_in'] = ego_in
        model_input['agents_in'] = agents_in
        model_input['roads'] = roads
        output = self._forward(model_input)

        ground_truth = torch.cat([inputs['center_gt_trajs'][..., :2], inputs['center_gt_trajs_mask'].unsqueeze(-1)],
                                 dim=-1)
        loss = self.criterion(output, ground_truth, inputs['center_gt_final_valid_idx'])
        # output['dataset_name'] = inputs['dataset_name']
        #output['predicted_probability'] = F.softmax(output['predicted_probability'], dim=-1)
        return output, loss


    def configure_optimizers(self):
        # First, create parameter groups:
        prob_predictor_params = []
        other_params = []
        factor = 1

        for name, param in self.named_parameters():
            if "prob_predictor" in name:
                prob_predictor_params.append(param)
            else:
                other_params.append(param)

        # Then configure the optimizer with different groups:
        if self.config.trainer.optimizer == "AdamW":
            optimizer = AdamW(
                [
                    {"params": other_params, "lr": self.learning_rate},
                    {"params": prob_predictor_params, "lr": self.learning_rate * factor},  # smaller LR for prob_predictor
                ],
                eps=self.config.trainer["epsilon"],
                weight_decay=self.config.trainer["weight_decay"],
            )
        elif self.config.trainer.optimizer == "SGD":
            optimizer = SGD(
                [
                    {"params": other_params, "lr": self.learning_rate},
                    {"params": prob_predictor_params, "lr": self.learning_rate * factor},
                ],
                weight_decay=self.config.trainer["weight_decay"],
            )

        # Now the OneCycleLR
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[self.learning_rate, self.learning_rate * factor],  # Important: one max_lr per param group
            total_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }


class Criterion(nn.Module):
    def __init__(self, config):
        super(Criterion, self).__init__()
        self.config = config
        pass

    def forward(self, out, gt, center_gt_final_valid_idx):
        return self.nll_loss_gmm_direct(out['predicted_log_probability'], out['predicted_trajectory'], gt,
                                        center_gt_final_valid_idx)

    def nll_loss_gmm_direct(self, pred_scores, pred_trajs, gt_trajs, center_gt_final_valid_idx,
                            pre_nearest_mode_idxs=None,
                            timestamp_loss_weight=None, use_square_gmm=False, log_std_range=(-1.609, 5.0),
                            rho_limit=0.5):
        """
        GMM Loss for Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
        Written by Shaoshuai Shi

        Negative Log-Likelihood (NLL) loss for a multi-modal trajectory prediction,
        plus a classification loss for mode selection.

        Args:
            pred_scores (batch_size, num_modes):
            pred_trajs (batch_size, num_modes, num_timestamps, 5 or 3)
            gt_trajs (batch_size, num_timestamps, 3):
            timestamp_loss_weight (num_timestamps):
        """
        # Check that the last dimension matches our assumption of 5 or 3 parameters
        if use_square_gmm:
            # Square GMM should have exactly 3 parameters: [x_mean, y_mean, log_std]
            assert pred_trajs.shape[-1] == 3, (
                f"Expected pred_trajs to have last dimension 3, got {pred_trajs.shape[-1]}"
            )
        else:
            # Full GMM has 5 parameters: [x_mean, y_mean, log_std_x, log_std_y, rho]
            assert pred_trajs.shape[-1] == 5, (
                f"Expected pred_trajs to have last dimension 5, got {pred_trajs.shape[-1]}"
            )
            
        batch_size = pred_trajs.shape[0]

        # The last channel in gt_trajs usually indicates validity (1 or 0)
        gt_valid_mask = gt_trajs[..., -1] # shape: (batch_size, num_timestamps)

        # Step 1: Find the "best" mode per sample if not already provided
        if pre_nearest_mode_idxs is not None:
            nearest_mode_idxs = pre_nearest_mode_idxs
        else:
            # Compute Euclidean distance between predicted means and ground truth, 
            # then pick the mode with the smallest distance across all timesteps.
            distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :2]).norm(dim=-1)
            distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1) # Zero out invalid timesteps and sum across remaining
            nearest_mode_idxs = distance.argmin(dim=-1)  # shape: (batch_size,)

        # Build a batch index for advanced indexing
        nearest_mode_bs_idxs = torch.arange(batch_size).type_as(nearest_mode_idxs)  # (batch_size, 2) # TODO: check?

        # Gather the GMM parameters for the chosen (nearest) mode in each sample
        nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]  # (batch_size, num_timestamps, 5) # TODO: check?

        # Residual between ground-truth (x, y) and predicted (x, y)
        res_trajs = gt_trajs[..., :2] - nearest_trajs[:, :, 0:2]  # (batch_size, num_timestamps, 2)
        dx = res_trajs[:, :, 0]
        dy = res_trajs[:, :, 1]

        # Step 2: Extract or construct Gaussian parameters: std_x, std_y, rho
        if use_square_gmm:
            # Single log_std for both x and y (isotropic), no correlation
            log_std1 = log_std2 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
            std1 = std2 = torch.exp(log_std1)  # (0.2m to 150m)
            rho = torch.zeros_like(log_std1)
        else:
            # log_std_x, log_std_y, and rho are predicted separately
            log_std1 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
            log_std2 = torch.clip(nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
            std1 = torch.exp(log_std1)  # (0.2m to 150m)
            std2 = torch.exp(log_std2)  # (0.2m to 150m)
            rho = torch.clip(nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)

        # Convert to float if needed
        gt_valid_mask = gt_valid_mask.type_as(pred_scores)

        # If we have per-timestep weights, multiply them into the valid mask
        if timestamp_loss_weight is not None:
            # broadcast to match batch size
            gt_valid_mask = gt_valid_mask * timestamp_loss_weight[None, :]

        # Step 3: Compute NLL for 2D Gaussian
        # -log(a^-1 * e^b) = log(a) - b
        reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho ** 2)  # (batch_size, num_timestamps)

        # 0.5 / (1 - rho^2) * [ (dx^2 / sigma_x^2) + (dy^2 / sigma_y^2) - 2*rho (dx dy / (sigma_x sigma_y)) ]
        reg_gmm_exp = (0.5 * 1 / (1 - rho ** 2)) * (
                (dx ** 2) / (std1 ** 2) + (dy ** 2) / (std2 ** 2) - 2 * rho * dx * dy / (
                std1 * std2))  # (batch_size, num_timestamps)

        # Multiply by valid mask and sum across timesteps
        reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask).sum(dim=-1)

        # Step 4: Classification loss: cross-entropy over modes
        loss_cls = (F.nll_loss(input=pred_scores, target=nearest_mode_idxs, reduction='none'))
        # Combine regression and classification, then average over the batch
        return (reg_loss + loss_cls).mean()





