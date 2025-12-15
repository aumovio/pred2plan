"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

from functools import partial
from enum import auto, Enum
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules import Module
from torch.nn.utils import parametrize

from torch.nn.utils.parametrizations import _SpectralNorm

MIN_SCALE_MONTE_CARLO = 1e-3

class HeteroscedasticGaussianProcess(Module):
    """
    A softmax layer that hierachically models 
    - epistemic uncertainty via a random Fourier feature gaussian process layer
    - aleatoric uncertainty via low-rank approximated heteroscedastic noise layer
    """
    def __init__(
            self,
            num_features: int, # dimension of input
            num_random_features: int, # dimension of gp
            num_factors: int, # dimension of low rank approximation factor
            num_classes: int, # dimension of output
            train_mc_samples: int = 1, # >0 get mc sampled predictions instead of mean predictions
            test_mc_samples: int = 32, # >0 get mc sampled predictions instead of mean predictions
            gp_output_bias: float = 0, # prior bias
            gp_random_feature_type: str = "orf", # orf might approximate better with less features
            gp_cov_momentum: float = -1, # >0 for moving average computation
            gp_cov_ridge_penalty: float = 1.0, # (ridge_penalty * I) prior variance
            likelihood: str = "gaussian", # likelihood
            use_input_normalized_gp: bool = True, # automatic relevance detection
            #gp_kernel_scale: float | None, # not necessary if we just use input_normalization
            het_var_weight: float = 1.0, # weighting of heteroscedastic marginal variances in mc sampling
            sngp_var_weight: float = 1.0, # weighting of gp marginal variances in mc sampling
            share_samples_across_batch: bool = False, # wether to draw one mc sample for whole batch
            mc_temperature: float = 1.0, # temperature in mc sampling
            epsilon: float = 1e-6, # for numerical stability
    ) -> None:
        
        super().__init__()
        self._num_classes = num_classes
        self._train_mc_samples = train_mc_samples
        self._test_mc_samples = test_mc_samples
        self.het_var_weight = het_var_weight
        self.sngp_var_weight = sngp_var_weight
        self._share_samples_across_batch = share_samples_across_batch
        self._num_factors = num_factors
        self._mc_temperature = mc_temperature
        self._epsilon = epsilon

        self._loc_layer = RandomFeatureGaussianProcess(
                num_features=num_features,
                num_classes=num_classes,
                num_random_features=num_random_features,
                gp_output_bias=gp_output_bias,
                gp_random_feature_type=gp_random_feature_type,
                use_input_normalized_gp=use_input_normalized_gp,
                gp_cov_momentum=gp_cov_momentum,
                gp_cov_ridge_penalty=gp_cov_ridge_penalty,
                likelihood=likelihood,
                num_mc_samples = -1
        )

        self._scale_layer = nn.Linear(num_features, num_classes * num_factors)
        self._diag_layer = nn.Linear(num_features, num_classes)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the HeteroscedasticSoftmax model.

        Args:
            inputs (torch.Tensor): Input tensor of shape [B, D].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - logits (torch.Tensor): Logits of shape [B, C].
                - log_probs (torch.Tensor): Log probabilities of shape [B, C].
                - pred_mean (torch.Tensor): Mean predictive probabilities of shape [B, C].
                - pred_variance (torch.Tensor): Predictive variance of shape [B, C].
        """
        gp_locs, gp_scale = self._loc_layer(inputs)
        factor_loadings, scale_comp = self._compute_scale_param(inputs, gp_scale)

        if self.training:
            total_mc_samples = self._train_mc_samples
        else:
            total_mc_samples = self._test_mc_samples

        pred_mean, _ = self._compute_predictive_mean(gp_locs, (factor_loadings, scale_comp), total_mc_samples)
        pred_variance = self._compute_predictive_variance(gp_locs, gp_locs, (factor_loadings, scale_comp), total_mc_samples)
        logits = torch.log(torch.clip(pred_mean, self._epsilon, 1.0))
        gp_pred_variance = gp_scale
        # logits = torch.log(pred_mean)
        return logits, pred_mean, pred_variance, gp_pred_variance


    def compute_loc_param(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Computes the location (logits) parameter using the random feature GP layer.

        Args:
            inputs (torch.Tensor): Input tensor of shape [B, D].

        Returns:
            torch.Tensor: Location parameters (logits) of shape [B, C] during training,
            or [B, S, C] if not training (with S = number of MC samples).
        """
        return self._loc_layer(inputs)
    
    def _compute_scale_param(self, inputs: torch.Tensor, gp_scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the low-rank factor loading and diagonal scale parameters based on
        the input features and covariance matrix.

        Args:
            inputs (torch.Tensor): Input tensor of shape [B, D].
            covmat (torch.Tensor): Covariance matrix of shape [..., D, D] (depending on GP outputs).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - factor_loading (torch.Tensor): Factor loadings of shape [B, C * F].
                - diag_scale_comp (torch.Tensor): Diagonal scale component of shape [B, C].
        """
        factor_loadings = self._scale_layer(inputs)
        diag_factor = F.softplus(self._diag_layer(inputs)) + MIN_SCALE_MONTE_CARLO

        if self.training:
            diag_scale_comp = diag_factor
        else:
            diag_scale_comp = torch.sqrt(self.het_var_weight * diag_factor ** 2 + self.sngp_var_weight * gp_scale)

            # B = factor_loadings.shape[0]
            # C = self._num_classes
            # F = self._num_factors

            # V = factor_loadings.view(B, C, F)
            # diag_outer_term = (V**2).sum(dim=-1)
            # cov_diagonal = diag_outer_term + diag_factor ** 2
        return factor_loadings, diag_scale_comp
        
    def _compute_predictive_mean(self, locs: Tensor, scale: tuple[Tensor, Tensor], total_mc_samples: int, seed: int = None) -> tuple[Tensor, Tensor]:
        """
        Computes the estimated predictive mean using Monte Carlo samples.

        Args:
            locs (Tensor): Logits of shape [B, C].
            scale (tuple[Tensor, Tensor]):
                - Factor loadings of shape [B, C * F].
                - Diagonal elements of shape [B, C].
            total_mc_samples (int): Number of Monte Carlo samples.
            seed (int): Random seed for reproducibility.

        Returns:
            tuple[Tensor, Tensor]:
                - Predictive mean of shape [B, C].
                - Raw Monte Carlo samples of shape [B, S, C].
        """
        samples = self._compute_mc_samples(locs, scale, total_mc_samples, seed)  # [B, S, C]
        return samples.mean(dim=1), samples

    def _compute_predictive_variance(self, mean: Tensor, locs: Tensor, scale: tuple[Tensor, Tensor], num_samples: int, seed: int = None) -> Tensor:
        """
        Computes the per-class predictive variance.

        Args:
            mean (Tensor): Predictive mean of shape [B, C].
            locs (Tensor): Logits of shape [B, C].
            scale (tuple[Tensor, Tensor]):
                - Factor loadings of shape [B, C * F].
                - Diagonal elements of shape [B, C].
            seed (int): Random seed for reproducibility.
            num_samples (int): Number of Monte Carlo samples.

        Returns:
            Tensor: Predictive variance of shape [B, C].
        """
        mean = mean.unsqueeze(dim=1)  # [B, 1, C]
        mc_samples = self._compute_mc_samples(locs, scale, num_samples, seed)  # [B, S, C]
        total_variance = ((mc_samples - mean) ** 2).mean(dim=1)  # [B, C]
        return total_variance
    
    def _compute_mc_samples(self, locs: Tensor, scale: tuple[Tensor, Tensor], num_samples: int, seed: int = None) -> Tensor:
        """
        Computes Monte Carlo samples of logits and applies softmax/sigmoid.

        Args:
            locs (Tensor): Logits of shape [B, C].
            scale (tuple[Tensor, Tensor]):
                - Factor loadings of shape [B, C * F].
                - Diagonal elements of shape [B, C].
            num_samples (int): Number of Monte Carlo samples.
            seed (int): Random seed for reproducibility.

        Returns:
            Tensor: Sampled probabilities of shape [B, S, C].
        """
        locs = locs.unsqueeze(1)  # [B, 1, C]
        noise_samples = self._compute_noise_samples(scale, num_samples, seed)  # [B, S, C]
        latents = locs + noise_samples  # [B, S, C]
        temperature = self._mc_temperature
        

        return F.softmax(latents / temperature, dim=-1)


    def _compute_diagonal_noise_samples(self, diag_scale: Tensor, num_samples: int, seed: int = None) -> Tensor:
        """
        Computes samples of the diagonal logit noise.

        Args:
            diag_scale (Tensor): Diagonal scale elements of shape [B, C].
            num_samples (int): Number of Monte Carlo samples.
            seed (int): Random seed for reproducibility.

        Returns:
            Tensor: Logit noise samples of shape [B, S, C].
        """
        num_noise_samples = 1 if self._share_samples_across_batch else diag_scale.shape[0]

        dist = torch.distributions.Normal(
            loc=torch.zeros(num_noise_samples, self._num_classes, dtype=diag_scale.dtype, device=diag_scale.device),
            scale=torch.ones(num_noise_samples, self._num_classes, dtype=diag_scale.dtype, device=diag_scale.device)
        )
        
        diag_noise_samples = dist.sample((num_samples,)).permute(1, 0, 2)  # [B, S, C]
        return diag_noise_samples * diag_scale.unsqueeze(1)

    def _compute_standard_normal_samples(self, factor_loadings: Tensor, num_samples: int, seed: int = None) -> Tensor:
        """
        Computes samples from a standard normal distribution.

        Args:
            factor_loadings (Tensor): Factor loadings of shape [B, C * F].
            num_samples (int): Number of Monte Carlo samples.
            seed (int): Random seed for reproducibility.

        Returns:
            Tensor: Samples of shape [B, S, F].
        """
        num_noise_samples = 1 if self._share_samples_across_batch else factor_loadings.shape[0]

        dist = torch.distributions.Normal(
            loc=torch.zeros(num_noise_samples, self._num_factors, dtype=factor_loadings.dtype, device=factor_loadings.device),
            scale=torch.ones(num_noise_samples, self._num_factors, dtype=factor_loadings.dtype, device=factor_loadings.device)
        )
        
        standard_normal_samples = dist.sample((num_samples,)).permute(1, 0, 2)  # [B, S, F]
        
        if self._share_samples_across_batch:
            standard_normal_samples = standard_normal_samples.repeat(factor_loadings.shape[0], 1, 1)
        
        return standard_normal_samples

    def _compute_noise_samples(self, scale: tuple[Tensor, Tensor], num_samples: int, seed: int = None) -> Tensor:
        """
        Computes samples of the logit noise.

        Args:
            scale (tuple[Tensor, Tensor]):
                - Factor loadings of shape [B, C * F].
                - Diagonal elements of shape [B, C].
            num_samples (int): Number of Monte Carlo samples.
            seed (int): Random seed for reproducibility.

        Returns:
            Tensor: Logit noise samples of shape [B, S, C].
        """
        factor_loadings, diag_scale = scale
        
        diag_noise_samples = self._compute_diagonal_noise_samples(diag_scale, num_samples, seed)  # [B, S, C]
        standard_normal_samples = self._compute_standard_normal_samples(factor_loadings, num_samples, seed)  # [B, S, F]
        
        factor_loadings = factor_loadings.view(-1, self._num_classes, self._num_factors)
        res = torch.einsum('ijk,iak->iaj', factor_loadings, standard_normal_samples)  # [B, S, C]
        
        return res + diag_noise_samples

    def reset_precision_matrix(self) -> None:
        self._loc_layer.reset_precision_matrix()


class RandomFeatureGaussianProcess(Module):
    """Random feature GP output layer.

    This layer implements a Gaussian Process output using random Fourier features.

    Args:
        num_features: Number of input features.
        num_classes: Number of output classes.
        num_mc_samples: Number of Monte Carlo samples.
        num_random_features: Number of random features to use.
        gp_output_bias: Output bias for the GP.
        gp_random_feature_type: Type of random features ('orf' or 'rff').
        use_input_normalized_gp: Whether to use input normalization for GP.
        gp_cov_momentum: Momentum for covariance update.
        gp_cov_ridge_penalty: Ridge penalty for covariance.
        likelihood: Likelihood type ('gaussian' or 'softmax').
    """

    def __init__(
        self,
        num_features: int, # dimension of input
        num_random_features: int, # dimension of gp
        num_classes: int, # dimension of output
        num_mc_samples: int = -1, # >0 get mc sampled predictions instead of mean predictions
        gp_output_bias: float = 0, # prior bias
        gp_random_feature_type: str = "orf", # orf might approximate better with less features
        gp_cov_momentum: float = -1, # >0 for moving average computation
        gp_cov_ridge_penalty: float = 1.0, # (ridge_penalty * I) prior variance
        likelihood: str = "gaussian",
        use_input_normalized_gp: bool = True,
        #gp_kernel_scale: float | None, # not necessary if we just use input_normalization
    ) -> None:
        super().__init__()
        self._num_classes = num_classes
        self._num_random_features = num_random_features
        self._num_mc_samples = num_mc_samples

        self._use_input_normalized_gp = use_input_normalized_gp

        # self._gp_input_scale = (
        #     1 / gp_kernel_scale**0.5 if gp_kernel_scale is not None else None
        # )
        # self._gp_kernel_scale = gp_kernel_scale
        self._gp_output_bias = gp_output_bias
        self._likelihood = likelihood

        if gp_random_feature_type == "orf":
            self._random_features_weight_initializer = partial(
                self._orthogonal_random_features_initializer, std=1.0
            )
        elif gp_random_feature_type == "rff":
            self._random_features_weight_initializer = partial(
                nn.init.normal_, mean=0.0, std=1.0
            )
        else:
            msg = (
                "gp_random_feature_type must be one of 'orf' or 'rff', got "
                f"{gp_random_feature_type}"
            )
            raise ValueError(msg)

        self._gp_cov_momentum = gp_cov_momentum
        self._gp_cov_ridge_penalty = gp_cov_ridge_penalty

        # Default to Gaussian RBF kernel with orthogonal random features.
        self._random_features_bias_initializer = partial(
            nn.init.uniform_, a=0, b=2 * torch.pi
        )

        if self._use_input_normalized_gp:
            self._input_norm_layer = nn.LayerNorm(num_features)

        self._random_feature_layer = self._make_random_feature_layer(num_features)

        num_cov_layers = 1 if self._likelihood == "gaussian" else num_classes
        self._gp_cov_layers = nn.ModuleList(
            LaplaceRandomFeatureCovariance(
                gp_feature_dim=self._num_random_features,
                momentum=self._gp_cov_momentum,
                ridge_penalty=self._gp_cov_ridge_penalty,
            )
            for _ in range(num_cov_layers)
        )

        self._gp_output_layer = nn.Linear(
            in_features=self._num_random_features,
            out_features=self._num_classes,
            bias=False,
        )

        self._gp_output_bias = nn.Parameter(
            torch.tensor([self._gp_output_bias] * self._num_classes),
            requires_grad=False,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Performs forward pass of the GP output layer.

        Args:
            gp_inputs: Input features.

        Returns:
            Output logits or samples.
        """
        # Computes random features.
        if self._use_input_normalized_gp:
            inputs = self._input_norm_layer(inputs)
        elif self._gp_input_scale is not None:
            # Supports lengthscale for custom random feature layer by directly
            # rescaling the input.
            inputs *= self._gp_input_scale

        gp_features = self._random_feature_layer(inputs).cos() # [B, D]

        # Computes predictive/posterior mean (i.e., MAP estimate)
        gp_pred_mean = self._gp_output_layer(gp_features) + self._gp_output_bias  # [B, C]

        if self._likelihood == "gaussian":
            multipliers = torch.ones_like(gp_pred_mean)
        elif self._likelihood == "binary": # if we use this for softmax we ignore off-diagonals -> numerically unstable!
            prob = F.softmax(gp_pred_mean, dim=-1)
            multipliers = prob * (1 - prob)
        elif self._likelihood == "softmax": # approximation as used in edward2, works empirically well
            # Note: precise multiplier computation scales quadratically with number of classes and complicates covariance updates, maybe something for later
            multipliers = torch.ones_like(gp_pred_mean) 

        with torch.no_grad():
            gp_pred_var = torch.zeros_like(gp_pred_mean)
            for i, cov_layer in enumerate(self._gp_cov_layers):
                gp_pred_var[:, i] = cov_layer(gp_features, multipliers[:,i])

        if self._num_mc_samples>0 and not self.training:
            return self.monte_carlo_sample_logits(logits=gp_pred_mean, vars=gp_pred_var, num_samples=self._num_mc_samples).mean(dim=1), gp_pred_var
        else:
            return gp_pred_mean, gp_pred_var

        # if self.training: # Updates posterior variance at training time
        #     if self._likelihood == "gaussian":
        #         self._gp_cov_layers[0].update(gp_features)
        #     else:  # self._likelihood == "softmax"
        #         prob = gp_outputs.softmax(dim=-1)
        #         multipliers = prob * (1 - prob)

        #         for cov_layer, multiplier in zip(
        #             self._gp_cov_layers, multipliers.T, strict=True
        #         ):
        #             cov_layer.update(gp_features, multiplier)

        #     return gp_outputs  # [B, C], [B, C]
            
        # else: # Computes predictive variance at test time
        #     if self._likelihood == "gaussian":
        #         gp_vars = (
        #             self._gp_cov_layers[0](gp_features)
        #             .unsqueeze(1)
        #             .repeat(1, gp_outputs.shape[-1])
        #         )
        #     else:
        #         with torch.no_grad():
        #             gp_vars = torch.zeros_like(gp_outputs)
        #             for i, cov_layer in enumerate(self._gp_cov_layers):
        #                 gp_vars[:, i] = cov_layer(gp_features)

        #     return self.monte_carlo_sample_logits(
        #         logits=gp_outputs, vars=gp_vars, num_samples=self._num_mc_samples
        #     )  # [B, S, C]

    def reset_precision_matrix(self) -> None:
        """Resets covariance matrix of the GP layer."""
        for cov_layer in self._gp_cov_layers:
            cov_layer.reset_precision_matrix()

    @staticmethod
    def mean_field_logits(
        logits: Tensor, vars: Tensor, mean_field_factor: float
    ) -> Tensor:
        """Computes mean-field logits.

        Args:
            logits: Input logits.
            vars: Variances.
            mean_field_factor: Mean field factor.

        Returns:
            Mean-field logits.
        """
        # Compute scaling coefficient for mean-field approximation.
        logits_scale = (1 + vars * mean_field_factor).sqrt()

        # Cast logits_scale to compatible dimension.
        logits_scale = logits_scale.reshape(-1, 1)

        return logits / logits_scale

    @staticmethod
    def monte_carlo_sample_logits(
        logits: Tensor, vars: Tensor, num_samples: int
    ) -> Tensor:
        """Performs Monte Carlo sampling of logits.

        Args:
            logits: Input logits.
            vars: Variances.
            num_samples: Number of samples.

        Returns:
            Sampled logits.
        """
        batch_size, num_classes = logits.shape
        vars = vars.unsqueeze(dim=1)  # [B, 1, C]

        std_normal_samples = torch.randn(
            batch_size, num_samples, num_classes, device=logits.device
        )  # [B, S, C]

        return vars.sqrt() * std_normal_samples + logits.unsqueeze(dim=1)

    @staticmethod
    def _orthogonal_random_features_initializer(tensor: Tensor, std: float) -> Tensor:
        """Initializes orthogonal random features.

        Args:
            tensor: Tensor to initialize.
            std: Standard deviation.

        Returns:
            Initialized tensor.
        """
        num_rows, num_cols = tensor.shape
        if num_rows < num_cols:
            # When num_rows < num_cols, sample multiple (num_rows, num_rows) matrices
            # and then concatenate.
            ortho_mat_list = []
            num_cols_sampled = 0

            while num_cols_sampled < num_cols:
                matrix = torch.empty_like(tensor[:, :num_rows])
                ortho_mat_square = nn.init.orthogonal_(matrix, gain=std)
                ortho_mat_list.append(ortho_mat_square)
                num_cols_sampled += num_rows

            # Reshape the matrix to the target shape (num_rows, num_cols)
            ortho_mat = torch.cat(ortho_mat_list, dim=-1)
            ortho_mat = ortho_mat[:, :num_cols]
        else:
            matrix = torch.empty_like(tensor)
            ortho_mat = nn.init.orthogonal_(matrix, gain=std)

        # Sample random feature norms.
        # Construct Monte-Carlo estimate of squared column norm of a random
        # Gaussian matrix.
        feature_norms_square = torch.randn_like(ortho_mat) ** 2
        feature_norms = feature_norms_square.sum(dim=0).sqrt()

        # Sets a random feature matrix with orthogonal columns and Gaussian-like
        # column norms.
        value = ortho_mat * feature_norms
        with torch.no_grad():
            tensor.copy_(value)

        return tensor

    def _make_random_feature_layer(self, num_features: int) -> nn.Module:
        """Creates a random feature layer.

        Args:
            num_features: Number of input features.

        Returns:
            Random feature layer.
        """
        # Use user-supplied configurations.
        custom_random_feature_layer = nn.Linear(
            in_features=num_features,
            out_features=self._num_random_features,
        )
        self._random_features_weight_initializer(custom_random_feature_layer.weight)
        self._random_features_bias_initializer(custom_random_feature_layer.bias)
        custom_random_feature_layer.weight.requires_grad_(False)
        custom_random_feature_layer.bias.requires_grad_(False)

        return custom_random_feature_layer


class LaplaceRandomFeatureCovariance(nn.Module):
    """Empirical covariance matrix for random feature GPs.

    This module computes and maintains the covariance matrix for random feature
    Gaussian Processes.

    Args:
        gp_feature_dim: Dimension of GP features.
        momentum: Momentum for updating precision matrix.
        ridge_penalty: Ridge penalty for covariance matrix.
    """

    def __init__(
        self,
        gp_feature_dim: int,
        momentum: float,
        ridge_penalty: float,
    ) -> None:
        
        super().__init__()
        self._ridge_penalty = ridge_penalty
        self._momentum = momentum

        # Posterior precision matrix for the GP's random feature coefficients
        precision_matrix = torch.eye(gp_feature_dim) 
        self.register_buffer("_precision_matrix", precision_matrix)
        cholesky_factor = torch.eye(gp_feature_dim)
        self.register_buffer("_cholesky_factor", cholesky_factor)
        self._gp_feature_dim = gp_feature_dim

        # Boolean flag to indicate whether to update the covariance matrix (i.e.,
        # by inverting the newly updated precision matrix) during inference.
        self._update_cholesky = False

    def forward(self, gp_features: Tensor, multipliers: Tensor) -> Tensor:
        """Updates precision matrix at training time or computes predictive variance at test time.

        Args:
            gp_features: [B,D], random feature vectors for each batch element
            multipliers: [B], multiplier for each batch element
        Returns:
            gp_var: [B], the predictive variance for each batch element
        """
        if self.training:
            self.update(gp_features, multipliers)

            return torch.ones_like(gp_features[:,0]) # dummy predictive variance
        else:
            # Lazily computes feature covariance matrix during inference.
            cholesky_factor_updated = self._update_cholesky_factor()

            # Store updated covariance matrix.
            self._cholesky_factor.copy_(cholesky_factor_updated)

            # Disable covariance update in future inference calls (to avoid the
            # expensive torch.linalg.inv op) unless there are new updates to precision
            # matrix.
            self._update_cholesky = False

            return self._compute_predictive_variance(gp_features)


    def reset_precision_matrix(self) -> None:
        """Resets precision matrix to its initial value."""
        gp_feature_dim = self._gp_feature_dim
        self._precision_matrix.copy_(self._ridge_penalty * torch.eye(gp_feature_dim))


    @torch.no_grad()
    def update(self, gp_features: Tensor, multiplier: float = 1) -> None:
        """Updates the feature precision matrix.

        Args:
            gp_features: GP features.
            multiplier: Multiplier for the update.
        """
        # Computes the updated feature precision matrix.
        precision_matrix_updated = self._update_feature_precision_matrix(
            gp_features=gp_features,
            multiplier=multiplier,
        )

        # Updates precision matrix.
        self._precision_matrix.copy_(precision_matrix_updated)

        # Enables covariance update in the next inference call.
        self._update_cholesky = True

    def _update_feature_precision_matrix(
        self, gp_features: Tensor, multiplier: float
    ) -> Tensor:
        """Computes the updated precision matrix of feature weights.

        Args:
            gp_features: GP features.
            multiplier: Multiplier for the update.

        Returns:
            Updated precision matrix.
        """
        batch_size = gp_features.shape[0]

        # Computes batch-specific normalized precision matrix.
        gp_features_adjusted = torch.sqrt(multiplier) * gp_features.T
        precision_matrix_minibatch = torch.matmul(gp_features_adjusted,gp_features_adjusted.T)
        #random_jitter = torch.rand(self._gp_feature_dim) * 1e-06
        #precision_matrix_minibatch += random_jitter
        # Updates the population-wise precision matrix.
        if self._momentum > 0:
            # Use moving-average updates to accumulate batch-specific precision
            # matrices.
            precision_matrix_minibatch /= batch_size
            precision_matrix_new = (
                self._momentum * self._precision_matrix
                + (1 - self._momentum) * precision_matrix_minibatch
            )
        else:
            # Compute exact population-wise covariance without momentum.
            # If use this option, make sure to pass through data only once.
            precision_matrix_new = self._precision_matrix + precision_matrix_minibatch

        #precision_matrix_new = 0.5 * (precision_matrix_new + precision_matrix_new.T) 
        return precision_matrix_new

    def _update_cholesky_factor(self) -> Tensor:
        """Computes the feature covariance matrix.

        Returns:
            Updated covariance matrix.
        """
        precision_matrix = self._precision_matrix
        cholesky_factor = self._cholesky_factor

        # Compute covariance matrix update only when `update_covariance = True`.
        if self._update_cholesky:
            cholesky_factor_updated = torch.linalg.cholesky(precision_matrix)
        else:
            cholesky_factor_updated = cholesky_factor

        return cholesky_factor_updated

    def _compute_predictive_variance(self, gp_feature: Tensor) -> Tensor:
        """Computes posterior predictive variance.

        Args:
            gp_feature: [B, D] GP features .

        Returns:
            gp_var: [B] predictive variance 
        """
        # Computes the variance of the posterior gp prediction.
        chol = self._cholesky_factor
        chol_feature = torch.linalg.solve_triangular(chol, gp_feature.T, left=True, upper=False)
        # predictive covariance (gp_feature x gp_feature)
        # gp_var = chol_feature.T @ chol_feature
        # predictive variance (diagonal elements)
        gp_var = torch.square(chol_feature).sum(axis=0)

        # gp_var = torch.einsum(
        #     "ij,jk,ik->i",
        #     gp_feature,
        #     self._covariance_matrix,
        #     gp_feature,
        # )

        return gp_var
    
class BoundedSpectralNorm(_SpectralNorm):

    def __init__(
        self,
        weight: torch.Tensor,
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
        bound: float = 1.0, # customized
    ) -> None:
        # initialize all of the parent’s buffers (u, v, reshaping, etc)
        super().__init__(weight, n_power_iterations, dim, eps)
        # now just store your extra parameter
        self.bound = bound

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 1:
            # Faster and more exact path, no need to approximate anything
            return F.normalize(weight, dim=0, eps=self.eps)
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight_mat, self.n_power_iterations)
            # See above on why we need to clone
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)
            # The proper way of computing this should be through F.bilinear, but
            # it seems to have some efficiency issues:
            # https://github.com/pytorch/pytorch/issues/58093
            sigma = torch.vdot(u, torch.mv(weight_mat, v))

            #custom part
            scale = torch.clamp(sigma / self.bound, min=1.0)
            return weight / scale

def spectral_norm_bound(
    module: Module,
    name: str = "weight",
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: Optional[int] = None,
    bound: float = 1.0, 
) -> Module:
    r"""Apply spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    When applied on a vector, it simplifies to

    .. math::
        \mathbf{x}_{SN} = \dfrac{\mathbf{x}}{\|\mathbf{x}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by reducing the Lipschitz constant
    of the model. :math:`\sigma` is approximated performing one iteration of the
    `power method`_ every time the weight is accessed. If the dimension of the
    weight tensor is greater than 2, it is reshaped to 2D in power iteration
    method to get spectral norm.


    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`power method`: https://en.wikipedia.org/wiki/Power_iteration
    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    .. note::
        This function is implemented using the parametrization functionality
        in :func:`~torch.nn.utils.parametrize.register_parametrization`. It is a
        reimplementation of :func:`torch.nn.utils.spectral_norm`.

    .. note::
        When this constraint is registered, the singular vectors associated to the largest
        singular value are estimated rather than sampled at random. These are then updated
        performing :attr:`n_power_iterations` of the `power method`_ whenever the tensor
        is accessed with the module on `training` mode.

    .. note::
        If the `_SpectralNorm` module, i.e., `module.parametrization.weight[idx]`,
        is in training mode on removal, it will perform another power iteration.
        If you'd like to avoid this iteration, set the module to eval mode
        before its removal.

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter. Default: ``"weight"``.
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm. Default: ``1``.
        eps (float, optional): epsilon for numerical stability in
            calculating norms. Default: ``1e-12``.
        dim (int, optional): dimension corresponding to number of outputs.
            Default: ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``
        bound (float, optional): upper spectral normalization bound. Default: ``1.0``.

    Returns:
        The original module with a new parametrization registered to the specified
        weight

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> snm = spectral_norm(nn.Linear(20, 40))
        >>> snm
        ParametrizedLinear(
          in_features=20, out_features=40, bias=True
          (parametrizations): ModuleDict(
            (weight): ParametrizationList(
              (0): _SpectralNorm()
            )
          )
        )
        >>> torch.linalg.matrix_norm(snm.weight, 2)
        tensor(1.0081, grad_fn=<AmaxBackward0>)
    """
    weight = getattr(module, name, None)
    if not isinstance(weight, Tensor):
        raise ValueError(
            f"Module '{module}' has no parameter or buffer with name '{name}'"
        )

    if dim is None:
        if isinstance(
            module,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            dim = 1
        else:
            dim = 0
    parametrize.register_parametrization(
        module, name, BoundedSpectralNorm(weight, n_power_iterations, dim, eps, bound)
    )
    return module