"""
Copyright 2025 AUMOVIO. All rights reserved.
"""
import torch
import torchmetrics as tm
from abc import ABC, abstractmethod
from typing import Optional, Callable
import matplotlib.pyplot as plt


from predictor.metrics.utils import _resolve_final_timestep_as_int, _extract_gaussian_params, _extract_xy_coordinates, _normalize_weights, _get_denominator

class SpectralDiversity(tm.Metric):
    """
    Spectral diversity metric for trajectory ensembles.

    Computes diversity from the eigenvalue spectrum of a pairwise similarity matrix.

    The measure treats the normalized eigenvalues as a probability distribution
    and computes either Shannon (q≈1) or Tsallis (q≠1) entropy, then transforms
    it into an "effective number of modes" (the exponential of the entropy).

    Formally:
        similarities = pairwise_similarity(pred[mask])  →  [A, N, N] or [N, N]
        eigvals = eigvalsh(similarities)
        p_i = eigvals / Σ_i eigvals
        D_q = exp( H_q(p) )

    where
        - H_1(p)   = -Σ_i p_i log p_i          (Shannon entropy)
        - H_q(p)   = (1 / (1 - q)) log Σ_i p_i^q  (Tsallis entropy)

    Args:
        pairwise_similarity: Callable that returns a pairwise similarity matrix
            given trajectories, e.g. a Minkowski or Wasserstein measure.
        q: Entropy order. q→1 recovers Shannon; q≠1 yields Tsallis entropy.
        eps: Small numerical constant for stability.
    """
    def __init__(
        self,
        pairwise_similarity: Callable,
        q: float = 1.0,
        eps: Optional[float] = 1e-8,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.q = q
        self.pairwise_similarity = pairwise_similarity

        self.add_state("sum_value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    @torch.no_grad()
    def compute_per_sample_values(
        self,
        trajs: torch.Tensor,  # [A, N, T, D]
        probs: torch.Tensor, # [A, N]
        mask: torch.Tensor, # [A] mask
        **kwargs
    ) -> torch.Tensor:
        
        similarities = self.pairwise_similarity(trajs[mask], probs[mask], **kwargs) # returns either [A, N, N] or [N, N]
        # Compute eigenvalues (guaranteed symmetric)
        eigvals = torch.linalg.eigvalsh(similarities)  
        dim = 1 if eigvals.ndim > 1 else 0
        # Ensure eigenvalues are non-negative (numerical stability)
        eigvals = eigvals.clamp(min=self.eps)
        # Normalize eigenvalues into probabilities
        eigvals_normalized = eigvals / eigvals.sum(dim=dim, keepdim=True) 
        # Compute entropy
        if abs(self.q - 1.0) < 1e-4:
            entropy = -torch.sum(eigvals_normalized * torch.log(eigvals_normalized + self.eps), dim=dim)
            diversity = torch.exp(entropy)
        else:
            tsallis_sum = torch.sum(eigvals_normalized ** self.q, dim=dim)
            diversity = torch.exp((1 / (1 - self.q)) * torch.log(tsallis_sum + self.eps))

        return diversity

    @torch.no_grad()
    def update(
        self,
        trajs: torch.Tensor,  # [A, N, T, D]
        probs: torch.Tensor, # [A, N]
        mask: torch.Tensor, # [A] mask
        **kwargs
    ) -> None:
        """
        Update the running diversity estimate.

        Args:
            pred: Trajectories [A, N, T, D].
            mask: Boolean or integer mask [A], selecting valid batches.

        Returns:
            None (internal states updated).
        """
        diversity = self.compute_per_sample_values(trajs, probs, mask, **kwargs)
        
        self.sum_value += diversity.sum()
        self.n_samples += diversity.numel()

    def compute(self) -> torch.Tensor:
        return self.sum_value / self.n_samples.clamp_min(1).float()

class PairwiseSimilarity():
    """
    Composes a pairwise measure with a similarity kernel.

    Given a measure M(i,j) that quantifies dissimilarity between trajectories
    and a kernel K that maps measures to [0,1], this class produces
    pairwise similarity matrices:

        S(i,j) = K(M(i,j))

    Args:
        measure: Object implementing `compute(trajs)` that returns distances or measures [A, N, N].
        kernel:  Object implementing `compute(distances)` that maps measures to similarities.
        weighted: If similarities are weighted by instance probabilities/frequencies
    """
    def __init__(self, measure, kernel, weighted: bool = False, eps: float=1e-8):
        self.measure = measure
        self.kernel = kernel
        self.weighted = weighted
        self.eps = eps

    def __call__(self, 
                 trajs: torch.Tensor, # [A, N, T, D]
                 probs: torch.Tensor, # [A, N]
                 **kwargs) -> torch.Tensor:
        """Alias for `.compute()`."""
        return self.compute(trajs=trajs, probs=probs, **kwargs)

    @torch.no_grad()
    def compute(self, 
                trajs: torch.Tensor, # [A, N, T, D]
                probs: Optional[torch.Tensor] = None, # [A, N]
                **kwargs) -> torch.Tensor:
        """
        Compute pairwise similarities.

        Args:
            trajs: Trajectory tensor of shape [A, N, T, D].

        Returns:
            Similarities of shape [A, N, N], symmetric with ones on the diagonal.
        """
        d = self.measure.compute(trajs=trajs, **kwargs)
        sims = self.kernel(d)
        if self.weighted: sims = self.weight_similarities(sims, probs)
        return sims

    def weight_similarities(self, similarities, probs):
        """
        Weight similarities per instance probabilities/frequencies/weights
        """
        w = probs
        w = w / w.sum(dim=1, keepdim=True).clamp_min(self.eps)  # [A', N]
        sqrt_w = torch.sqrt(w.clamp_min(self.eps))              # [A', N]
        # S^{(p)}_a = D_a S_a D_a  with D_a = diag(sqrt(w_a))
        similarities = (
            similarities
            * sqrt_w[:, :, None]    # multiply rows
            * sqrt_w[:, None, :]    # multiply columns
        )

        return similarities


class ComposedPairwiseSimilarity(PairwiseSimilarity):
    """
    Combines multiple pairwise similarities into a weighted composite.

    Each sub-similarity produces [A, N, N] matrices in [0,1].
    The final similarity is their weighted average:

        S = Σ w_k * S_k / Σ w_k
    """
    def __init__(self, similarities, weights):
        self.similarities = similarities
        self.weights = weights

    @torch.no_grad()
    def compute(self, trajs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute the weighted composite similarity.

        Args:
            trajs: Trajectory tensor of shape [A, N, T, D].

        Returns:
            Combined similarity matrix [A, N, N].
        """
        sims = [s(trajs=trajs, **kwargs) for s in self.similarities] 
        if self.weights is None or len(self.weights) != len(sims):
            raise ValueError("Weights must match number of similarities")
        w = torch.tensor(self.weights, device=sims[0].device, dtype=sims[0].dtype)
        w = w / w.sum()
        S = torch.stack(sims) * w[:, None, None, None]
        return S.sum(dim=0)


class PooledSimilarity(PairwiseSimilarity):
    """
    Applies pooling across agents for a given pairwise similarity.

    Pooling summarizes similarities across the agent dimension:

        pooling='max' → retain maximum similarity (most similar pair)
        pooling='min' → retain minimum similarity (least similar pair)
    """
    def __init__(self, base: PairwiseSimilarity, pooling: Optional[str] = None):
        self.base = base
        self.pooling = pooling

    @torch.no_grad()
    def compute(self, *, trajs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute pooled similarities.

        Args:
            trajs: Trajectory tensor of shape [A, N, T, D].

        Returns:
            Pooled similarity matrix [N, N] (if pooled) or [A, N, N] (if not).
        """
        sims = self.base(trajs=trajs, **kwargs)  # [A, N, N] or [N, N] if already pooled

        if self.pooling is None:
            return sims

        if self.pooling.lower() == "max":
            # max pooling ⇒ keep the most similar pair
            sims = sims.max(dim=0).values
        elif self.pooling.lower() == "min":
            # min pooling ⇒ keep the least similar pair
            sims = sims.min(dim=0).values
        else:
            raise ValueError(f"Unknown pooling type {self.pooling!r}")

        return sims
    

class SimilarityKernel(ABC):
    """
    Abstract base for similarity kernels that map measures to similarities in [0,1].
    """
    def __call__(self, distances: torch.Tensor) -> torch.Tensor:
        return self.compute(distances)

    @abstractmethod
    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    @abstractmethod
    def visualize(self, max_distance: float, num_points) -> None:
        """Plot the similarity kernel as a function of the measure."""
        raise NotImplementedError()


class GaussianKernel(SimilarityKernel):
    """
    Gaussian (RBF) similarity kernel.

    Produces a smooth, radial decay of similarity from 1.0 to ~0.0
    as distances increase beyond a given lower bound:

        sim(d) = exp(- relu(d - lower_bound)^2 / (2 * scale^2))

    Args:
        lower_bound: Distance below which similarity ≈ 1.0.
        scale: Gaussian standard deviation σ controlling decay rate.
    """
    def __init__(self, lower_bound: float = 1.0, scale: float = 5.0):
        if not (scale > 0 and lower_bound >= 0):
            raise ValueError(
                f"scale must be > 0 and lower_bound >= 0, got ({lower_bound}, {scale})"
            )
        self.lower_bound = float(lower_bound)
        self.scale = float(scale)

    @torch.no_grad()
    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian kernel to a matrix of distances.

        Args:
            distances: Tensor of distances (any shape).
        Returns:
            similarities: Tensor of same shape, values in [0, 1].
        """
        d = torch.relu(distances - self.lower_bound)
        return torch.exp(- (d * d) / (2 * self.scale ** 2))
    
    def visualize(self, max_distance: float = 100.0, num_points: int = 1000) -> None:
        """
        Plot the similarity curve as a function of distance.
        """
        d = torch.linspace(0, max_distance, num_points)
        sims = self.compute(d)

        plt.figure(figsize=(6, 4))
        plt.plot(d.numpy(), sims.numpy(), label="Gaussian similarity", lw=2)
        plt.axvline(self.lower_bound, color="gray", ls="--", label="lower_bound")
        plt.axvline(self.scale, color="gray", ls=":", label="scale")
        plt.xlabel(f"Distance")
        plt.ylabel("Similarity")
        plt.ylim(-0.05, 1.05)
        plt.title(f"Gaussian Squashed Similarity (lower={self.lower_bound}, scale={self.scale})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


class LinearKernel(SimilarityKernel):
    """
    Linear similarity kernel.

    Maps input values linearly to [0,1] within [lower_bound, upper_bound]:

        sim(x) = clamp((x - lower) / (upper - lower), 0, 1)

    Values below lower_bound map to 0, above upper_bound to 1.
    """
    def __init__(self, lower_bound: float = 0.0, upper_bound: float = 1.0):
        # bounds checks
        if not (-1.0 <= lower_bound < upper_bound <= 1.0):
            raise ValueError(f"cosine bounds must satisfy -1 <= lower < upper <= 1, got ({lower_bound}, {upper_bound})")
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)

    @torch.no_grad()
    def compute(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Apply linear kernel to a matrix of distances.

        Args:
            distances: Tensor of distances (any shape).
        Returns:
            similarities: Tensor of same shape, values in [0, 1].
        """
        return torch.clamp((distances - self.lower_bound)/ (self.upper_bound - self.lower_bound), 0.0, 1.0)

    def visualize(self, num_points: int = 400) -> None:
        """Plot the linear kernel mapping."""
        d = torch.linspace(-1.0, 1.0, num_points)
        sims = (d - self.lower_bound) / (self.upper_bound - self.lower_bound)
        sims = sims.clamp(0.0, 1.0)

        plt.figure(figsize=(6, 4))
        plt.plot(d.cpu().numpy(), sims.cpu().numpy(), lw=2, label="linear mapping")
        plt.axvline(self.lower_bound, color="gray", ls="--", label="lower_bound")
        plt.axvline(self.upper_bound, color="gray", ls=":", label="upper_bound")
        plt.xlabel("Cosine similarity (cos Δθ)")
        plt.ylabel("Mapped similarity")
        plt.title(f"HeadingSimilarity (lower={self.lower_bound}, upper={self.upper_bound})")
        plt.ylim(-0.05, 1.05)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


class HeadingCosineMeasure():
    """
    Computes pairwise angular similarity based on agent headings at a chosen t_max.

    The measure returns pairwise cosine similarities in [-1, 1] between agent headings.
    It supports both scalar heading angles (yaw in radians) and 2D unit vectors 
    representing [cos(θ), sin(θ)].

    Args:
        t: Timestep index to evaluate (negative = relative to end).
    """
    def __init__(self, t_max: int = -1):
        self.t_max = int(t_max)

    @torch.no_grad()
    def compute(self, trajs: torch.Tensor, heading_idx: int | slice | list[int] = 2, **kwargs) -> torch.Tensor:
        """
        Compute pairwise cosine similarity of agent headings.

        Args:
            trajs: Trajectory tensor [A, N, T, D].
            heading_idx: Index, slice, or list of indices selecting heading components.

        Returns:
            cos_sim: Tensor [A, N, N] containing cosine similarities in [-1, 1].
        """
        A, N, T, D = trajs.shape # [A,N,T] = #agents #modes #timestep
        # resolve timestep with clamping so out-of-range indices don't crash
        t = _resolve_final_timestep_as_int(T, self.t_max)

        headings = trajs[..., t, heading_idx]  
        # if we got multiple components, interpret as a 2D vector
        if headings.ndim == 3:
            # "Assuming [cos(theta),sin(theta)] headings"
            norm = torch.norm(headings, dim=-1, keepdim=True)
            headings = headings / norm
            cos_t = headings[..., 0]
            sin_t = headings[..., 1]
        else:
            # "Assuming yaw"
            cos_t = torch.cos(headings)
            sin_t = torch.sin(headings)

        cos_sim = cos_t[:, :, None] * cos_t[:, None, :] + sin_t[:, :, None] * sin_t[:, None, :] 
        return cos_sim


class GeometricEndpointMeasure():
    """
    Computes pairwise Minkowski distances between geometric trajectory endpoints.

    Measures spatial dissimilarity at a specific timestep t_max:
        d(i,j) = ||x_i(t) - x_j(t)||_order

    Args:
        t_max: Timestep index to evaluate (negative = relative to end).
        order: : order of the Minkowski norm
    """
    def __init__(self, t_max: int = -1, order: float = 2.0, normalize: bool = False):
        if order <= 0:
            raise ValueError("order must be positive (p > 0).")
        self.t_max = int(t_max)
        self.order = float(order)
        self.normalize = bool(normalize)

    @torch.no_grad()
    def compute(self, trajs: torch.Tensor, position_idx: int | slice | list[int] = slice(0,2), curr_vel: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Compute pairwise Minkowski distances between geometric endpoints.

        Args:
            trajs: Trajectory tensor [A, N, T, D].
            position_idx: Indices or slice selecting spatial coordinates.
            curr_vel [Optional]: initial velocities [A] to normalize distances

        Returns:
            distances: Tensor [A, N, N] of pairwise distances [m] or [s].
        """
        A, N, T, D = trajs.shape
        # resolve timestep safely
        t = _resolve_final_timestep_as_int(T, self.t_max)

        endpoints = trajs[..., t, position_idx]           # [A, N, 2] only need last two dimensions
        distances = torch.cdist(endpoints, endpoints, p=self.order)  # [A, N, N]

        denom = _get_denominator(A, distances.device, distances.dtype, curr_vel=curr_vel, normalize=self.normalize)

        return distances / denom.reshape(-1, 1, 1) # in meters (no curr_vel) or seconds (with curr_vel)
    

class GeometricLockstepMeasure():
    """
    Computes pairwise Minkowski distances between geometric trajectories points(lockstep).

    Each trajectory is flattened across time and compared directly in feature space:
        d(i,j) = ||x_i[..., :t+1, ...] - x_j[..., :t+1, ...]||_order

    Args:
        t_max: range of timesteps to evaluate
        order: order of the Minkowski norm
    """
    def __init__(self, t_max: int = -1, order: float=2.0, normalize: bool=False):
        if order <= 0:
            raise ValueError("order must be positive (p > 0).")
        self.t_max = t_max
        self.order = order
        self.normalize = normalize

    @torch.no_grad()
    def compute(self, trajs: torch.Tensor, position_idx: int | slice | list[int] = slice(0,2), curr_vel: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Compute pairwise Minkowski lockstep distances across full trajectories.

        Args:
            trajs: Trajectory tensor [A, N, T, D].
            position_idx: Indices or slice selecting spatial coordinates.
            curr_vel [Optional]: initial velocities [A] to normalize distances

        Returns:
            distances: Tensor [A, N, N] of Euclidean distances [m].
        """
        A, N, T, D = trajs.shape
        # resolve timestep safely
        t = _resolve_final_timestep_as_int(T, self.t_max)

        trajs_xy = trajs[..., :t+1, position_idx] # only need last two dimensions
        traj_flat = trajs_xy.reshape(A, N, -1)                   # [A, N, T*2]
        diff = traj_flat[:, :, None, :] - traj_flat[:, None, :, :]  # [A, N, N, T*2]
        if self.order == torch.inf:
            distances = torch.linalg.vector_norm(diff, ord=float('inf'), dim=-1)
        else:
            distances = torch.linalg.vector_norm(diff, ord=self.order, dim=-1)
        
        denom = _get_denominator(A, distances.device, distances.dtype, curr_vel=curr_vel, normalize=self.normalize)

        return distances / denom.reshape(-1, 1, 1) # in meters (no curr_vel) or seconds (with curr_vel)
    

class GaussianWasserstein2EndpointMeasure():
    """
    Pairwise 2-Wasserstein distances between bivariate Gaussian endpoints.

    Each endpoint is parameterized as (mu_x, mu_y, std_x, std_y, rho) at timestep t_max.
    Uses the closed-form W2^2 between Gaussians:
      W2^2 = ||m1 - m2||^2 + Tr(S1 + S2 - 2*(S1^{1/2} S2 S1^{1/2})^{1/2})

    Args:
        t_max: which timestep to read (negative = relative to end).
        squared: if True, return W2^2; else return W2.
        eps: numerical floor for stds / rho / eigenvalues.
        backend: "native" (pure PyTorch) or "pot" (POT gaussian tools).
    """
    def __init__(self, t_max: int = -1, squared: bool = False, eps: float = 1e-8, backend: str="native", normalize: bool=False):
        self.t_max = int(t_max)
        self.squared = squared
        self.eps = float(eps)
        self.clamp_rho = 1 - self.eps
        self.backend = backend
        self.normalize = normalize
        #self.unit = "m"  # same unit convention as your Euclidean measure

    @staticmethod
    def _sqrtm_psd(M: torch.Tensor, eps: float) -> torch.Tensor:
        """
        Batched symmetric PSD matrix square root via eigen-decomposition.
        M: [..., d, d] with d=2 here.
        """
        # eigh: symmetric eigendecomposition; values sorted ascending
        evals, evecs = torch.linalg.eigh(M)
        evals_clamped = torch.clamp(evals, min=0.0)
        # Stabilize tiny negatives from numerics
        evals_sqrt = torch.sqrt(evals_clamped + eps)
        # Recompose: V diag(sqrt) V^T
        return (evecs * evals_sqrt.unsqueeze(-2)) @ evecs.transpose(-2, -1)
    
    def _pairwise_bures_native(self, Sigma: torch.Tensor) -> torch.Tensor:
        """
        Closed-form Bures distance between all pairs in Sigma.
        Sigma: [A, N, 2, 2] -> returns [A, N, N] (non-squared Bures).
        """
        A, N, _, _ = Sigma.shape
        S1 = Sigma[:, :, None, :, :]              # [A, N, 1, 2, 2]
        S2 = Sigma[:, None, :, :, :]              # [A, 1, N, 2, 2]

        S1_sqrt = self._sqrtm_psd(S1, self.eps)   # [A, N, N, 2, 2]
        inner = S1_sqrt @ S2 @ S1_sqrt            # [A, N, N, 2, 2]
        inner_sqrt = self._sqrtm_psd(inner, self.eps)

        trace = lambda M: M[..., 0, 0] + M[..., 1, 1]
        tr_S1 = trace(S1.expand_as(inner))
        tr_S2 = trace(S2.expand_as(inner))
        tr_inner_sqrt = trace(inner_sqrt)

        cov_term = tr_S1 + tr_S2 - 2.0 * tr_inner_sqrt         # [A, N, N]
        cov_term = torch.clamp(cov_term, min=0.0)              # numerical safety
        return torch.sqrt(cov_term)
    
    def _pairwise_bures_pot(self, Sigma: torch.Tensor) -> torch.Tensor:
        """
        Bures distance using POT (torch backend), all pairs per batch.
        Sigma: [A, N, 2, 2] -> returns [A, N, N] (non-squared Bures).
        """
        try:
            import ot
        except Exception as e:
            raise ImportError(
                "backend='pot' requires the POT package (pip install POT)"
            ) from e
        
        A, N, _, _ = Sigma.shape
        B_all = []
        for aidx in range(A):
            # ot.gaussian.bures_distance accepts a batch of covariances and returns pairwise distances [N, N]
            B = ot.gaussian.bures_distance(Sigma[aidx], Sigma[aidx])  # [N, N], non-squared Bures
            # Ensure tensor on the same device as inputs
            if not isinstance(B, torch.Tensor):
                # POT might return numpy if backend is mismatched; convert safely
                B = torch.as_tensor(B, device=Sigma.device, dtype=Sigma.dtype)
            else:
                B = B.to(Sigma.device, dtype=Sigma.dtype)
            B_all.append(B)
        return torch.stack(B_all, dim=0)  # [A, N, N]

    def enforce_symmetry_zero_diag(self, Dmat: torch.Tensor) -> torch.Tensor:
        Dmat = 0.5 * (Dmat + Dmat.transpose(1,2))
        
        diag = torch.diagonal(Dmat, dim1=-2, dim2=-1)
        diag.zero_()

        return Dmat

    @torch.no_grad()
    def compute(self, trajs: torch.Tensor, param_idx: int | slice | list[int] = slice(0, 5), curr_vel: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Compute pairwise W2 (or W2^2) distances between Gaussian endpoints.

        Args:
            trajs: Tensor [A, N, T, D] containing the 5-tuple per agent at timestep t_max:
                   (mu_x, mu_y, std_x, std_y, rho) in the dimensions selected by `param_idx`.
            param_idx: indices/slice selecting those 5 parameters.
            curr_vel [Optional]: initial velocities [A] to normalize distances

        Returns:
            distances: [A, N, N] tensor of W2 distances (or squared if squared=True).
        """
        A, N, T, D = trajs.shape
        # resolve timestep safely
        t = _resolve_final_timestep_as_int(T, self.t_max)

        mu, _, _ , _, Sigma = _extract_gaussian_params(trajs, param_idx = param_idx, t = t)

        # Pairwise mean term: ||m_i - m_j||^2
        dm = mu[:, :, None, :] - mu[:, None, :, :]              # [A, N, N, 2]
        mean_term = (dm * dm).sum(dim=-1)                       # [A, N, N]

        if self.backend == "native":
            bures = self._pairwise_bures_native(Sigma)
        elif self.backend == "pot":
            bures = self._pairwise_bures_pot(Sigma)
        else:
            raise ValueError("backend must be 'native' or 'pot'")

        bures = self.enforce_symmetry_zero_diag(bures)
        w2_sq = mean_term + bures ** 2 

        # handling of velocity normalization and squared settings
        distances = w2_sq if self.squared else torch.sqrt(torch.clamp(w2_sq, min=0.0))

        denom = _get_denominator(A, distances.device, distances.dtype, curr_vel=curr_vel, normalize=self.normalize)

        return distances / denom.reshape(-1, 1, 1)
