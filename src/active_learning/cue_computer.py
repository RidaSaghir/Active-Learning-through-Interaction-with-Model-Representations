import torch
import numpy as np

def entropy(P: torch.Tensor) -> torch.Tensor:
    logP = torch.log(P + 1e-12)
    return -(P * logP).sum(dim=1)

def diversity_to_labeled(Z_u: torch.Tensor, Z_l: torch.Tensor) -> torch.Tensor:
    """
    d(i) = min_{l in L} ||z_i - z_l||, i over unlabeled.
    Z_u: [N_u, d], Z_l: [N_l, d]
    """
    if Z_l.numel() == 0 or Z_u.numel() == 0:
        return torch.zeros(Z_u.shape[0])
    dist = torch.cdist(Z_u, Z_l)              # [N_u, N_l]
    return dist.min(dim=1).values             # [N_u]

def knn_density(Z_u: torch.Tensor, k: int = 15) -> torch.Tensor:
    """
    Local density among unlabeled: rho(i) ≈ 1 / avg kNN distance.
    Z_u: [N_u, d]
    """
    N_u = Z_u.shape[0]
    if N_u <= 1:
        return torch.zeros(N_u)

    dist = torch.cdist(Z_u, Z_u)              # [N_u, N_u]
    # nearest k+1 (including self), drop self
    knn = dist.topk(k=min(k+1, N_u), largest=False).values[:, 1:]
    avg = knn.mean(dim=1)
    return 1.0 / (avg + 1e-8)

def novelty_from_density(rho_u: torch.Tensor) -> torch.Tensor:
    """Novelty = 1 / density (global rarity in unlabeled pool)."""
    return 1.0 / (rho_u + 1e-8)

def coverage_pressure(P_u: torch.Tensor, labeled_counts: dict) -> torch.Tensor:
    """
    w_cov(i) = 1 / (N_L(pred_class) + eps)
    P_u: [N_u, C] probabilities for unlabeled
    labeled_counts: {class_id -> count_in_labeled}
    """
    pred = P_u.argmax(dim=1)                  # [N_u]
    counts = torch.tensor(
        [labeled_counts.get(int(c), 0) for c in pred],
        dtype=torch.float32
    )
    return 1.0 / (counts + 1e-3)              # [N_u]

def minmax(x: torch.Tensor) -> torch.Tensor:
    """Normalize to [0, 1] per vector."""
    minv, maxv = x.min(), x.max()
    return (x - minv) / (maxv - minv + 1e-8)

def compute_cues_from_view(
    emb_np: np.ndarray,
    prob_np: np.ndarray,
    is_labeled_np: np.ndarray,
    labeled_counts: dict,
    k_density: int = 15,
    random_seed: int | None = None,
):

    # Torch-ify inputs
    Z = torch.from_numpy(emb_np).float()  # [N, d]
    P = torch.from_numpy(prob_np).float()  # [N, C]
    is_labeled = torch.from_numpy(is_labeled_np.astype(bool))  # [N]

    mask_u = ~is_labeled  # unlabeled mask
    Z_u = Z[mask_u]  # [N_u, d]
    P_u = P[mask_u]  # [N_u, C]
    Z_l = Z[is_labeled]  # [N_l, d]
    N = Z.shape[0]

    # Uncertainty
    uncertainty = entropy(P)                  # [N]

    # Density
    rho_u = knn_density(Z_u, k=k_density)     # [N_u]
    density = torch.zeros(N)
    density[mask_u] = rho_u  # 0 for labeled

    # Novelty
    novelty_u = novelty_from_density(rho_u)  # [N_u]
    novelty = torch.zeros(N)
    novelty[mask_u] = novelty_u

    # Diversity-to-labeled: d(i) = min dist to any labeled
    d_u = diversity_to_labeled(Z_u, Z_l)  # [N_u]
    diversity = torch.zeros(N)
    diversity[mask_u] = d_u

    # Coverage pressure: only defined for unlabeled
    cov_u = coverage_pressure(P_u, labeled_counts)  # [N_u]
    coverage = torch.zeros(N)
    coverage[mask_u] = cov_u

    # Random
    rng = torch.Generator()
    if random_seed is not None:
        rng.manual_seed(random_seed)
    random_scores = torch.zeros(N)
    random_scores[mask_u] = torch.rand(mask_u.sum(), generator=rng)

    return {
        "uncertainty": uncertainty.numpy(),
        "density": density.numpy(),
        "novelty": novelty.numpy(),
        "diversity": diversity.numpy(),
        "coverage": coverage.numpy(),
        "random": random_scores.numpy(),
    }



