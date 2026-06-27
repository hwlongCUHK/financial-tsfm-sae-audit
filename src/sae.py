"""TopK Sparse Autoencoder for residual stream interpretability.

This module implements the TopK SAE architecture used throughout the paper.
The SAE learns a sparse overcomplete dictionary of the residual stream at a
given Kronos transformer layer. Each input is encoded as a sparse linear
combination of at most *k* dictionary elements (features).

Architecture
------------
    encode:  z = TopK( W_enc (x - b_pre) )
    decode:  x_hat = W_dec z + b_pre

where TopK keeps only the *k* largest entries of the pre-activation latent
vector and zeros the rest.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TopKSAE(nn.Module):
    """Top-K Sparse Autoencoder.

    Args:
        d_input: Dimensionality of the input (residual stream width).
        d_hidden: Dimensionality of the latent space (d_input * expansion).
        k: Number of active (non-zero) latent features per sample.
    """

    def __init__(self, d_input: int, d_hidden: int, k: int = 64) -> None:
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.k = k

        self.enc = nn.Linear(d_input, d_hidden, bias=True)
        self.dec = nn.Linear(d_hidden, d_input, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(d_input))

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse latent representation.

        Args:
            x: Input tensor of shape ``(batch, d_input)``.

        Returns:
            Sparse latent tensor of shape ``(batch, d_hidden)`` with at most
            *k* non-zero entries per sample.
        """
        xc = x - self.b_pre
        latent = self.enc(xc)
        _, topk_idx = torch.topk(latent, self.k, dim=-1)
        mask = torch.zeros_like(latent)
        mask.scatter_(-1, topk_idx, 1.0)
        return latent * mask

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode sparse latent representation back to input space.

        Args:
            latent: Sparse latent tensor of shape ``(batch, d_hidden)``.

        Returns:
            Reconstructed tensor of shape ``(batch, d_input)``.
        """
        return self.dec(latent) + self.b_pre

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full encode-decode pass.

        Args:
            x: Input tensor of shape ``(batch, d_input)``.

        Returns:
            Tuple of (reconstruction, sparse_latent).
        """
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent

    def ablate_reconstruct(
        self, x: torch.Tensor, feature_ids: list[int]
    ) -> torch.Tensor:
        """Reconstruct with specified latent features zeroed out (ablated).

        This is used for intervention experiments: encode through the SAE,
        zero target features, decode back into the residual stream.

        Args:
            x: Input tensor of shape ``(batch, d_input)``.
            feature_ids: Indices of latent features to ablate.

        Returns:
            Ablated reconstruction of shape ``(batch, d_input)``.
        """
        xc = x - self.b_pre
        latent = self.enc(xc)
        _, topk_idx = torch.topk(latent, self.k, dim=-1)
        mask = torch.zeros_like(latent)
        mask.scatter_(-1, topk_idx, 1.0)
        mask[:, feature_ids] = 0.0
        return self.dec(latent * mask) + self.b_pre


def train_sae(
    sae: TopKSAE,
    train_acts: torch.Tensor,
    steps: int = 3000,
    batch_size: int = 256,
    lr: float = 1e-4,
    grad_clip: float = 1.0,
    device: Optional[str] = None,
    log_interval: int = 1000,
) -> TopKSAE:
    """Train a TopK SAE on pre-extracted activations.

    Args:
        sae: The SAE model (already on device).
        train_acts: Activation tensor of shape ``(n_samples, d_input)``,
            as a numpy array or torch tensor.
        steps: Number of gradient steps.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        grad_clip: Maximum gradient norm.
        device: Target device string (e.g. ``"cuda:0"``).
        log_interval: Print loss every N steps (0 to disable).

    Returns:
        The trained SAE (same object, modified in place).
    """
    if device is None:
        device = next(sae.parameters()).device

    if not isinstance(train_acts, torch.Tensor):
        train_acts = torch.from_numpy(train_acts).float()
    train_acts = train_acts.to(device)

    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    for step in range(steps):
        idx = torch.randint(0, len(train_acts), (batch_size,), device=device)
        x_batch = train_acts[idx]

        latent = sae.encode(x_batch)
        recon = sae.decode(latent)
        loss = F.mse_loss(recon, x_batch)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(sae.parameters(), grad_clip)
        optimizer.step()

        if log_interval > 0 and (step + 1) % log_interval == 0:
            logger.info("step %d/%d  loss=%.6f", step + 1, steps, loss.item())

    return sae
