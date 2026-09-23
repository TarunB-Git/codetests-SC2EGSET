from __future__ import annotations

import numpy as np
import ot
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule


class VelocityMLP(nn.Module):
    """Time-conditioned velocity field v_θ(z, t) → dz/dt."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, t], dim=-1))


class LitOTFlowMatching(LightningModule):
    """Lightning module for OT-paired Flow Matching.

    During training, each mini-batch is re-paired using the exact EMD transport
    plan so that the learned velocity field follows the shortest possible
    trajectories between the two distributions.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.net = VelocityMLP(latent_dim=latent_dim, hidden_dim=hidden_dim)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(z, t)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        z0, z1 = batch
        batch_size = z0.shape[0]

        # OT pairing within the mini-batch using exact EMD
        z0_np = z0.detach().cpu().numpy()
        z1_np = z1.detach().cpu().numpy()
        M = ot.dist(z0_np, z1_np, metric="sqeuclidean")
        a, b = ot.unif(batch_size), ot.unif(batch_size)
        T = ot.emd(a, b, M)
        best_indices = np.argmax(T, axis=1)
        z1_paired = z1[best_indices]

        t = torch.rand((batch_size, 1), device=self.device)
        z_t = (1 - t) * z0 + t * z1_paired
        target_v = z1_paired - z0
        v_pred = self(z_t, t)
        loss = F.mse_loss(v_pred, target_v)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        z0, z1 = batch
        batch_size = z0.shape[0]
        t = torch.rand((batch_size, 1), device=self.device)
        z_t = (1 - t) * z0 + t * z1
        target_v = z1 - z0
        v_pred = self(z_t, t)
        loss = F.mse_loss(v_pred, target_v)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @torch.no_grad()
    def predict_path(
        self,
        z_start: np.ndarray,
        steps: int = 20,
        t_max: float = 1.0,
    ) -> np.ndarray:
        """Euler integration from z_start through the learned velocity field.

        Parameters
        ----------
        z_start:
            Starting latent vector, shape ``(latent_dim,)``.
        steps:
            Number of Euler integration steps.
        t_max:
            Integration end time (default 1.0 = full transport).

        Returns
        -------
        np.ndarray
            Trajectory of shape ``(steps + 1, latent_dim)``.
        """
        self.eval()
        device = next(self.parameters()).device
        z = torch.as_tensor(z_start, device=device, dtype=torch.float32)
        if z.dim() == 1:
            z = z.unsqueeze(0)

        dt = t_max / steps
        trajectory = [z.cpu().numpy().squeeze()]

        for i in range(steps):
            t = torch.full((z.shape[0], 1), i / steps, device=device)
            v = self(z, t)
            z = z + v * dt
            trajectory.append(z.cpu().numpy().squeeze())

        return np.array(trajectory)

    def predict_path_guided(
        self,
        z_start: np.ndarray,
        score_fn,
        guidance_scale: float = 0.05,
        steps: int = 20,
        t_max: float = 1.0,
    ) -> np.ndarray:
        """Euler integration with classifier guidance.

        At each step the velocity field is augmented by the gradient of the
        score function (P(win)) with respect to the current latent.  This
        steers the path toward the high-P(win) subset of the winning
        distribution rather than the unweighted average.

        Parameters
        ----------
        z_start:
            Starting latent vector, shape ``(latent_dim,)``.
        score_fn:
            Callable ``(z: Tensor[batch, latent_dim]) → Tensor[batch]``
            returning P(win) ∈ [0, 1].  Must support autograd.
        guidance_scale:
            Weight on the score gradient relative to the flow velocity.
            Larger values push harder toward high-P(win) regions but may
            leave the learned manifold.
        steps:
            Number of Euler integration steps.
        t_max:
            Integration end time.

        Returns
        -------
        np.ndarray
            Trajectory of shape ``(steps + 1, latent_dim)``.
        """
        self.eval()
        device = next(self.parameters()).device
        z = torch.as_tensor(z_start, device=device, dtype=torch.float32)
        if z.dim() == 1:
            z = z.unsqueeze(0)

        dt = t_max / steps
        trajectory = [z.detach().cpu().numpy().squeeze()]

        for i in range(steps):
            t = torch.full((z.shape[0], 1), i / steps, device=device)

            with torch.no_grad():
                v = self(z, t)

            # Classifier guidance: ∇_z P(win), normalised to unit norm so that
            # guidance_scale is a consistent fraction of the flow step size
            # regardless of how large or small the raw gradient happens to be.
            z_g = z.detach().requires_grad_(True)
            score = score_fn(z_g)
            score.sum().backward()
            if z_g.grad is None:
                raise RuntimeError("Classifier guidance did not produce a gradient")
            grad = z_g.grad.detach()
            grad_unit = grad / (grad.norm(dim=-1, keepdim=True) + 1e-8)

            z = z.detach() + dt * (v + guidance_scale * grad_unit)
            trajectory.append(z.detach().cpu().numpy().squeeze())

        return np.array(trajectory)
