from __future__ import annotations

import lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim

from latent_trainer.models.guided_vae import Classifier, suGuidedVAE
from latent_trainer.models.losses import loss_supervised


class LitGuidedVAE(pl.LightningModule):
    """Lightning module for supervised Guided VAE training with adversarial disentanglement."""

    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dims: list[int],
        supervised_dim: int,
        latent_dim: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        learning_rate_cls: float = 1e-4,
        weight_decay_cls: float = 1e-4,
        classification_weight: float = 50.0,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        compatibility_metadata: dict[str, str | int] | None = None,
    ) -> None:
        """Initialise the LitGuidedVAE module.

        Parameters
        ----------
        input_dim:
            Input feature dimension (depends on the transform used).
        encoder_hidden_dims:
            Hidden layer widths for the encoder.  The decoder mirrors these in
            reverse order.  Defaults to ``[64, 128, 256, 512]``.
        supervised_dim:
            Number of latent dims (per player) reserved for supervised
            classification.  The remaining ``latent_dim - supervised_dim`` dims are
            adversarially disentangled.
        latent_dim:
            Size of the VAE latent distribution.
        learning_rate:
            Learning rate for the VAE and adversarial optimizers.
        weight_decay:
            Weight decay for the VAE optimizer.
        learning_rate_cls:
            Learning rate for the classifier optimizer.
        weight_decay_cls:
            Weight decay for the classifier optimizer.
        classification_weight:
            Weight for the classification loss.
        mean:
            Optional pre-computed training data mean for input normalisation.
        std:
            Optional pre-computed training data std for input normalisation.
        """
        super().__init__()
        self.save_hyperparameters()

        # Dimensions:
        self.input_dim = input_dim
        self.encoder_hidden_dims = encoder_hidden_dims
        self.supervised_dim = supervised_dim
        self.latent_dim = latent_dim

        # Normalization stats, needed for working with unnormalized data
        # after training (e.g. when encoding new samples with the trained model):
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        # Hyperparameters:
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.learning_rate_cls = learning_rate_cls
        self.weight_decay_cls = weight_decay_cls
        self.classification_weight = classification_weight
        self.compatibility_metadata = compatibility_metadata or {}

        # Manual optimisation (3 optimizers), required for the
        # alternating adversarial training steps:
        self.automatic_optimization = False

        self.model = suGuidedVAE(
            latent_dim=latent_dim,
            input_dim=input_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            supervised_dim=supervised_dim,
        )

        # Adversarial classifier operating on the non-supervised latent dims for adversarial disentanglement.
        # Pushes the VAE to learn class-discriminative information in the supervised dims, leaving the rest
        # free of class information and hopefully more disentangled:
        self.adversarial_classifier = Classifier(
            latent_dim=latent_dim,
            supervised_dim=supervised_dim,
        )

    # Helpers
    @staticmethod
    def _prepare_label(label: torch.Tensor) -> torch.Tensor:
        """Coerce label dtype/shape and assert all values are valid (0 or 1).

        Returns ``[batch, 1]`` float32 tensor.  Raises ``ValueError`` if any
        label is outside {0, 1} — invalid samples must be filtered upstream.
        """
        if label.dtype != torch.float32:
            label = label.float()
        if label.dim() == 1:
            label = label.unsqueeze(1)

        if (label < 0).any() or (label > 1).any():
            raise ValueError(
                "Batch contains labels outside [0, 1]. "
                "Filter invalid samples in the dataset/dataloader, not here."
            )
        return label

    def _slice_free_dims(self, z: torch.Tensor) -> torch.Tensor:
        """Slice the free (non-supervised) latent dimensions from ``z``.

        The latent space is partitioned as:
          ``z[:supervised_dim]``  — supervised dims, used by the internal VAE classifier
          ``z[supervised_dim:]``  — free dims, passed to the adversarial classifier

        Handles both 2-D ``[batch, latent]`` and 3-D ``[batch, players, latent]`` inputs.
        """
        # 3-D: per-player latents [batch, players, latent] → slice last dim
        # 2-D: flat latents [batch, latent] → slice last dim
        return (
            z[:, :, self.supervised_dim :]
            if z.dim() == 3
            else z[:, self.supervised_dim :]
        )

    # Forward / training / validation
    def forward(self, x: torch.Tensor):  # noqa: D401
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        data, label = batch[0], batch[1]
        valid_data = data
        valid_label = self._prepare_label(label=label)

        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)) or len(optimizers) != 3:
            raise RuntimeError("Guided VAE requires three optimizers")
        optimizer_vae, optimizer_classification, optimizer_adversarial = optimizers
        # Step 1: VAE
        optimizer_vae.zero_grad()
        recon_batch, mu, logvar, re = self.model(valid_data)
        vae_loss = loss_supervised(recon_batch, valid_data, mu, logvar)[0]
        cls_loss = F.binary_cross_entropy(re, valid_label, reduction="mean")
        vae_total = vae_loss + cls_loss * self.classification_weight

        acc = (
            (re > 0.5).float().eq(valid_label).sum().item() / valid_label.numel() * 100
        )

        self.log("train_vae_loss", vae_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_cls_loss", cls_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_vae_acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        self.manual_backward(vae_total)
        optimizer_vae.step()

        # Step 2: Adversarial classifier
        optimizer_classification.zero_grad()
        mu, logvar = self.model.encode(valid_data)
        z = self.model.reparameterize(mu, logvar).detach()
        cls1 = self.adversarial_classifier(self._slice_free_dims(z))
        adversarial_classification_loss = (
            F.binary_cross_entropy(cls1, valid_label, reduction="mean")
            * self.classification_weight
        )

        adversarial_classification_accuracy = (
            (cls1 > 0.5).float().eq(valid_label).sum().item()
            / valid_label.numel()
            * 100
        )

        self.log(
            "train_adv_cls_loss",
            adversarial_classification_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train_adv_cls_acc",
            adversarial_classification_accuracy,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        self.manual_backward(adversarial_classification_loss)
        optimizer_classification.step()

        # Step 3: Adversarial (VAE fools the classifier)
        optimizer_adversarial.zero_grad()
        mu, logvar = self.model.encode(valid_data)
        z = self.model.reparameterize(mu, logvar)
        cls2 = self.adversarial_classifier(self._slice_free_dims(z))
        label_half = torch.empty_like(valid_label).fill_(0.5)
        adv_loss = (
            F.binary_cross_entropy(cls2, label_half, reduction="mean")
            * self.classification_weight
        )

        self.log(
            "train_adv_loss",
            adv_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.manual_backward(adv_loss)
        optimizer_adversarial.step()

        return vae_total + adversarial_classification_loss + adv_loss

    def on_train_epoch_end(self) -> None:
        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, (list, tuple)) or len(schedulers) != 3:
            raise RuntimeError("Guided VAE requires three schedulers")
        for scheduler in schedulers:
            if not isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
                raise RuntimeError("Guided VAE scheduler type differs")
            scheduler.step()

    def validation_step(self, batch, batch_idx):
        data, label = batch[0], batch[1]
        valid_data = data
        valid_label = self._prepare_label(label)

        recon_batch, mu, logvar, re = self.model(valid_data)
        vae_loss = loss_supervised(recon_batch, valid_data, mu, logvar)[0]
        cls_loss = F.binary_cross_entropy(re, valid_label, reduction="mean")
        total_loss = vae_loss + cls_loss * self.classification_weight

        acc = (
            (re > 0.5).float().eq(valid_label).sum().item() / valid_label.numel() * 100
        )

        self.log("val_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("val_vae_loss", vae_loss, prog_bar=True, sync_dist=True)
        self.log("val_cls_loss", cls_loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)

        return total_loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        if not self.trainer.max_epochs:
            raise ValueError(
                "Trainer max_epochs must be set for LR scheduler configuration."
            )

        opt_vae = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        opt_cls = optim.AdamW(
            self.adversarial_classifier.parameters(),
            lr=self.learning_rate_cls,
            weight_decay=self.weight_decay_cls,
        )
        # Adversarial optimizer updates VAE params to fool the classifier
        opt_adv = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        epochs = self.trainer.max_epochs
        sched_vae = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=opt_vae, T_max=epochs
        )
        sched_cls = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=opt_cls, T_max=epochs
        )
        sched_adv = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=opt_adv, T_max=epochs
        )

        return [opt_vae, opt_cls, opt_adv], [sched_vae, sched_cls, sched_adv]

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["benchmark_compatibility"] = self.compatibility_metadata
