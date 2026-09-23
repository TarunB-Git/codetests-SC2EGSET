"""Method dispatch: runs a single (sample, method) pair and returns PathRunResult."""

from __future__ import annotations

import math
import time
import traceback
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.neighbors import KernelDensity

from latent_trainer.paths.compare.metrics import (
    auc_trapezoidal,
    first_crossover,
    monotonicity_fraction,
    normalised_crossover_alpha,
    path_length,
    sparsity_n_changed,
)
from latent_trainer.paths.compare.results import MethodSpec, PathRunResult
from latent_trainer.paths.data import (
    FEATURE_NAMES,
    PathContext,
    get_supervised_dim,
    nearest_winning_target,
    opponent_aware_logit,
    opponent_aware_score,
)
from latent_trainer.paths.feedback import compute_feedback
from latent_trainer.paths.strategies.gradient_ascent import path_gradient_ascent
from latent_trainer.paths.strategies.linear import path_linear
from latent_trainer.paths.strategies.neural_flow import path_neural_flow
from latent_trainer.paths.strategies.optimal_transport import path_optimal_transport

if TYPE_CHECKING:
    from latent_trainer.paths.flow import LitOTFlowMatching

_KNN_K = 5


def run_method(
    *,
    spec: MethodSpec,
    ctx: PathContext,
    n_steps: int,
    top_k: int,
    flow_model: "LitOTFlowMatching | None",
    win_latents: torch.Tensor,
    loss_latents: torch.Tensor,
    all_latents: torch.Tensor,
    win_kde: KernelDensity,
) -> PathRunResult:
    """Execute one path-charting method for one sample and collect all metrics.

    On exception, returns a PathRunResult with ``error`` set and NaN metrics
    so a single failure does not abort the whole comparison sweep.
    """
    try:
        return _run_method_inner(
            spec=spec,
            ctx=ctx,
            n_steps=n_steps,
            top_k=top_k,
            flow_model=flow_model,
            win_latents=win_latents,
            loss_latents=loss_latents,
            all_latents=all_latents,
            win_kde=win_kde,
        )
    except Exception:
        tb = traceback.format_exc()
        nan = math.nan
        return PathRunResult(
            sample_idx=ctx.chosen,
            method_name=spec.name,
            player_idx=ctx.player_idx,
            path_z=np.full((n_steps, 1), nan),
            p_win_curve=np.full(n_steps, nan),
            alphas=np.linspace(0.0, 1.0, n_steps),
            crossover_alpha=None,
            crossover_wp=None,
            success=False,
            p_win_start=nan,
            p_win_end=nan,
            p_win_gain=nan,
            p_win_max=nan,
            auc_p_win=nan,
            monotonicity=nan,
            dist_to_nearest_win_start=nan,
            dist_to_nearest_win_end=nan,
            dist_to_knn_win_start=nan,
            dist_to_knn_win_end=nan,
            kde_density_start=nan,
            kde_density_end=nan,
            path_length_l2=nan,
            top_k_features_raw=(),
            top_k_features_mv=(),
            top_k_features_weighted=(),
            wall_time_s=nan,
            error=tb,
        )


def _run_method_inner(
    *,
    spec: MethodSpec,
    ctx: PathContext,
    n_steps: int,
    top_k: int,
    flow_model: "LitOTFlowMatching | None",
    win_latents: torch.Tensor,
    loss_latents: torch.Tensor,
    all_latents: torch.Tensor,
    win_kde: KernelDensity,
) -> PathRunResult:
    opponent_z = (
        ctx.latents_p1[ctx.chosen]
        if ctx.player_idx == 0
        else ctx.latents_p0[ctx.chosen]
    )
    score_fn = partial(
        opponent_aware_score,
        guided_vae=ctx.guided_vae,
        opponent_z=opponent_z,
        player_idx=ctx.player_idx,
    )
    logit_fn = partial(
        opponent_aware_logit,
        guided_vae=ctx.guided_vae,
        opponent_z=opponent_z,
        player_idx=ctx.player_idx,
    )

    device = ctx.sample_z.device
    z_start_np = ctx.sample_z.detach().cpu().numpy()
    win_np = win_latents.detach().cpu().numpy()
    # Partition latent space: strategies operate only on supervised dims;
    # free dims are pinned to the starting value and reattached after.
    sup_dim = get_supervised_dim(ctx.guided_vae)
    z_free = z_start_np[sup_dim:]

    t0 = time.perf_counter()

    match spec.strategy:
        case "linear":
            k_opp = int(spec.params.get("k_opponents", 50))
            # Opponent filtering in supervised subspace — free dims carry style/race
            # noise that is irrelevant to outcome-driven similarity matching.
            opp_dists = torch.cdist(
                opponent_z[:sup_dim].unsqueeze(0),
                loss_latents[:, :sup_dim],
            ).squeeze(0)
            opp_indices = opp_dists.topk(k_opp, largest=False).indices
            filtered_wins_sup = win_latents[opp_indices, :sup_dim]
            if spec.params.get("method") == "nearest":
                k_nn = int(spec.params.get("k_neighbours", 5))
                target_z = (
                    nearest_winning_target(
                        ctx.sample_z[:sup_dim], filtered_wins_sup, k=k_nn
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                target_z = filtered_wins_sup.mean(dim=0).detach().cpu().numpy()
            path_z_np = path_linear(
                z_start=z_start_np[:sup_dim], z_target=target_z, n_waypoints=n_steps
            )

        case "gradient_ascent":
            p = spec.params
            path_z_np = path_gradient_ascent(
                z_start=z_start_np[:sup_dim],
                score_fn=score_fn,
                logit_fn=logit_fn,
                Z_all=win_np[:, :sup_dim],
                steps=int(p.get("steps", 1000)),
                lr=float(p.get("lr", 0.005)),
                momentum=float(p.get("momentum", 0.5)),
                density_weight=float(p.get("density_weight", 0.3)),
                kde_bandwidth=float(p.get("kde_bandwidth", 0.5)),
                n_waypoints=n_steps,
                convergence_threshold=float(p.get("convergence_threshold", 0.95)),
            )

        case "optimal_transport":
            p = spec.params
            path_z_np = path_optimal_transport(
                z_start=z_start_np[:sup_dim],
                Z_win=win_np[:, :sup_dim],
                reg=float(p.get("reg", 0.01)),
                n_waypoints=n_steps,
                step_size=float(p.get("step_size", 0.1)),
                opponent_z=opponent_z.detach().cpu().numpy()[:sup_dim],
                Z_loss=loss_latents.detach().cpu().numpy()[:, :sup_dim],
                k_opponents=int(p.get("k_opponents", 50)),
            )

        case "neural_flow":
            if flow_model is None:
                raise RuntimeError(
                    "neural_flow requires a flow_model but none was provided."
                )
            guidance_scale = float(spec.params.get("guidance_scale", 0.0))
            if guidance_scale > 0.0:
                path_z_np = flow_model.predict_path_guided(
                    z_start=z_start_np[:sup_dim],
                    score_fn=score_fn,
                    guidance_scale=guidance_scale,
                    steps=n_steps - 1,
                )
            else:
                path_z_np = path_neural_flow(
                    z_start=z_start_np[:sup_dim],
                    flow_model=flow_model,
                    n_waypoints=n_steps,
                )

        case _:
            raise ValueError(f"Unknown strategy: {spec.strategy!r}")

    wall_time_s = time.perf_counter() - t0

    # Reattach the fixed free dims so downstream decoding and scoring receive
    # full-dim latent vectors as the model expects.
    path_z_np = np.concatenate(
        [path_z_np, np.tile(z_free, (len(path_z_np), 1))], axis=1
    )

    # Ensure path has the expected number of waypoints
    path_z_np = _ensure_waypoints(path_z_np, n_steps)
    alphas = np.linspace(0.0, 1.0, len(path_z_np))

    # P(win) curve
    path_tensor = torch.tensor(path_z_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        p_win_curve = score_fn(path_tensor).cpu().numpy().astype(np.float32)

    # Crossover
    crossover_wp = first_crossover(p_win_curve)
    crossover_alpha = normalised_crossover_alpha(crossover_wp, len(p_win_curve))

    # Geometry: nearest-win and k-NN distances — supervised subspace only,
    # consistent with KDE metric and path generation.
    win_sup = win_latents[:, :sup_dim]
    z_start_sup = torch.tensor(
        z_start_np[:sup_dim], dtype=torch.float32, device=device
    ).unsqueeze(0)
    z_end_sup = path_tensor[-1, :sup_dim].unsqueeze(0)
    dists_start = torch.cdist(z_start_sup, win_sup).squeeze(0).cpu().numpy()
    dists_end = torch.cdist(z_end_sup, win_sup).squeeze(0).cpu().numpy()

    nearest_start = float(dists_start.min())
    nearest_end = float(dists_end.min())
    knn_start = float(np.sort(dists_start)[:_KNN_K].mean())
    knn_end = float(np.sort(dists_end)[:_KNN_K].mean())

    # KDE density at start and end — evaluated in supervised subspace only,
    # matching how win_kde was fitted in the orchestrator.
    kde_start = float(win_kde.score_samples(z_start_np[:sup_dim].reshape(1, -1))[0])
    kde_end = float(win_kde.score_samples(path_z_np[-1, :sup_dim].reshape(1, -1))[0])

    # Feedback (lightweight — no PNG plots)
    feedback = compute_feedback(
        path_z=path_z_np,
        decode_fn=ctx.guided_vae.model.decode,
        score_fn=score_fn,
        norm_mean=ctx.guided_vae.mean,
        norm_std=ctx.guided_vae.std,
        feature_names=FEATURE_NAMES,
        top_k=top_k,
        method_name=spec.name,
        device=device,
    )
    top_k_raw = tuple(r["feature"] for r in feedback["raw"])
    top_k_mv = tuple(r["feature"] for r in feedback["minimum_viable"])
    top_k_wgt = tuple(r["feature"] for r in feedback["gain_weighted"])

    # Feature-space path length — reuse path_length() on decoded original-scale features.
    x_orig = feedback["_x_orig"]  # (n_waypoints, n_features) original scale
    raw_delta = feedback["_raw_delta"]  # (n_features,) original scale
    feat_path_len = path_length(x_orig)

    # Sparsity: count features where |Δ| > 1σ of training data.
    norm_std_np = ctx.guided_vae.std
    if isinstance(norm_std_np, torch.Tensor):
        norm_std_np = norm_std_np.cpu().numpy()
    else:
        norm_std_np = np.asarray(norm_std_np)
    n_changed = sparsity_n_changed(raw_delta, norm_std_np)

    # Path-averaged KDE log-density — all waypoints, not just the endpoint.
    kde_path_scores = win_kde.score_samples(path_z_np[:, :sup_dim])
    kde_path_mean_val = float(kde_path_scores.mean())

    # Reconstruction cycle consistency: decode(z) → encode → compare with z.
    try:
        with torch.no_grad():
            x_hat_norm = ctx.guided_vae.model.decode(
                path_tensor
            )  # (n_steps, F) normalized
            z_reenc, _ = ctx.guided_vae.model.encode(
                x_hat_norm
            )  # (n_steps, latent_dim)
            cycle_errs = (
                torch.norm(path_tensor[:, :sup_dim] - z_reenc[:, :sup_dim], dim=1)
                .cpu()
                .numpy()
            )
        recon_cycle_mean_val = float(cycle_errs.mean())
        recon_cycle_max_val = float(cycle_errs.max())
    except Exception:
        recon_cycle_mean_val = math.nan
        recon_cycle_max_val = math.nan

    return PathRunResult(
        sample_idx=ctx.chosen,
        method_name=spec.name,
        player_idx=ctx.player_idx,
        path_z=path_z_np,
        p_win_curve=p_win_curve,
        alphas=alphas,
        crossover_alpha=crossover_alpha,
        crossover_wp=crossover_wp,
        success=crossover_alpha is not None,
        p_win_start=float(p_win_curve[0]),
        p_win_end=float(p_win_curve[-1]),
        p_win_gain=float(p_win_curve[-1] - p_win_curve[0]),
        p_win_max=float(p_win_curve.max()),
        auc_p_win=auc_trapezoidal(p_win_curve),
        monotonicity=monotonicity_fraction(p_win_curve),
        dist_to_nearest_win_start=nearest_start,
        dist_to_nearest_win_end=nearest_end,
        dist_to_knn_win_start=knn_start,
        dist_to_knn_win_end=knn_end,
        kde_density_start=kde_start,
        kde_density_end=kde_end,
        path_length_l2=path_length(path_z_np),
        top_k_features_raw=top_k_raw,
        top_k_features_mv=top_k_mv,
        top_k_features_weighted=top_k_wgt,
        wall_time_s=wall_time_s,
        error=None,
        path_length_feature=feat_path_len,
        n_features_changed=n_changed,
        kde_density_path_mean=kde_path_mean_val,
        recon_cycle_error_mean=recon_cycle_mean_val,
        recon_cycle_error_max=recon_cycle_max_val,
    )


def _ensure_waypoints(path: np.ndarray, n: int) -> np.ndarray:
    """Resample path to exactly n waypoints via linear index interpolation."""
    if len(path) == n:
        return path
    indices = np.linspace(0, len(path) - 1, n, dtype=int)
    return path[indices]
