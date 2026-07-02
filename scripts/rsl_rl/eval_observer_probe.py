"""E8 — Observer Interpretability Probe.

Three analyses on an ObserverActorModel checkpoint:

  1. Linear probe   — fit D_hat → D_gt; report R² (how predictive is D_hat of D_gt?)
  2. Gradient norms — ||∂μ_π/∂D_hat|| vs ||∂μ_π/∂o_task|| (does policy use D_hat?)
  3. Zeroing test   — re-run eval at k=0..4 with D_hat forced to zeros; report ΔSR

Usage::

    python scripts/rsl_rl/eval_observer_probe.py \\
        --task Isaaclab-RANSv2-Observer-Position-v0 \\
        --checkpoint logs/rsl_rl/Observer_GoToPosition/.../model_4999.pt \\
        --variant OBS_FULL \\
        --output_dir docs/results/e8_observer_probe/OBS_FULL/seed_42 \\
        --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="E8 observer interpretability probe.")
parser.add_argument("--task", type=str, default="Isaaclab-RANSv2-Observer-Position-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--variant", type=str, default="OBS_FULL",
                    help="Label for this checkpoint (e.g. OBS_FULL, ABL_NOMSE)")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--probe_steps", type=int, default=3000,
                    help="Steps to collect (D_hat, D_gt) pairs for the linear probe.")
parser.add_argument("--eval_episodes_per_env", type=int, default=5)
parser.add_argument("--max_failures", type=int, default=4)
parser.add_argument("--pos_tol", type=float, default=0.05)
parser.add_argument("--success_steps", type=int, default=50)
parser.add_argument("--output_dir", type=str, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import json
import os
import statistics

import torch
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Isaaclab_RANSv2.tasks  # noqa: F401


def _force_reset_all(env) -> None:
    base = env.unwrapped
    base.episode_length_buf = torch.full_like(base.episode_length_buf, base.max_episode_length - 1)


def _warmup(env, policy, device):
    """Flush mass timeouts after force-reset."""
    obs = env.get_observations()
    policy.reset()
    with torch.inference_mode():
        obs, _, _, _ = env.step(policy(obs))
    policy.reset()
    return obs


# ── Analysis 1: Linear Probe ──────────────────────────────────────────────────

def collect_dhat_dgt(env, policy, actor, device, num_steps: int, k: int):
    """Collect (D_hat, D_gt) pairs over num_steps steps with force_failure_count=k."""
    task = env.unwrapped.task_api
    task.force_failure_count(k)
    _force_reset_all(env)
    obs = _warmup(env, policy, device)

    d_hats, d_gts = [], []
    with torch.inference_mode():
        for _ in range(num_steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if (dones > 0).any():
                policy.reset(dones)
            d_hat = actor.get_observer_output(obs)
            d_gt = obs["privileged"][:, :actor.degradation_dim]
            d_hats.append(d_hat.cpu())
            d_gts.append(d_gt.cpu())

    task.force_failure_count(None)
    return torch.cat(d_hats, dim=0), torch.cat(d_gts, dim=0)


def linear_probe(d_hat: torch.Tensor, d_gt: torch.Tensor) -> dict:
    """Fit linear regression D_hat → D_gt and return R² per dim + overall."""
    X = d_hat.numpy()
    Y = d_gt.numpy()

    # Simple closed-form linear regression: W = pinv(X.T X) X.T Y
    try:
        from numpy.linalg import lstsq
        W, _, _, _ = lstsq(X, Y, rcond=None)
        Y_pred = X @ W
        ss_res = ((Y - Y_pred) ** 2).sum(axis=0)
        ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
        r2_per_dim = (1 - ss_res / (ss_tot + 1e-12)).tolist()
        r2_overall = float((1 - ss_res.sum() / (ss_tot.sum() + 1e-12)))
        mse_overall = float(((Y - Y_pred) ** 2).mean())
    except Exception as e:
        r2_per_dim = [float("nan")] * d_gt.shape[1]
        r2_overall = float("nan")
        mse_overall = float("nan")
        print(f"[WARN] Linear probe failed: {e}")

    return {
        "r2_overall": r2_overall,
        "r2_per_dim": r2_per_dim,
        "mse_overall": mse_overall,
        "num_samples": len(d_hat),
    }


# ── Analysis 2: Gradient Attribution ─────────────────────────────────────────

def gradient_attribution(env, policy, actor, device, num_steps: int, k: int) -> dict:
    """Measure ||∂μ_π/∂D_hat|| and ||∂μ_π/∂o_task|| at failure count k."""
    task = env.unwrapped.task_api
    task.force_failure_count(k)
    _force_reset_all(env)
    obs = _warmup(env, policy, device)

    grad_norms_dhat, grad_norms_otask = [], []

    # Enable gradients for attribution (override inference_mode)
    for _ in range(num_steps):
        with torch.no_grad():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if (dones > 0).any():
                policy.reset(dones)

        # Gradient pass — build augmented obs manually to isolate gradients
        with torch.enable_grad():
            obs_list = [obs[g] for g in actor.obs_groups]
            policy_obs_raw = torch.cat(obs_list, dim=-1).detach()

            d_hat = actor.get_observer_output(obs).detach().requires_grad_(True)
            o_task = policy_obs_raw.clone().requires_grad_(True)

            if actor.obs_normalization:
                o_task_norm = actor.obs_normalizer(o_task)
            else:
                o_task_norm = o_task

            aug_obs = torch.cat([o_task_norm, d_hat], dim=-1)
            action_mean = actor.mlp(aug_obs)

            loss = action_mean.sum()
            loss.backward()

        if d_hat.grad is not None:
            grad_norms_dhat.append(d_hat.grad.norm(dim=-1).mean().item())
        if o_task.grad is not None:
            grad_norms_otask.append(o_task.grad.norm(dim=-1).mean().item())

    task.force_failure_count(None)
    return {
        "grad_norm_dhat_mean": statistics.mean(grad_norms_dhat) if grad_norms_dhat else float("nan"),
        "grad_norm_otask_mean": statistics.mean(grad_norms_otask) if grad_norms_otask else float("nan"),
        "ratio_dhat_otask": (
            statistics.mean(grad_norms_dhat) / (statistics.mean(grad_norms_otask) + 1e-12)
            if grad_norms_dhat and grad_norms_otask else float("nan")
        ),
        "num_steps": num_steps,
    }


# ── Analysis 3: Zeroing Test ──────────────────────────────────────────────────

def run_eval_zeroed(env, policy, actor, device, k: int, steps_per_k: int,
                    zero_dhat: bool, pos_tol: float, success_steps: int) -> float:
    """Run eval at failure count k with D_hat optionally zeroed. Returns SR."""
    task = env.unwrapped.task_api
    task.force_failure_count(k)
    _force_reset_all(env)
    obs = _warmup(env, policy, device)

    if zero_dhat:
        # Monkey-patch actor.get_latent to zero out D_hat
        original_get_latent = actor.get_latent.__func__

        def get_latent_zeroed(self, obs_td, masks=None, hidden_state=None):
            obs_list = [obs_td[g] for g in self.obs_groups]
            policy_obs = torch.cat(obs_list, dim=-1)
            policy_obs = self.obs_normalizer(policy_obs)
            d_hat_zero = torch.zeros(
                policy_obs.shape[0], self.degradation_dim, device=policy_obs.device
            )
            return torch.cat([policy_obs, d_hat_zero], dim=-1)

        import types
        actor.get_latent = types.MethodType(get_latent_zeroed, actor)

    num_envs = env.num_envs
    pos_dist = task._position_dist.clone()
    sustained = torch.zeros(num_envs, device=device)
    episode_success = torch.zeros(num_envs, dtype=torch.bool, device=device)

    completed_success = []
    with torch.inference_mode():
        for _ in range(steps_per_k):
            prev_pos_dist = pos_dist.clone()
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            done_mask = dones > 0
            if done_mask.any():
                policy.reset(dones)
            pos_dist = task._position_dist.clone()
            effective_pos = pos_dist.clone()
            done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                effective_pos[done_ids] = prev_pos_dist[done_ids]
            in_tol = effective_pos < pos_tol
            sustained = torch.where(in_tol, sustained + 1, torch.zeros_like(sustained))
            episode_success |= sustained >= success_steps
            if done_ids.numel() > 0:
                for i in done_ids.tolist():
                    completed_success.append(bool(episode_success[i].item()))
                episode_success[done_ids] = False
                sustained[done_ids] = 0.0

    if zero_dhat:
        # Restore original get_latent
        del actor.get_latent

    task.force_failure_count(None)
    return statistics.mean(completed_success) if completed_success else float("nan")


# ── Main ─────────────────────────────────────────────────────────────────────

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    checkpoint = args_cli.checkpoint
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    print(f"[Probe] Checkpoint: {checkpoint}")
    print(f"[Probe] Variant:    {args_cli.variant}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    device = agent_cfg.device
    tmp_log_dir = os.path.join(os.path.dirname(checkpoint), "probe_tmp")
    os.makedirs(tmp_log_dir, exist_ok=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=tmp_log_dir, device=device)
    runner.load(checkpoint)

    policy = runner.get_inference_policy(device=device)
    actor = runner.alg.actor
    assert hasattr(actor, "get_observer_output"), (
        "This script requires an ObserverActorModel checkpoint. "
        f"Got actor type: {type(actor).__name__}"
    )

    max_ep_len = env.unwrapped.max_episode_length
    steps_per_k = int(max_ep_len * args_cli.eval_episodes_per_env)
    ks = list(range(args_cli.max_failures + 1))

    out_dir = args_cli.output_dir or os.path.dirname(checkpoint)
    os.makedirs(out_dir, exist_ok=True)

    results = {"variant": args_cli.variant, "checkpoint": checkpoint}

    # ── 1. Linear probe at k=2 (interesting: enough failures to stress the observer) ──
    print("\n[Probe] === Analysis 1: Linear Probe (D_hat → D_gt) at k=2 ===")
    d_hat_k2, d_gt_k2 = collect_dhat_dgt(
        env, policy, actor, device, num_steps=args_cli.probe_steps, k=2
    )
    probe_result = linear_probe(d_hat_k2, d_gt_k2)
    results["linear_probe_k2"] = probe_result
    print(f"  R² overall = {probe_result['r2_overall']:.4f}  "
          f"MSE = {probe_result['mse_overall']:.6f}  "
          f"n = {probe_result['num_samples']}")
    print(f"  R² per dim (first 8 scale dims): "
          f"{[round(x,3) for x in probe_result['r2_per_dim'][:8]]}")

    # Also probe at k=0 (no failures — D_gt is all 1s/0s, observer should predict trivially)
    print("\n[Probe] === Analysis 1b: Linear Probe at k=0 ===")
    d_hat_k0, d_gt_k0 = collect_dhat_dgt(
        env, policy, actor, device, num_steps=args_cli.probe_steps, k=0
    )
    probe_k0 = linear_probe(d_hat_k0, d_gt_k0)
    results["linear_probe_k0"] = probe_k0
    print(f"  R² overall = {probe_k0['r2_overall']:.4f}  "
          f"MSE = {probe_k0['mse_overall']:.6f}")

    # ── 2. Gradient attribution at k=0,2,4 ──────────────────────────────────────
    print("\n[Probe] === Analysis 2: Gradient Attribution ===")
    results["gradient_attribution"] = {}
    for k in [0, 2, 4]:
        print(f"  k={k} ...", end=" ", flush=True)
        grad_result = gradient_attribution(env, policy, actor, device, num_steps=200, k=k)
        results["gradient_attribution"][str(k)] = grad_result
        print(
            f"||∂μ/∂D_hat||={grad_result['grad_norm_dhat_mean']:.4f}  "
            f"||∂μ/∂o_task||={grad_result['grad_norm_otask_mean']:.4f}  "
            f"ratio={grad_result['ratio_dhat_otask']:.4f}"
        )

    # ── 3. Zeroing test at k=0..4 ────────────────────────────────────────────────
    print("\n[Probe] === Analysis 3: D_hat Zeroing Test ===")
    results["zeroing_test"] = {}
    for k in ks:
        sr_normal = run_eval_zeroed(
            env, policy, actor, device, k, steps_per_k,
            zero_dhat=False, pos_tol=args_cli.pos_tol, success_steps=args_cli.success_steps
        )
        sr_zeroed = run_eval_zeroed(
            env, policy, actor, device, k, steps_per_k,
            zero_dhat=True, pos_tol=args_cli.pos_tol, success_steps=args_cli.success_steps
        )
        delta = sr_normal - sr_zeroed
        results["zeroing_test"][str(k)] = {
            "sr_normal": sr_normal,
            "sr_zeroed": sr_zeroed,
            "delta_sr": delta,
        }
        print(
            f"  k={k}  SR_normal={sr_normal*100:.1f}%  "
            f"SR_zeroed={sr_zeroed*100:.1f}%  "
            f"Δ={delta*100:+.1f}pp"
        )

    # ── Save ──────────────────────────────────────────────────────────────────────
    out_path = os.path.join(out_dir, "probe_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Probe] Saved → {out_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
