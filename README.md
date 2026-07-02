# RAFT — Paper Reproduction Guide

Reproduction instructions for the paper
**"Privileged Critic Training Enables Sensor-Free Thruster Fault Adaptation in End-to-End RL"**

The main method is **RAFT** (Recurrent Asymmetric Fault-Tolerant): a GRU-64 actor
trained with an asymmetric PPO critic that receives the ground-truth degradation
vector `D_gt` during training only. Actor sees no fault information at any point.

---

## 1. Repository layout

```
Isaaclab_RANSv2/
├── docker/                       # Docker + docker-compose setup
├── source/Isaaclab_RANSv2/       # Task + environment + agent configs
│   └── .../tasks/direct/isaaclab_ransv2/
│       ├── agents/               # RSL-RL PPO config for every variant
│       ├── environments/         # Env cfgs (Observer, GroundTruth, Vanilla, …)
│       └── tasks_cfg/            # Reward / termination configs
├── scripts/
│   ├── rsl_rl/train.py           # Main training entry point
│   ├── rsl_rl/eval_gt_failures.py         # Standard reset-time evaluation
│   ├── rsl_rl/eval_mid_episode_failures.py# Mid-episode failure injection (E4)
│   └── experiments/              # End-to-end paper experiment scripts (train + eval)
├── paper_checkpoints/            # Compact checkpoint archive (main methods)
│   └── RAFT/{seed_42,seed_7,seed_1337}.pt
└── docs/
    ├── paper_draft.tex
    └── paper_checkpoints/        # Full archive (all methods, all seeds)
```

### Checkpoint locations

- **RAFT** (main method):
  `paper_checkpoints/RAFT/seed_{42,7,1337}.pt` (3 seeds × 1.4 MB each)
- **All paper methods** (VAN, VAN-MLP-AC, RAFT, OBS, OBS-MSE, GT-Oracle, RNN
  ablations, AC ablations, Observer ablations):
  `docs/paper_checkpoints/<METHOD>/seed_{42,7,1337}.pt`

To regenerate the archive from a fresh training run, use
[`scripts/archive_paper_checkpoints.sh`](scripts/archive_paper_checkpoints.sh)
inside the container — it copies the final `model_4999.pt` for every
experiment × seed listed in the script into the archive layout above.

---

## 2. Build

### 2.1 Prerequisites

- Linux host with an NVIDIA GPU (tested on Ampere/Hopper) + recent NVIDIA driver.
- Docker + `docker compose` + NVIDIA Container Toolkit.
- Isaac Sim assets (thruster-actuated robots): downloaded during image build via
  `ASSETS_URL` defined in `docker/Dockerfile.base`.

The training set-up assumes **Isaac Lab (our fork)** and **rsl_rl (our fork)**
live next to this repository:

```
<workspace>/
├── Isaaclab            # SpaceR-x-DreamLab-RL/Isaaclab fork
├── rsl_rl              # rsl_rl fork with asymmetric-critic support
└── Isaaclab_RANSv2     # this repo
```

Clone them side-by-side before building.

### 2.2 Build and enter the container

From `Isaaclab_RANSv2/`:

```bash
# Build the image and start a detached container
docker/container.py start

# Attach an interactive shell (repeat as needed)
docker/container.py enter
```

Inside the container the repo is bind-mounted at `/root/ws/`. All commands
below assume you are running from `/root/ws/` in the container.

### 2.3 Verify the install

```bash
# List all registered Gym tasks
${ISAACSIM_ROOT_PATH}/python.sh scripts/list_envs.py | grep RANSv2
```

You should see the six task IDs used in the paper:
`Isaaclab-RANSv2-Observer-Position-v0`, `Isaaclab-RANSv2-Vanilla-Position-v0`,
`Isaaclab-RANSv2-GroundTruth-Position-v0`, and a few others.

---

## 3. Evaluate the released RAFT checkpoints

The fastest reproduction path — no training needed. Skip to §4 if you want to
retrain from scratch.

### 3.1 Reset-time evaluation (E1, main table)

```bash
${ISAACSIM_ROOT_PATH}/python.sh scripts/rsl_rl/eval_gt_failures.py \
    --task=Isaaclab-RANSv2-Observer-Position-v0 \
    --agent=rsl_rl_rnn_gru64_ac_cfg_entry_point \
    --checkpoint=paper_checkpoints/RAFT/seed_42.pt \
    --num_envs=512 \
    --max_failures=4 \
    --eval_episodes_per_env=10 \
    --pos_tol=0.05 \
    --success_steps=50 \
    --headless \
    --output_dir=docs/results/RAFT_eval/seed_42
```

Sweeps `k = 0..4` mixed-mode failures, writes `eval_gt_failures.json` with the
success rate and final-position error per `k`. Run all three seeds by looping
`seed_42.pt → seed_7.pt → seed_1337.pt`.

The expected numbers (paper Table §V-A):

| k | SR (%) |
|---|:------:|
| 0 | 100.0  |
| 1 | 100.0  |
| 2 |  98.6  |
| 3 |  92.5  |
| 4 |  70.2  |

### 3.2 Mid-episode failure injection (E4)

```bash
${ISAACSIM_ROOT_PATH}/python.sh scripts/rsl_rl/eval_mid_episode_failures.py \
    --task=Isaaclab-RANSv2-Observer-Position-v0 \
    --agent=rsl_rl_rnn_gru64_ac_cfg_entry_point \
    --checkpoint=paper_checkpoints/RAFT/seed_42.pt \
    --num_envs=512 \
    --eval_episodes_per_env=10 \
    --max_failures=4 \
    --inject_step=100 \
    --headless \
    --output_dir=docs/results/RAFT_mid_episode/seed_42
```

### 3.3 Evaluating other methods

Pass a different agent entry point + checkpoint. The mapping between paper
methods and agent entry points is:

| Paper method   | Checkpoint dir                     | `--agent=`                              |
|----------------|------------------------------------|-----------------------------------------|
| **RAFT**       | `paper_checkpoints/RAFT/`          | `rsl_rl_rnn_gru64_ac_cfg_entry_point`   |
| VAN            | `docs/paper_checkpoints/VAN/`      | `rsl_rl_cfg_entry_point`                |
| VAN-MLP-AC     | `docs/paper_checkpoints/VAN_MLP_AC/`| `rsl_rl_van_ac_cfg_entry_point`        |
| Oracle (GT)    | `docs/paper_checkpoints/GT_ORACLE/`| `rsl_rl_gt_observer_cfg_entry_point`    |
| OBS (λ=0)      | `docs/paper_checkpoints/OBS/`      | `rsl_rl_cfg_entry_point`                |
| OBS-MSE (λ=1)  | `docs/paper_checkpoints/OBS_MSE/`  | `rsl_rl_cfg_entry_point`                |
| GRU-256-AC     | `docs/paper_checkpoints/GRU256_AC/`| `rsl_rl_rnn_gru256_ac_cfg_entry_point`  |
| LSTM-64-AC     | `docs/paper_checkpoints/LSTM64_AC/`| `rsl_rl_rnn_lstm64_ac_cfg_entry_point`  |
| LSTM-256-AC    | `docs/paper_checkpoints/LSTM256_AC/`| `rsl_rl_rnn_lstm256_ac_cfg_entry_point`|

For the VAN, GRU/LSTM (no-AC) variants, use `Isaaclab-RANSv2-Vanilla-Position-v0`
as the `--task`. For GT-Oracle, use `Isaaclab-RANSv2-GroundTruth-Position-v0`.
Every other method uses `Isaaclab-RANSv2-Observer-Position-v0`.

---

## 4. Retrain a method from scratch

RAFT (single seed):

```bash
${ISAACSIM_ROOT_PATH}/python.sh scripts/rsl_rl/train.py \
    --task=Isaaclab-RANSv2-Observer-Position-v0 \
    --agent=rsl_rl_rnn_gru64_ac_cfg_entry_point \
    --num_envs=4096 \
    --headless \
    --seed=42 \
    agent.experiment_name=VAN_GRU64_AC_GoToPosition
```

Training uses 5 000 PPO iterations × 24 steps × 4 096 envs
(≈ 5·10⁸ env steps). On a single A100/H100 this takes ~4–6 hours per seed.
Checkpoints land in `logs/rsl_rl/VAN_GRU64_AC_GoToPosition/<timestamp>_seed_42/`.

Swap the agent entry-point and task to retrain any other method — see the table
in §3.3. All paper policies use the same PPO hyperparameters
(`num_learning_epochs=5`, `num_mini_batches=8`, `learning_rate=3e-4`,
`entropy_coef=0.005`, `clip_param=0.2`), differing only in actor architecture
and observation-group wiring.

---

## 5. Reproduce every paper experiment (E1–E6)

Each of the six paper experiments has a self-contained shell driver in
`scripts/experiments/` that trains all required seeds and runs the evaluation.
The mapping between paper labels and driver scripts:

| Paper section | Driver script                                          | Purpose |
|---------------|--------------------------------------------------------|---------|
| **E1** — main comparison           | `e1_main_comparison.sh`                    | VAN vs Oracle vs OBS/OBS-MSE vs RAFT, `k=0..4` mixed modes |
| **E2** — severity sweep            | `e3_severity_sweep.sh`                     | DEG scale + STK offset swept at `k=1` |
| **E3** — multi-failure scalability | `e4_multifailure_scalability.sh` + `e3_raft_eval.sh` | Per-mode `k=0..4` for RAFT / OBS / Oracle |
| **E4** — mid-episode injection     | `e9_mid_episode_failures.sh` + `e8_raft_mid_episode.sh` | Failures atomically injected at step 100 |
| **E5** — recurrent-policy ablation | `e10_rnn_ablation.sh`                      | GRU/LSTM 64/256 **without** AC |
| **E6** — asymmetric-critic ablation| `e11_ac_ablation.sh`                       | VAN-MLP / GRU / LSTM **with** AC |

Run them from the repo root inside the container:

```bash
bash scripts/experiments/e1_main_comparison.sh
bash scripts/experiments/e3_severity_sweep.sh
bash scripts/experiments/e4_multifailure_scalability.sh
bash scripts/experiments/e3_raft_eval.sh
bash scripts/experiments/e9_mid_episode_failures.sh
bash scripts/experiments/e8_raft_mid_episode.sh
bash scripts/experiments/e10_rnn_ablation.sh
bash scripts/experiments/e11_ac_ablation.sh
```

Every driver:

1. Trains all required variants × 3 seeds (skipping ones that already have a
   checkpoint in `logs/rsl_rl/`).
2. Runs `eval_gt_failures.py` / `eval_mid_episode_failures.py` /
   `eval_severity.py` at 512 envs × 10 episodes per condition.
3. Writes per-seed JSON to `docs/results/<exp_name>/` and prints an
   aggregated `mean ± std` table to `summary.txt`.

Total wall-time to reproduce the full paper on a single A100/H100:
~72–96 hours (dominated by 5 000-iter training runs).

### 5.1 Regenerate plots

Once results are in `docs/results/`, generate the paper figures with:

```bash
${ISAACSIM_ROOT_PATH}/python.sh scripts/experiments/plot_e1_main.py
${ISAACSIM_ROOT_PATH}/python.sh scripts/experiments/plot_e3_severity.py
${ISAACSIM_ROOT_PATH}/python.sh scripts/experiments/plot_e4_multifailure.py
${ISAACSIM_ROOT_PATH}/python.sh scripts/experiments/plot_e9_mid_episode.py
${ISAACSIM_ROOT_PATH}/python.sh scripts/experiments/plot_e10_rnn_ablation.py
${ISAACSIM_ROOT_PATH}/python.sh scripts/experiments/plot_e11_ac_ablation.py
```

Figures are written to `docs/figures/`.

---

## 6. Task registry cheat-sheet

The three environments used in the paper differ only in what appears in
`obs["policy"]`, `obs["privileged"]`, and `obs["history"]`:

| Gym ID                                        | `obs["policy"]` | `obs["privileged"]` | `obs["history"]` |
|-----------------------------------------------|:---------------:|:-------------------:|:----------------:|
| `Isaaclab-RANSv2-Vanilla-Position-v0`         | 15-dim task obs | —                   | —                |
| `Isaaclab-RANSv2-Observer-Position-v0`        | 15-dim task obs | 16-dim `D_gt`       | 32×15 buffer     |
| `Isaaclab-RANSv2-GroundTruth-Position-v0`     | 15+16 (obs ‖ `D_gt`) | —              | —                |

Which slots each variant actually consumes is set by `obs_groups` in the
matching PPO cfg under `source/.../agents/rsl_rl_ppo-*.py`.

---

## 7. Citing

If you use RAFT or this codebase, please cite:

```bibtex
@article{castan2026raft,
  title   = {Privileged Critic Training Enables Sensor-Free Thruster Fault
             Adaptation in End-to-End RL},
  author  = {Castan, Ricard M. and Olivares-Mendez, Miguel A.},
  year    = {2026},
}
```

## 8. Licence

BSD-3-Clause (inherited from Isaac Lab). See `LICENSE`.
