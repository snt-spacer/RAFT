#!/usr/bin/env bash

PYTHON_EXE="${ISAACSIM_ROOT_PATH}/python.sh"

for SEED in 42 1337 7; do
    $PYTHON_EXE scripts/rsl_rl/train.py \
        --task Isaaclab-RANSv2-History-Position-v0 \
        --num_envs 4096 \
        --headless \
        --seed $SEED \
        env.robot_name=CuboThrusterFailureTraining
done