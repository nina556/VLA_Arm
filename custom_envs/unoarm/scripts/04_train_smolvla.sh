#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

# Finetune from smolvla_base. make_policy now forces input/output features from the
# dataset (UnoArm 16-D state/action), so pretrained SO-100 6-D metadata is overridden.
uv run unoarm-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=doki/smolvla_unoarm_v1 \
  --env.type=unoarm \
  --env.task=UnoarmFreeSpace-v0 \
  --dataset.repo_id=doki/unoarm_demo \
  --dataset.root=custom_envs/unoarm/data/unoarm_demo \
  --dataset.video_backend=pyav \
  --rename_map='{"observation.images.top":"observation.images.camera1"}' \
  --policy.use_pointmap=true \
  --policy.pointmap_feature=observation.pointmap \
  --policy.pointmap_norm_radius=0.7 \
  --steps=20000 \
  --batch_size=4 \
  --log_freq=200 \
  --save_freq=2000 \
  --policy.push_to_hub=false \
  --eval.use_async_envs=false \
  --eval.batch_size=1 \
  --eval_steps=0
