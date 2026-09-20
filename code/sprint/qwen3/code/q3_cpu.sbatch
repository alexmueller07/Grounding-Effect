#!/bin/bash
#SBATCH --partition=batch
#SBATCH --nodelist=vgi1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --nice=250
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --output=/data/alexmueller/sprint/qwen3/logs/%x_%j.out
#SBATCH --error=/data/alexmueller/sprint/qwen3/logs/%x_%j.err
set -uo pipefail
export PY=/data/alexmueller/miniconda3/envs/reason/bin/python
export HF_HOME=/data/alexmueller/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/data/alexmueller/sprint/qwen3/code
test -f /data/alexmueller/sprint/qwen3/code/q3_common.py || { echo "FATAL_NO_LANE_CODE"; exit 9; }
test -f /data/datasets/coco/annotations/instances_val2017.json || { echo "FATAL_NO_COCO"; exit 9; }
echo "=== $SLURM_JOB_NAME job $SLURM_JOB_ID host=$(hostname) $(date -u) ==="
eval "$1" || exit 9
echo "=== end $SLURM_JOB_NAME rc=$? $(date -u) ==="
