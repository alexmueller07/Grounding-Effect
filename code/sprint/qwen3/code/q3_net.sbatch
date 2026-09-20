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
# ONLINE wrapper: the ONLY stage allowed to reach the network is the checkpoint fetch.
export PY=/data/alexmueller/miniconda3/envs/reason/bin/python
export HF_HOME=/data/alexmueller/hf_cache
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=0
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export PYTHONPATH=/data/alexmueller/sprint/qwen3/code
test -f /data/alexmueller/sprint/qwen3/code/q3_common.py || { echo "FATAL_NO_LANE_CODE"; exit 9; }
env | grep -iE "HF_|OFFLINE" | sort
df -h /
echo "=== $SLURM_JOB_NAME job $SLURM_JOB_ID host=$(hostname) $(date -u) ==="
eval "$1" || exit 9
env | grep -iE "HF_|OFFLINE" | sort
df -h /
echo "=== end $SLURM_JOB_NAME rc=$? $(date -u) ==="
