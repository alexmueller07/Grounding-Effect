#!/bin/bash
#SBATCH --partition=batch
#SBATCH --nodelist=vgi1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --nice=250
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --output=/data/alexmueller/sprint/pope/logs/%x_%j.out
#SBATCH --error=/data/alexmueller/sprint/pope/logs/%x_%j.err
set -uo pipefail
W=/data/alexmueller/sprint/pope
export PY=/data/alexmueller/miniconda3/envs/reason/bin/python
export HF_HOME=/data/alexmueller/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$W/code
test -f $W/code/pope_score.py || { echo "FATAL_WRONG_NODE_OR_MISSING_CODE"; exit 9; }
echo "=== $SLURM_JOB_NAME job $SLURM_JOB_ID host=$(hostname) $(date -u) ==="
eval "$1"
echo "=== end $SLURM_JOB_NAME rc=$? $(date -u) ==="
