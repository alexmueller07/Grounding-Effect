#!/bin/bash
#SBATCH --partition=batch
#SBATCH --nodelist=vgi1
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --nice=250
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --output=/data/alexmueller/sprint/qwen3/logs/%x_%j.out
#SBATCH --error=/data/alexmueller/sprint/qwen3/logs/%x_%j.err
set -uo pipefail
export PY=/data/alexmueller/miniconda3/envs/reason/bin/python
export HF_HOME=/data/alexmueller/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/data/alexmueller/sprint/qwen3/code
test -f /data/alexmueller/sprint/qwen3/code/q3_common.py || { echo "FATAL_NO_LANE_CODE"; exit 9; }
test -f /data/alexmueller/sprint/qwen3/code/j5_common.py || { echo "FATAL_NO_J5_COPY"; exit 9; }
test -f /data/datasets/coco/annotations/instances_val2017.json || { echo "FATAL_NO_COCO"; exit 9; }
test -d $HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct || { echo "FATAL_NO_QWEN3_MODEL"; exit 9; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
free -g | head -2
echo "=== $SLURM_JOB_NAME job $SLURM_JOB_ID host=$(hostname) $(date -u) ==="
eval "$1" || exit 9
echo "=== end $SLURM_JOB_NAME rc=$? $(date -u) ==="
