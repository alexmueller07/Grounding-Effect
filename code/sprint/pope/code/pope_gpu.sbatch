#!/bin/bash
#SBATCH --partition=batch
#SBATCH --nodelist=vgi1
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=02:00:00
#SBATCH --nice=250
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --output=/data/alexmueller/sprint/pope/logs/%x_%j.out
#SBATCH --error=/data/alexmueller/sprint/pope/logs/%x_%j.err
set -uo pipefail
W=/data/alexmueller/sprint/pope
export PY=/data/alexmueller/miniconda3/envs/reason/bin/python
export HF_HOME=/data/alexmueller/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$W/code
test -f $W/code/pope_common.py || { echo "FATAL_WRONG_NODE_OR_MISSING_CODE"; exit 9; }
test -f $W/data/coco_pope_random.json || { echo "FATAL_NO_POPE_QUESTIONS"; exit 9; }
test -f $W/data/image_manifest.json || { echo "FATAL_NO_IMAGE_MANIFEST"; exit 9; }
test -d $HF_HOME/hub/models--llava-hf--llava-1.5-7b-hf || { echo "FATAL_NO_MODEL"; exit 9; }
mkdir -p $W/out $W/logs
echo "=== $SLURM_JOB_NAME job $SLURM_JOB_ID host=$(hostname) $(date -u) ==="
eval "$1"
echo "=== end $SLURM_JOB_NAME rc=$? $(date -u) ==="
