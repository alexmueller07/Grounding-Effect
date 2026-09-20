#!/bin/bash
# Ablation-ladder lane -- GPU payload runner. Own wrapper on purpose: the shared wrappers end
# in `echo`, so every job through them reports COMPLETED 0:0 however the payload died and
# --dependency=afterok never blocks. Here the payload's status is the job's status.
#SBATCH --job-name=ladgpu
#SBATCH --partition=batch
#SBATCH --nodelist=vgi1
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --nice=150
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --output=/data/alexmueller/sprint/ladder/logs/%x_%j.out
#SBATCH --error=/data/alexmueller/sprint/ladder/logs/%x_%j.err
set -uo pipefail
W=/data/alexmueller/sprint/ladder
export PY=/data/alexmueller/miniconda3/envs/reason/bin/python
export HF_HOME=/data/alexmueller/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
H=$(hostname)
case "$H" in vgi1*) ;; *) echo "FATAL_WRONG_NODE $H"; exit 9;; esac
test -f $W/code/lad_common.py                                    || { echo "FATAL_NO_CODE";   exit 9; }
test -d $HF_HOME/hub/models--llava-hf--llava-1.5-7b-hf           || { echo "FATAL_NO_MODEL";  exit 9; }
test -f /data/datasets/coco/annotations/instances_val2017.json   || { echo "FATAL_NO_COCO";   exit 9; }
test -f /data/alexmueller/j5_gates/code/synonyms.txt             || { echo "FATAL_NO_SYN";    exit 9; }
test -f /data/alexmueller/sc1_interv/out/at_corrob.jsonl         || { echo "FATAL_NO_SIGHTED";exit 9; }
test -w $W/out                                                   || { echo "FATAL_OUT_NOT_WRITABLE"; exit 9; }
echo "=== $SLURM_JOB_NAME job ${SLURM_JOB_ID:-local} host=$H $(date -u) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
df -h / | tail -1
eval "$1" || { echo "LAD_ABORT rc=$?"; exit 9; }
df -h / | tail -1
echo "=== end $SLURM_JOB_NAME $(date -u) ==="
exit 0
