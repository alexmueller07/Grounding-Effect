#!/bin/bash
# Ablation-ladder lane -- CPU payload runner. Own wrapper: the shared wrappers end in `echo`,
# so a job through them reports COMPLETED 0:0 whatever the payload did.
#SBATCH --job-name=ladcpu
#SBATCH --partition=batch
#SBATCH --nodelist=vgi1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --nice=200
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --output=/data/alexmueller/sprint/ladder/logs/%x_%j.out
#SBATCH --error=/data/alexmueller/sprint/ladder/logs/%x_%j.err
set -uo pipefail
W=/data/alexmueller/sprint/ladder
export PY=/data/alexmueller/miniconda3/envs/reason/bin/python
export HF_HOME=/data/alexmueller/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
H=$(hostname)
case "$H" in vgi1*) ;; *) echo "FATAL_WRONG_NODE $H"; exit 9;; esac
test -f $W/code/lad_common.py                                  || { echo "FATAL_NO_CODE";  exit 9; }
test -d $HF_HOME/hub/models--llava-hf--llava-1.5-7b-hf         || { echo "FATAL_NO_MODEL"; exit 9; }
test -f /data/datasets/coco/annotations/instances_val2017.json || { echo "FATAL_NO_COCO";  exit 9; }
test -w $W/out                                                 || { echo "FATAL_OUT_NOT_WRITABLE"; exit 9; }
echo "=== $SLURM_JOB_NAME job ${SLURM_JOB_ID:-local} host=$H $(date -u) ==="
df -h / | tail -1
eval "$1" || exit 9
echo "=== end $SLURM_JOB_NAME $(date -u) ==="
exit 0
