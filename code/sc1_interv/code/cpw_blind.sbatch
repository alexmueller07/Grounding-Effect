#!/bin/bash
#SBATCH --job-name=cpwblind
#SBATCH --partition=batch
#SBATCH --nodelist=vgi1
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --nice=120
#SBATCH --output=/data/alexmueller/sc1_interv/logs/%x_%j.out
#SBATCH --error=/data/alexmueller/sc1_interv/logs/%x_%j.err
set -uo pipefail
W=/data/alexmueller/sc1_interv
export PY=/data/alexmueller/miniconda3/envs/reason/bin/python
export HF_HOME=/data/alexmueller/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$W/code:/data/alexmueller/j5_gates/code
H=$(hostname); case "$H" in vgi1*) ;; *) echo "FATAL_WRONG_NODE $H"; exit 9;; esac
test -f $W/code/wia_cpw_gen_blind.py || { echo FATAL_NO_CODE; exit 9; }
test -f $W/out/corrobpw_cells.json  || { echo FATAL_NO_CELLS; exit 9; }
mkdir -p $W/logs
echo "=== $SLURM_JOB_NAME job $SLURM_JOB_ID host=$H block=$BLOCK $(date -u) ==="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
eval "$1" || { echo "CPWB_ABORT rc=$?"; exit 9; }
echo "CPWB_SENTINEL_OK block=$BLOCK"
