#!/bin/bash
#SBATCH --job-name=cpw2x2
#SBATCH --partition=batch
#SBATCH --nodelist=vgi1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24000
#SBATCH --time=03:00:00
#SBATCH --output=/data/alexmueller/sc1_interv/logs/%x_%j.out
#SBATCH --error=/data/alexmueller/sc1_interv/logs/%x_%j.err
set -uo pipefail
W=/data/alexmueller/sc1_interv
PY=/data/alexmueller/miniconda3/envs/reason/bin/python
export PYTHONPATH=$W/code:/data/alexmueller/j5_gates/code
test -s $W/code/cpw_2x2_extra.py || { echo FATAL_NO_CODE; exit 9; }
$PY $W/code/cpw_2x2_extra.py || exit 9
