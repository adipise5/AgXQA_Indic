#!/usr/bin/env bash
# Submit from the AgXQA-Indic checkout root, for example:
#   sbatch -p <gpu-partition> phase_03_multilingual_finetuning/hpc/slurm_phase3.sbatch
#   SEED=123 sbatch -p <gpu-partition> phase_03_multilingual_finetuning/hpc/slurm_phase3.sbatch
#SBATCH --job-name=agxqa-phase3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/phase_03_multilingual_finetuning/hpc-logs/slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
bash phase_03_multilingual_finetuning/hpc/run_phase3.sh
