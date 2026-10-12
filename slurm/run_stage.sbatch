#!/bin/bash
#SBATCH --job-name=position-bias
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

mkdir -p logs
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

# Example:
# sbatch --export=ALL,STAGE=main,MODELS=qwen2.5-7b,DATASETS=fineweb-edu slurm/run_stage.sbatch
STAGE="${STAGE:-main}"
MODELS="${MODELS:-all}"
DATASETS="${DATASETS:-all}"

uv sync --frozen
uv run python reproduce.py run "$STAGE" --models "$MODELS" --datasets "$DATASETS"
