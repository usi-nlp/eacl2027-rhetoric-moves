#!/bin/bash
# ---------------------------------------------------------------------------
# SLURM job: serve an open-weight model with vLLM on ONE node of 4× A100 80GB,
# for the AI introduction generator (Step 7, open-weight / privacy-friendly
# condition).
#
# Submit:   sbatch intro_generation/vllm_serve.slurm
# Override: MODEL=meta-llama/Llama-3.3-70B-Instruct sbatch intro_generation/vllm_serve.slurm
#
# The server logs its hostname; from the node running generate_intros.py set
#   export OPENAI_COMPAT_BASE_URL=http://<that-host>:8000/v1
#   export OPENAI_COMPAT_API_KEY=intro-gen-key
# then run with --provider openai_compatible --model "$MODEL".
#
# NOTE: partition / account / GPU-request syntax vary by cluster — adjust the
# three lines marked below to match your site before submitting, and create
# the log folder first:  mkdir -p intro_generation/results/slurm_logs
# ---------------------------------------------------------------------------
#SBATCH --job-name=vllm-intro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4               # 4× A100 80GB on one node
##SBATCH --gres=gpu:a100:4              # alternative GPU request syntax (some clusters)
#SBATCH --cpus-per-task=32
#SBATCH --mem=0                         # all memory on the node
#SBATCH --time=12:00:00
#SBATCH --output=intro_generation/results/slurm_logs/vllm-%j.out
#SBATCH --error=intro_generation/results/slurm_logs/vllm-%j.err
# Set these for your cluster ↓
#SBATCH --account=<your-account>
##SBATCH --partition=normal
##SBATCH --constraint=gpu

set -euo pipefail

# --- environment ----------------------------------------------------------
# Conda env that has vLLM installed (adjust the name).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nlp

# HF cache on scratch so 30–140 GB of weights don't fill $HOME (adjust path).
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
# export HF_TOKEN=...                   # needed for gated models (e.g. Llama)

export MODEL="${MODEL:-Qwen/Qwen3-32B}"
export PORT="${PORT:-8000}"
export TP=4
export OPENAI_COMPAT_API_KEY="${OPENAI_COMPAT_API_KEY:-intro-gen-key}"

echo "==============================================================="
echo " vLLM intro-generation server"
echo " node      : $(hostname)"
echo " model     : ${MODEL}  (TP=${TP}, 4× A100 80GB)"
echo " endpoint  : http://$(hostname):${PORT}/v1"
echo " client env: export OPENAI_COMPAT_BASE_URL=http://$(hostname):${PORT}/v1"
echo "             export OPENAI_COMPAT_API_KEY=${OPENAI_COMPAT_API_KEY}"
echo "==============================================================="

# Launch the server (runs until the job is cancelled or the wall-time ends).
srun bash "$(dirname "$0")/vllm_serve.sh"
