#!/bin/bash
# Serve each open-weight model on 4 GH200 with vLLM and generate intros against it.
# Submit from the repo root:  sbatch intro_generation/run_vllm_generation.sbatch
#SBATCH --job-name=vllm-intro-gen
#SBATCH --account=<your-account>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=intro_generation/results/slurm_logs/%x_%j.out
#SBATCH --error=intro_generation/results/slurm_logs/%x_%j.out

set -uo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"   # submit from the repo root
cd "$REPO"

# --- environment ----------------------------------------------------------
# Use the uv venv (coherent torch/vLLM/CUDA stack). Do NOT use the conda base
# env — its hand-installed cu13 torch nightly is incompatible with its vLLM build.
source "${VLLM_ENV:?set VLLM_ENV to a virtualenv that has vLLM installed}/bin/activate"
export HF_HOME="${HF_HOME:?set HF_HOME to the Hugging Face cache holding the models}"
# Clear cache-path overrides possibly inherited from the submitting shell.
# huggingface_hub gives HF_HUB_CACHE / *_CACHE precedence over HF_HOME, so a
# stale value would cause an offline cache-miss (LocalEntryNotFoundError) even
# though the models are cached under HF_HOME. This is why a full run submitted
# from an interactive shell failed while a clean submission worked.
unset HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HUB_OFFLINE=1          # models are pre-downloaded; compute node has no internet
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false
# flashinfer 0.6.12 (pinned by vLLM 0.24) JIT-compiles its sampler and is
# incompatible with CUDA 13 → bypass it: native sampler + Triton attention.
# Both are compile-free; verified working on GH200 (Qwen3-4B smoke test).
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=TRITON_ATTN

echo "[env] HF_HOME=$HF_HOME  HF_HUB_CACHE=${HF_HUB_CACHE:-<unset>}  HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-<unset>}  python=$(command -v python)"

mkdir -p intro_generation/results/slurm_logs

# Models to run. Override with a space-separated list to test one, e.g.:
#   sbatch --export=ALL,MODELS_STR=Qwen/Qwen3.5-27B,PAPERS=1 run_vllm_generation.sbatch
if [ -n "${MODELS_STR:-}" ]; then read -ra MODELS <<< "$MODELS_STR"; else
  MODELS=("google/gemma-4-31B-it" "Qwen/Qwen3.5-27B")
fi
MODES=("basic" "cars_guided")
# Paper selection. Default: every paper (all 76). For a quick test, override at
# submit time:  sbatch --export=ALL,PAPERS=3 intro_generation/run_vllm_generation.sbatch
PAPERS="${PAPERS:-all}"
if [ "$PAPERS" = "all" ]; then SELECT=(--all); else SELECT=(--n "$PAPERS"); fi
CONCURRENCY="${CONCURRENCY:-16}"   # parallel requests to vLLM (continuous batching)
PORT=8000

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

for MODEL in "${MODELS[@]}"; do
  SAFE=$(echo "$MODEL" | tr '/' '_')
  VLOG="intro_generation/results/slurm_logs/vllm_${SLURM_JOB_ID}_${SAFE}.log"
  echo "========================================================"
  echo "Serving $MODEL  (tensor-parallel-size 4)  → $VLOG"
  echo "========================================================"

  # Model-specific serve args. Qwen3.5 is a Gated-DeltaNet hybrid whose GDN
  # prefill defaults to flashinfer (JIT-compiles → breaks on CUDA 13). Force the
  # Triton/FLA GDN kernel, which needs no nvcc. Harmless for non-GDN models.
  EXTRA=()
  case "$MODEL" in
    *Qwen3.5*|*Qwen3-Next*) EXTRA+=(--additional-config '{"gdn_prefill_backend": "triton"}') ;;
  esac

  vllm serve "$MODEL" \
      --tensor-parallel-size 4 \
      --port "$PORT" \
      --gpu-memory-utilization 0.90 \
      "${EXTRA[@]}" \
      > "$VLOG" 2>&1 &
  VLLM_PID=$!

  # Wait for the server to accept requests (model load + graph capture can take minutes).
  READY=0
  for i in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "vLLM ready after ~$(( (i-1)*10 ))s"; READY=1; break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "!! vLLM exited during startup — see $VLOG"; break
    fi
    sleep 10
  done

  if [ "$READY" = "1" ]; then
    for MODE in "${MODES[@]}"; do
      echo "----- generating  $MODEL / $MODE  (papers: $PAPERS, concurrency: $CONCURRENCY) -----"
      python intro_generation/generate_intros.py \
          --provider openai_compatible \
          --model "$MODEL" \
          --base-url "http://localhost:${PORT}/v1" \
          --mode "$MODE" \
          "${SELECT[@]}" \
          --concurrency "$CONCURRENCY" \
        || echo "!! generation failed for $MODEL / $MODE"
    done
  fi

  echo "Stopping vLLM for $MODEL"
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
  pkill -u "$USER" -f "vllm serve $MODEL" 2>/dev/null || true
  # Wait for GPU memory to be released before loading the next model.
  for i in $(seq 1 12); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
    echo "  GPU mem used across 4 GPUs: ${USED} MiB"
    [ "${USED:-99999}" -lt 4000 ] && break
    sleep 5
  done
done

echo "ALL_RUNS_DONE — $(date)"
