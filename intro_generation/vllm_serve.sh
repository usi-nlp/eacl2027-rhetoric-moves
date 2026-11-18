#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Launch a vLLM OpenAI-compatible server for the open-weight intro generator.
#
# This is the privacy-friendly generation back-end (no closed model sees the
# inputs). It exposes the standard OpenAI Chat Completions API, which the
# generator talks to via `--provider openai_compatible`.
#
# Once the server is up, point the generator at it (from the node that runs
# generate_intros.py):
#     export OPENAI_COMPAT_BASE_URL=http://<server-host>:<PORT>/v1
#     export OPENAI_COMPAT_API_KEY=<the API_KEY below>   # any non-empty string
#     python intro_generation/generate_intros.py \
#         --provider openai_compatible --model "$MODEL" --n 5 --mode basic
#
# Sized for one node with 4× A100 80 GB (320 GB VRAM total): comfortably serves
# a ~32B model (default) and fits a 70B model at TP=4 as well. Override any knob
# with an env var, e.g.  MODEL=meta-llama/Llama-3.3-70B-Instruct TP=4 ./vllm_serve.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# Model to serve. Default: Qwen3-32B (Apache-2.0, strong academic writing,
# widely used, fits easily on 4× A100-80GB). Alternatives that also fit at TP=4:
#   meta-llama/Llama-3.3-70B-Instruct   (heavier, very common baseline)
#   OpenSciLM/Llama-3.1_OpenScholar-8B  (fine-tuned for scientific writing)
MODEL="${MODEL:-Qwen/Qwen3-32B}"
PORT="${PORT:-8000}"
TP="${TP:-4}"                       # tensor-parallel size = number of GPUs
MAX_LEN="${MAX_LEN:-32768}"         # context length; raise if a paper body needs it
GPU_UTIL="${GPU_UTIL:-0.92}"        # fraction of each GPU's memory vLLM may use
API_KEY="${OPENAI_COMPAT_API_KEY:-intro-gen-key}"
HOST="${HOST:-0.0.0.0}"

echo "[vllm_serve] serving '${MODEL}' on ${HOST}:${PORT}  (TP=${TP}, max_len=${MAX_LEN})"
echo "[vllm_serve] visible GPUs: ${CUDA_VISIBLE_DEVICES:-all}"
echo "[vllm_serve] clients set: OPENAI_COMPAT_BASE_URL=http://$(hostname):${PORT}/v1"

exec vllm serve "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --served-model-name "${MODEL}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_LEN}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --dtype auto \
    --api-key "${API_KEY}"
