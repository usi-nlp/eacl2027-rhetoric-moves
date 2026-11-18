#!/bin/bash
# Regenerate the 9 Qwen3.5-27B cars_guided intros that failed in the 2026-07-06
# full run (the model emitted its internal plan as JSON and stopped without the
# "introduction" key — stochastic; papers 001 004 006 025 031 042 067 073 077).
# Serves Qwen3.5-27B with vLLM and retries any paper whose output is still a
# JSON artifact, up to 3 attempts. The resulting run dirs are patched into the
# canonical 2026-07-06_17-24-47 run afterwards with patch_run_papers.py.
# Submit from the repo root:  sbatch intro_generation/regen_qwen_cars_guided.sbatch
#SBATCH --job-name=vllm-qwen-regen
#SBATCH --account=<your-account>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=intro_generation/results/slurm_logs/%x_%j.out
#SBATCH --error=intro_generation/results/slurm_logs/%x_%j.out

set -uo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"   # submit from the repo root
cd "$REPO"

# --- environment (same recipe as run_vllm_generation.sbatch) ---------------
source "${VLLM_ENV:?set VLLM_ENV to a virtualenv that has vLLM installed}/bin/activate"
export HF_HOME="${HF_HOME:?set HF_HOME to the Hugging Face cache holding the models}"
unset HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=TRITON_ATTN

MODEL="Qwen/Qwen3.5-27B"
MODE="cars_guided"
PORT=8000
IDS="1 4 6 25 31 42 67 73 77"

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
mkdir -p intro_generation/results/slurm_logs

VLOG="intro_generation/results/slurm_logs/vllm_${SLURM_JOB_ID}_regen.log"
echo "Serving $MODEL (TP=4) → $VLOG"
vllm serve "$MODEL" \
    --tensor-parallel-size 4 \
    --port "$PORT" \
    --gpu-memory-utilization 0.90 \
    --additional-config '{"gdn_prefill_backend": "triton"}' \
    > "$VLOG" 2>&1 &
VLLM_PID=$!

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
  for ATTEMPT in 1 2 3; do
    echo "----- attempt $ATTEMPT: papers [$IDS] -----"
    python intro_generation/generate_intros.py \
        --provider openai_compatible \
        --model "$MODEL" \
        --base-url "http://localhost:${PORT}/v1" \
        --mode "$MODE" \
        --paper-ids $IDS \
        --concurrency 9 \
      || echo "!! generation call failed on attempt $ATTEMPT"

    # Which of the requested papers are STILL broken in the newest regen run?
    IDS=$(python - <<'PYEOF'
import json, sys
from pathlib import Path
runs = sorted(Path("intro_generation/results").glob("*_Qwen-Qwen3.5-27B_cars_guided"))
latest = runs[-1]
broken = []
for e in json.load(open(latest / "generated_intros.json")):
    t = (e.get("introduction") or "").lstrip()
    if not t or t.startswith("{") or '"internal_plan"' in t:
        broken.append(str(int(e["paper_id"])))
print(" ".join(broken))
print(f"[check] {latest.name}: {len(broken)} still broken", file=sys.stderr)
PYEOF
)
    if [ -z "$IDS" ]; then echo "ALL PAPERS FIXED after attempt $ATTEMPT"; break; fi
    echo "still broken after attempt $ATTEMPT: [$IDS]"
  done
  [ -n "$IDS" ] && echo "STILL_BROKEN_AT_END: [$IDS]"
fi

echo "Stopping vLLM"
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
pkill -u "$USER" -f "vllm serve $MODEL" 2>/dev/null || true

echo "REGEN_DONE — $(date)"
