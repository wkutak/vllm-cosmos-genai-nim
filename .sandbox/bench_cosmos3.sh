#!/usr/bin/env bash
# Measure serving performance of Cosmos3 checkpoints across quantization formats.
#
# Runs each checkpoint through `vllm serve` + `vllm bench serve` using the
# built-in synthetic text+image dataset (random-mm), then collates the results
# into a single CSV of throughput / TTFT / end-to-end latency tradeoffs.
#
# Usage (note: invoke with bash, the repo default shell is tcsh):
#   bash .sandbox/bench_cosmos3.sh
#
# Edit the CONFIG block below to point at your three checkpoint directories.
set -euo pipefail

# ------------------------------------------------------------------ CONFIG ---
# The two checkpoints you actually have on disk:
#   - a pre-quantized FP8 checkpoint (static activation scales, amax-calibrated)
#   - the original BF16 checkpoint
DATA_ROOT="/home/scratch.wkutak_other_1/dev/cosmos3/quantization/data/cosmos3-super"
FP8_STATIC_CKPT="${DATA_ROOT}/fp8"        # modelopt static FP8, amax-calibrated (10 layers kept bf16)
FP8_MIXED_CKPT="${DATA_ROOT}/fp8-mixed"   # same, but divergent layers kept bf16 (26 layers -> fp8+bf16)
BF16_CKPT="${DATA_ROOT}/bf16"             # unquantized reference

# One entry per run: "LABEL|/path/to/checkpoint|extra serve args".
# vLLM auto-detects quantization from a pre-quantized checkpoint (modelopt), so
# fp8_static / fp8_mixed need no flag. For the "dynamic activations" config we
# have no checkpoint, so we let vLLM do online weight->FP8 quantization of the
# BF16 weights via `--quantization fp8`, which uses dynamic (per-token)
# activation scales (fp8 default scheme).
CHECKPOINTS=(
  "fp8_static|${FP8_STATIC_CKPT}|"
  "fp8_dynamic|${BF16_CKPT}|--quantization fp8"
  "fp8_mixed|${FP8_MIXED_CKPT}|"
  "bf16|${BF16_CKPT}|"
)

# Benchmark workload knobs (identical across all checkpoints for a fair compare).
NUM_PROMPTS=512           # total requests sent
MAX_CONCURRENCY=32        # in-flight requests (server load level)
INPUT_LEN=512             # synthetic text prompt length (tokens)
OUTPUT_LEN=256            # tokens to generate per request
# One image per request. (height, width, num_frames): probability
MM_BUCKET_CONFIG='{(720, 1280, 1): 1.0}'
LIMIT_MM_PER_PROMPT='{"image": 1}'

# Server / engine knobs.
PORT=8123
TP_SIZE=1
GPU_MEM_UTIL=0.9
SERVE_READY_TIMEOUT=900  # seconds to wait for the server to come up

# Output.
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cosmos3_bench_out"
CSV_OUT="${OUT_DIR}/cosmos3_perf.csv"

VLLM_BIN="${VLLM_BIN:-/usr/local/bin/vllm}"
PY_BIN="${PY_BIN:-/usr/bin/python}"
# ---------------------------------------------------------------------------

mkdir -p "${OUT_DIR}"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo ">> stopping server (pid ${SERVER_PID})"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup EXIT

wait_for_ready() {
  local deadline=$((SECONDS + SERVE_READY_TIMEOUT))
  until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "!! server process exited before becoming ready; see log" >&2
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "!! timed out waiting for server after ${SERVE_READY_TIMEOUT}s" >&2
      return 1
    fi
    sleep 3
  done
}

for entry in "${CHECKPOINTS[@]}"; do
  IFS='|' read -r LABEL CKPT EXTRA <<< "${entry}"
  echo "================================================================"
  echo ">> benchmarking [${LABEL}] -> ${CKPT}"
  echo "================================================================"

  server_log="${OUT_DIR}/serve_${LABEL}.log"
  result_json="${LABEL}.json"   # bench serve writes into --result-dir

  # shellcheck disable=SC2086
  VLLM_OMNI_USE_QUACK_FP8=0 "${VLLM_BIN}" serve "${CKPT}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}" \
    --max-model-len 146768 \
    ${EXTRA} > "${server_log}" 2>&1 &
  SERVER_PID=$!

  echo ">> waiting for server (pid ${SERVER_PID}), log: ${server_log}"
  wait_for_ready

  echo ">> running bench serve"
  "${VLLM_BIN}" bench serve \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --host 127.0.0.1 --port "${PORT}" \
    --model "${CKPT}" \
    --dataset-name random-mm \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --random-input-len "${INPUT_LEN}" \
    --random-output-len "${OUTPUT_LEN}" \
    --random-mm-base-items-per-request 1 \
    --random-mm-limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}" \
    --random-mm-bucket-config "${MM_BUCKET_CONFIG}" \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --save-result \
    --result-dir "${OUT_DIR}" \
    --result-filename "${result_json}" \
    --metadata label="${LABEL}" checkpoint="${CKPT}"

  cleanup
  echo ">> done [${LABEL}]"
  sleep 5   # let the GPU/port drain before the next run
done

echo ">> collating results into ${CSV_OUT}"
LABELS=()
for entry in "${CHECKPOINTS[@]}"; do LABELS+=("${entry%%|*}"); done
LABEL_CSV="$(IFS=,; echo "${LABELS[*]}")"
"${PY_BIN}" "$(dirname "${BASH_SOURCE[0]}")/collate_cosmos3.py" \
  --result-dir "${OUT_DIR}" --out "${CSV_OUT}" --labels "${LABEL_CSV}"

echo ""
echo ">> CSV written to ${CSV_OUT}"
column -s, -t "${CSV_OUT}" || cat "${CSV_OUT}"
