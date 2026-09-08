#!/usr/bin/env bash
# serve.sh — serve the abliterated Qwen3.8-Flash-Next hybrid on 1x DGX Spark.
# Usage:
#   ./serve.sh                       # download (if needed) + serve on :8000
#   MODEL_DIR=/path ./serve.sh       # use existing local checkpoint
#   ./serve.sh --stop                # stop the server
set -euo pipefail

MODEL_ID="${MODEL_ID:-bidhata/Qwen3.8-Flash-Next-AutoRound-Uncensored}"
TABLE_ID="${TABLE_ID:-Saren/Qwen3.8-Flash-Next-ple-table-fp8}"
MODEL_DIR="${MODEL_DIR:-/models/uncensored-hybrid}"
TABLE_DIR="${TABLE_DIR:-/models/ple-table-fp8}"
PORT="${PORT:-8000}"
CTX="${CTX:-262144}"
SEQS="${SEQS:-8}"
KV_BYTES="${KV_BYTES:-16g}"
MTP="${MTP:-3}"
IMAGE="${IMAGE:-qwen38-flash-dgx}"
NAME="${NAME:-qwen38-uncensored}"

[[ "${1:-}" == "--stop" ]] && { docker stop "$NAME" 2>/dev/null || true; echo "stopped"; exit 0; }

if [[ ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
  echo ">> downloading $MODEL_ID -> $MODEL_DIR"
  hf download "$MODEL_ID" --local-dir "$MODEL_DIR"
fi
if ! ls "$TABLE_DIR"/*.safetensors >/dev/null 2>&1; then
  echo ">> downloading $TABLE_ID -> $TABLE_DIR"
  hf download "$TABLE_ID" --local-dir "$TABLE_DIR"
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo ">> removing old container $NAME"
  docker stop "$NAME" >/dev/null 2>&1 || true
  docker rm "$NAME" >/dev/null 2>&1 || true
fi

echo ">> serving $MODEL_ID on :$PORT"
docker run -d --name "$NAME" --restart unless-stopped \
  --gpus all --ipc host --shm-size 16g \
  -p "$PORT:8000" \
  -v "$MODEL_DIR:/model" -v "$TABLE_DIR:/ple-table" \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_FP8_HYBRID=1 \
  -e VLLM_USE_DEEP_GEMM=0 \
  -e VLLM_PLE_MMAP=1 \
  -e VLLM_PLE_MMAP_DIR=/ple-table \
  -e VLLM_PLE_MMAP_PREWARM=1 \
  -e VLLM_PLE_MMAP_PREFETCH=0 \
  -e VLLM_PLE_MMAP_WORKERS=32 \
  -e VLLM_PLE_MMAP_MADV_RANDOM=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  "$IMAGE" serve /model \
  --served-model-name qwen --host 0.0.0.0 --port 8000 \
  --load-format fastsafetensors \
  --max-model-len "$CTX" --max-num-seqs "$SEQS" \
  --gpu-memory-utilization 0.01 \
  --enable-prefix-caching --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  -cc.cudagraph_mode=PIECEWISE \
  "-cc.splitting_ops=[\"vllm::unified_attention_with_output\",\"vllm::unified_mla_attention_with_output\",\"vllm::mamba_mixer2\",\"vllm::mamba_mixer\",\"vllm::short_conv\",\"vllm::qwen3_8_flash_next_ple_short_conv\",\"vllm::qwen3_8_flash_next_qsa_with_output\",\"vllm::linear_attention\",\"vllm::qwen_gdn_attention_core\",\"vllm::qwen_gdn_attention_core_fused_norm_packed\",\"vllm::sparse_attn_indexer\",\"vllm::ple_mmap_lookup\"]" \
  --no-enable-flashinfer-autotune \
  --kv-cache-dtype auto --kv-cache-memory-bytes "$KV_BYTES" \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}"

echo ">> boot takes ~4-5 min (fastsafetensors). watch: docker logs -f $NAME"
echo ">> test: curl http://localhost:$PORT/v1/models"
