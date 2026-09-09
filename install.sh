#!/usr/bin/env bash
# install.sh — personal install for the uncensored Qwen3.8-Flash-Next hybrid.
#
# Model:  bidhata/Qwen3.8-Flash-Next-AutoRound-Uncensored (gated, ~71 GiB)
# Table:  Saren/Qwen3.8-Flash-Next-ple-table-fp8 (reused, ~49 GiB)
# Stack:  Saren-Arterius/qwen3.8-Flash-DGX-AutoRound (vLLM + PLE mmap + MTP)
#
# Result: container "qwen38-uncensored" serving OpenAI-compatible API on
# 0.0.0.0:8000 (all interfaces, LAN-reachable), with
# --restart unless-stopped + docker enabled at boot, so it comes back
# automatically after a system reboot. A Docker healthcheck reports state.
#
# One model at a time per box: install stops the stock "qwen38-flash"
# container (two residents exceed the 128 GiB unified pool and fight for :8000).
#
# Usage:
#   ./install.sh                  # full install + start
#   PORT=8001 ./install.sh        # serve on a different host port
#   FORCE=1 ./install.sh          # reinstall even if a healthy container is up
#
# Extra knobs (env):
#   EXTRA='...'                       # args appended verbatim to the vLLM command
#   VLLM_ALLOW_LONG_MAX_MODEL_LEN=1   # for contexts beyond the checkpoint default
#   STACK_REF=<commit|tag|branch>     # pin the serving stack for a reproducible build
set -euo pipefail

MODEL_ID="${MODEL_ID:-bidhata/Qwen3.8-Flash-Next-AutoRound-Uncensored}"
TABLE_ID="${TABLE_ID:-Saren/Qwen3.8-Flash-Next-ple-table-fp8}"
MODEL_DIR="${MODEL_DIR:-/models/uncensored-hybrid}"
TABLE_DIR="${TABLE_DIR:-/models/ple-table-fp8}"
PORT="${PORT:-8000}"
CTX="${CTX:-262144}"
SEQS="${SEQS:-8}"
KV_BYTES="${KV_BYTES:-20g}"
MTP="${MTP:-3}"
IMAGE="${IMAGE:-qwen38-flash-dgx}"
NAME="${NAME:-qwen38-uncensored}"
STOCK_NAME="${STOCK_NAME:-qwen38-flash}"
STACK_REPO="${STACK_REPO:-https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound.git}"
STACK_DIR="${STACK_DIR:-/root/qwen3.8-Flash-DGX-AutoRound}"
STACK_REF="${STACK_REF:-}"
FORCE="${FORCE:-0}"
EXTRA="${EXTRA:-}"
VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-0}"

have() { command -v "$1" >/dev/null 2>&1; }
die() { echo "error: $*" >&2; exit 1; }
say() { echo "==> $*"; }

# Auto-sudo only when not already root; empty otherwise.
SUDO=""
[ "$(id -u)" -eq 0 ] || SUDO="sudo"

STARTED_CONTAINER=0
cleanup_on_fail() {
  local ec=$?
  [ "$ec" -eq 0 ] && exit 0
  if [ "$STARTED_CONTAINER" = 1 ]; then
    echo "error: install failed (exit $ec) — last logs from $NAME:" >&2
    docker logs --tail 40 "$NAME" 2>&1 | sed 's/^/    /' >&2 || true
    echo "error: removing half-started container $NAME" >&2
    docker rm -f "$NAME" >/dev/null 2>&1 || true
  fi
  exit "$ec"
}
trap cleanup_on_fail EXIT

api_up() { curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; }
container_running() { [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = "true" ]; }

# --- preflight ------------------------------------------------------------
have docker || die "docker not found"
docker info >/dev/null 2>&1 || die "docker daemon not reachable (try: sudo systemctl start docker)"
docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -qi 'nvidia' \
  || docker info 2>/dev/null | grep -qi 'nvidia' \
  || die "NVIDIA container runtime not visible to docker (need nvidia-container-toolkit for --gpus all)"
have hf || die "hf CLI not found (pip install -U 'huggingface_hub[hf_transfer,cli]' && hf auth login)"
[ "$(uname -m)" = "aarch64" ] || die "this stack targets aarch64/GB10, found $(uname -m)"

# --- already-healthy short-circuit -------------------------------------
if [ "$FORCE" != 1 ] && container_running && api_up; then
  say "$NAME already running and answering on :$PORT — nothing to do (FORCE=1 to reinstall)"
  trap - EXIT
  exit 0
fi

# Gated repo: fail fast with a clear message instead of mid-download.
hf download "$MODEL_ID" --local-dir /tmp/.hf-auth-probe --include "config.json" >/dev/null 2>&1 \
  || die "cannot access $MODEL_ID — run: hf auth login (repo is gated)"
rm -rf /tmp/.hf-auth-probe

# ~71 GiB for the model (table is reused, not re-downloaded).
avail_gib="$(df -B1 --output=avail "$(dirname "$MODEL_DIR")" 2>/dev/null | tail -1 | awk '{printf "%d", $1/1073741824}')"
[ "$avail_gib" -ge 80 ] || die "only ${avail_gib} GiB free, need ~80 GiB for the uncensored checkpoint"

# --- one model at a time --------------------------------------------------
if docker ps -a --format '{{.Names}}' | grep -qx "$STOCK_NAME"; then
  say "stopping stock container $STOCK_NAME (one model at a time per box)"
  docker stop "$STOCK_NAME" >/dev/null 2>&1 || true
  docker rm "$STOCK_NAME" >/dev/null 2>&1 || true
fi

# --- serving image --------------------------------------------------------
if [ "$FORCE" != 1 ] && docker image inspect "$IMAGE" >/dev/null 2>&1; then
  say "image $IMAGE already present, skipping build (FORCE=1 to rebuild)"
else
  if [ ! -d "$STACK_DIR/.git" ]; then
    say "cloning serving stack"
    git clone "$STACK_REPO" "$STACK_DIR"
  fi
  if [ -n "$STACK_REF" ]; then
    say "pinning serving stack to $STACK_REF"
    git -C "$STACK_DIR" fetch --tags --force origin "$STACK_REF" 2>/dev/null \
      || git -C "$STACK_DIR" fetch --tags --force origin
    git -C "$STACK_DIR" checkout -q "$STACK_REF"
  fi
  say "building $IMAGE ($(git -C "$STACK_DIR" rev-parse --short HEAD 2>/dev/null || echo upstream), one-time, ~10 min)"
  docker build -t "$IMAGE" "$STACK_DIR"
fi

# --- weights --------------------------------------------------------------
if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
  say "downloading $MODEL_ID -> $MODEL_DIR (~71 GiB, resumable)"
  hf download "$MODEL_ID" --local-dir "$MODEL_DIR"
else
  say "model present at $MODEL_DIR, skipping download"
fi
if ! ls "$TABLE_DIR"/*.safetensors >/dev/null 2>&1; then
  say "downloading $TABLE_ID -> $TABLE_DIR (~49 GiB, resumable)"
  hf download "$TABLE_ID" --local-dir "$TABLE_DIR"
else
  say "PLE table present at $TABLE_DIR, skipping download"
fi

# --- survive reboot -----------------------------------------------------
# Container restart policy handles the reboot; the daemon must start at boot.
if have systemctl; then
  $SUDO systemctl enable docker >/dev/null 2>&1 \
    && say "docker enabled at boot" \
    || say "warn: could not enable docker at boot (run manually: sudo systemctl enable docker)"
else
  say "warn: no systemctl — ensure the docker daemon starts at boot yourself"
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  say "removing old container $NAME"
  docker stop "$NAME" >/dev/null 2>&1 || true
  docker rm "$NAME" >/dev/null 2>&1 || true
fi

# --- serve (all interfaces, always restart, healthcheck) ---------------
say "starting $NAME on :$PORT (bind 0.0.0.0, restart unless-stopped)"
[ -n "$EXTRA" ] && say "appending EXTRA to the vLLM command: $EXTRA"
# shellcheck disable=SC2086
docker run -d --name "$NAME" --restart unless-stopped \
  --gpus all --ipc=host --shm-size 16g -p "${PORT}:8000" \
  --health-cmd 'curl -fsS http://127.0.0.1:8000/v1/models || exit 1' \
  --health-interval 30s --health-start-period 300s --health-timeout 5s --health-retries 3 \
  -v "$MODEL_DIR:/model:ro" -v "$TABLE_DIR:/ple-table:ro" \
  -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_WORKERS=32 -e VLLM_PLE_MMAP_PREWARM=1 -e VLLM_PLE_MMAP_PREFETCH=0 \
  -e VLLM_PLE_MMAP_MADV_RANDOM=0 \
  -e VLLM_PLE_MMAP_DIR=/ple-table \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_FP8_HYBRID=1 \
  -e VLLM_USE_DEEP_GEMM=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN="$VLLM_ALLOW_LONG_MAX_MODEL_LEN" \
  "$IMAGE" \
  /model --served-model-name qwen \
    --host 0.0.0.0 --port 8000 --load-format fastsafetensors \
    --max-model-len "$CTX" --max-num-seqs "$SEQS" --gpu-memory-utilization 0.01 \
    --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192 \
    -cc.cudagraph_mode=PIECEWISE \
    "-cc.splitting_ops=[\"vllm::unified_attention_with_output\",\"vllm::unified_mla_attention_with_output\",\"vllm::mamba_mixer2\",\"vllm::mamba_mixer\",\"vllm::short_conv\",\"vllm::qwen3_8_flash_next_ple_short_conv\",\"vllm::qwen3_8_flash_next_qsa_with_output\",\"vllm::linear_attention\",\"vllm::qwen_gdn_attention_core\",\"vllm::qwen_gdn_attention_core_fused_norm_packed\",\"vllm::sparse_attn_indexer\",\"vllm::ple_mmap_lookup\"]" \
    --no-enable-flashinfer-autotune \
    --kv-cache-dtype auto --kv-cache-memory-bytes "$KV_BYTES" \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}" \
    ${EXTRA}
STARTED_CONTAINER=1

# --- wait + smoke -------------------------------------------------------
say "waiting for API (boot takes ~5 min with fastsafetensors)"
i=0
until api_up; do
  docker ps --format '{{.Names}} {{.Status}}' --filter "name=$NAME" | grep -q . \
    || die "container $NAME exited — see: docker logs $NAME"
  [ "$i" -ge 900 ] && die "no API after 15 min — see: docker logs -f $NAME"
  sleep 5; i=$((i+5))
done
say "API responding, smoke test:"
curl -fsS --max-time 300 "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Reply with exactly: uncensored serve OK"}],"max_tokens":32,"temperature":0}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'

STARTED_CONTAINER=0   # past the fragile part — keep the container on any later error
trap - EXIT

lanip="$(ip -4 -brief addr show scope global 2>/dev/null | awk '{split($3,a,"/"); print a[1]; exit}')"
say "done — model id 'qwen' serving (context $CTX):"
echo "    local: http://127.0.0.1:${PORT}/v1"
[ -n "$lanip" ] && echo "    LAN:   http://${lanip}:${PORT}/v1"
echo "    Health: docker ps --filter name=${NAME}   (STATUS shows healthy/unhealthy)"
echo "    It restarts automatically after reboot (restart policy + docker at boot)."
