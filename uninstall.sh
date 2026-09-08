#!/usr/bin/env bash
# uninstall.sh — remove the personal uncensored Qwen serve.
#
# Usage:
#   ./uninstall.sh              # stop + remove container only (keeps image + weights)
#   ./uninstall.sh --image      # also remove the serving image
#                               # (warn: shared with the stock qwen38-flash server)
#   ./uninstall.sh --models     # also delete /models/uncensored-hybrid (~71 GiB)
#                               # (never touches the shared PLE table)
#   ./uninstall.sh --image --models
set -euo pipefail

NAME="${NAME:-qwen38-uncensored}"
IMAGE="${IMAGE:-qwen38-flash-dgx}"
MODEL_DIR="${MODEL_DIR:-/models/uncensored-hybrid}"

RMI=0; RMMODELS=0
for a in "$@"; do
  case "$a" in
    --image) RMI=1 ;;
    --models) RMMODELS=1 ;;
    -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown flag: $a (try --help)" >&2; exit 1 ;;
  esac
done

if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$NAME"; then
  echo "==> stopping $NAME"
  docker stop "$NAME" >/dev/null 2>&1 || true
  docker rm "$NAME" >/dev/null 2>&1 || true
else
  echo "==> no container $NAME"
fi

if [ "$RMI" = 1 ]; then
  echo "==> removing image $IMAGE (shared with stock server — rebuild via upstream repo if needed)"
  docker rmi "$IMAGE" 2>/dev/null || echo "    image not present"
fi

if [ "$RMMODELS" = 1 ]; then
  if [ -d "$MODEL_DIR" ]; then
    echo "==> deleting $MODEL_DIR"
    rm -rf "$MODEL_DIR"
  else
    echo "==> no model dir $MODEL_DIR"
  fi
  echo "    (PLE table left untouched)"
fi

echo "==> uninstalled"
