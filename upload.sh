#!/usr/bin/env bash
# upload.sh — upload the staged model to Hugging Face.
# Usage: ./upload.sh [REPO_ID]   (default: bidhata/Qwen3.8-Flash-Next-AutoRound-Uncensored)
# Creates a gated repo if it doesn't exist, then uploads /root/upload/model.
set -euo pipefail
REPO_ID="${1:-${MODEL_ID:-bidhata/Qwen3.8-Flash-Next-AutoRound-Uncensored}}"
STAGE="${STAGE:-/root/upload/model}"

python3 - "$REPO_ID" <<'EOF'
import sys
from huggingface_hub import HfApi
repo = sys.argv[1]
api = HfApi()
api.create_repo(repo, repo_type="model", exist_ok=True)
print("repo created (gated):", repo)
EOF

hf upload "$REPO_ID" "$STAGE" --repo-type model
echo ">> uploaded: https://huggingface.co/$REPO_ID"
