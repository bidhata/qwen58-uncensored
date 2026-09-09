# Qwen3.8-Flash-Next Uncensored — personal serve

**Author:** Krishnendu Paul &lt;me@krishnendu.com&gt; · https://krishnendu.com

Uncensored Qwen3.8-Flash-Next on a single DGX Spark (GB10), served with vLLM:
OpenAI-compatible API on all interfaces, auto-restart after reboot.

- **Weights:** https://huggingface.co/bidhata/Qwen3.8-Flash-Next-AutoRound-Uncensored
  (gated repo, ~71 GiB — same int4/int8/fp8 hybrid format as the stock recipe,
  drop-in compatible with its serving stack)
- **PLE table (reused, not re-downloaded):**
  `Saren/Qwen3.8-Flash-Next-ple-table-fp8` (~49 GiB, mmapped from local NVMe)
- **Serving stack:** https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound
  (vLLM + PLE mmap + MTP speculative decoding + prefix caching)
- **This box measured:** ~37 tok/s fresh decode, ~44 tok/s cached, ~135 tok/s
  aggregate over 8 streams, TTFT ~0.2–0.9 s, 500 k context working
  (needle retrieved exactly at 320 k tokens)

## Requirements

- NVIDIA DGX Spark / GB10 box, 128 GB unified memory, aarch64
- Docker with the NVIDIA container runtime, `git`, `curl`, `ip`
- `hf` CLI logged in (the weights repo is gated):
  `pip install -U 'huggingface_hub[hf_transfer,cli]' && hf auth login`
- ~80 GiB free disk for the weights (`/models/uncensored-hybrid`)
- One model at a time per box — install stops the stock `qwen38-flash`
  container (two residents exceed the 128 GiB pool and fight for :8000)

## Install

```bash
cd /root/qwen-serve
./install.sh                 # full install + start on :8000 (~5 min boot)
PORT=8001 ./install.sh       # or a different host port
```

What it does:

1. Preflight (docker, `hf` auth against the gated repo, arch, disk space).
2. Stops the stock `qwen38-flash` container if present.
3. Builds the `qwen38-flash-dgx` image from the upstream recipe (skipped if present).
4. Downloads the weights to `/models/uncensored-hybrid` (resumable, skipped if present).
5. Enables docker at boot (`systemctl enable docker`) and starts container
   `qwen38-uncensored` with `--restart unless-stopped` on `0.0.0.0:8000`
   (model id `qwen`, MTP=3, prefix caching on, 262144 context).
6. Waits for the API, runs a smoke completion, prints local + LAN URLs.

It comes back automatically after a system reboot (restart policy + docker at boot).

## Use from the LAN (opencode)

On each other box, merge into `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "dgx-spark": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "DGX Spark (LAN)",
      "options": { "baseURL": "http://10.1.10.41:8000/v1" },
      "models": {
        "qwen": {
          "name": "Qwen3.8-Flash-Next Uncensored (int4 hybrid)",
          "limit": { "context": 500000, "output": 32768 }
        }
      }
    }
  }
}
```

Replace `10.1.10.41` with this box's LAN IP if it changes. Then `/models`
in opencode → `dgx-spark/qwen`. No API key needed (vLLM has no auth).
No client-side tuning needed for throughput — concurrent requests batch
server-side automatically.

## Benchmarks (measured on 1× DGX Spark, GB10)

Config: MTP=3, prefix caching on, SEQS=8, KV 20g, 500k YaRN context unless noted.
Harness: upstream `bench/decode_bench.py` (W1 = fresh 1000-token decode,
W2 = ~8k prefix hit + 256 tokens) and `agg_bench.py` (8×256 unique prompts).

| Workload | Result |
|---|---|
| Fresh decode, single stream (W1) | **37.5 tok/s** median, TTFT ~0.2 s |
| Prefix-cache hit (W2) | **44.2 tok/s** median, TTFT ~0.9 s |
| Aggregate, 8 concurrent streams | **134.6 tok/s** (2048 tokens / 15.2 s wall) |
| Spec decode (MTP=3) | ~2.5–2.9 tok/step, 50–65% accept |
| Long context (YaRN) | needle exact at **320,089 tokens**; ~2800 tok/s prefill |
| 262k stock baseline (for reference) | W1 25.2 tok/s, W2 45.5 tok/s |

Single-run numbers swing >10% on this stack — medians of 3 runs reported.

## Verify / benchmark

```bash
curl http://localhost:8000/v1/models            # model id: qwen
docker logs -f qwen38-uncensored                # boot / health
python3 /root/qwen-serve/agg_bench.py 8 256     # aggregate tok/s, 8 streams
```

## Optional: 500 k context (YaRN)

Validated on the stock checkpoint on this box (needle exact at 320 k tokens,
same-or-better tok/s, no OOM). Same format, so it applies here identically:

```bash
CTX=500000 EXTRA='--hf-overrides {"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}} --speculative-config {"method":"mtp","num_speculative_tokens":3,"max_model_len":500000}' \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 ./install.sh
```

Notes: `install.sh` does not forward `EXTRA` yet — this needs a small patch
to the script (ask before using). Bump opencode `limit.context` to `500000`
when enabled. Do not go to 800 k+ (documented early-OOM).

## Uninstall

```bash
./uninstall.sh              # stop + remove container (keeps image + weights)
./uninstall.sh --image      # also remove image (shared with stock server!)
./uninstall.sh --models     # also delete /models/uncensored-hybrid (~71 GiB)
```

The shared PLE table is never touched.

## Files

| File | Purpose |
|---|---|
| `install.sh` | full install + always-on serve |
| `uninstall.sh` | teardown (`--image`, `--models` flags) |
| `agg_bench.py` | N-stream aggregate tok/s harness |
| `README.md` | this file |

## Contact

Krishnendu Paul — me@krishnendu.com — https://krishnendu.com
