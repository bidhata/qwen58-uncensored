# Qwen3.8-Flash-Next Uncensored Hybrid — DGX Spark

**Author:** Krishnendu Paul &lt;me@krishnendu.com&gt;

Uncensored Qwen3.8-Flash-Next on a **single DGX Spark** at **full hybrid
speed** (~32 tok/s decode, ~230 ms TTFT) — no VRAM-hungry rebuild, no
360 GB download, no requantization pipeline.

Forked from
[Saren-Arterius/qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound):
same int4/int8/fp8 checkpoint format, same vLLM serving stack, same PLE
mmap table — with the safety refusal direction surgically removed
**in-quant**, directly inside the quantized weights.

## Specialty — why this fork exists

Everyone else's uncensored Flash-Next costs you something: the NVFP4 build
runs at ~half speed, GGUF via llama.cpp loses MTP speculative decoding, and
the FP8 build doesn't fit on one Spark at all. Rebuilding the fast hybrid
from abliterated BF16 weights would mean re-running the entire AutoRound
quantization — slow, heavy, and quality-risky.

This fork takes a different route: **refusal-direction transfer**. The
abliteration in
[orcarouter/Qwen3.8-Flash-Next-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored)
turns out to be a single clean rank-1 edit, so instead of rebuilding
anything, we extract that one direction from a ~2 GB shard pair and apply it
to the already-quantized hybrid:

- Downloaded **one** 1.9 GB shard (stock) + same shard (abliterated) — not
  the 360 GB checkpoint.
- Stock-vs-abliterated diff is cleanly **rank-1** (top singular value
  ~100× the second) with the **identical vector (cos = 1.0000)** across
  attention o_proj, GDN out_proj, and shared-expert down_proj, at ~2%
  relative magnitude. Gates, inputs, embeddings: untouched.
- Applied `W − r(rᵀW)` to all **25,185** residual-writing matrices in place:
  96 fp8 (12 QSA o_proj + 36 GDN out_proj + 48 shared-expert down_proj),
  24,576 int4 MoE-expert down_proj, 513 bf16 MTP draft-layer tensors.
- Key subtlety the naive approach misses: a 2% edit **vanishes** under
  int4 requantization with frozen scales (only ~500 of 1.6M values flip per
  matrix). The fix is fresh per-column symmetric scales recomputed from the
  edited weights — 86% of values then carry the edit, zero clamps.
- Result is bit-compatible with the stock serving stack: same files, same
  shapes, same dtypes, same `quantization_config`. Drop-in restart, no
  image or config changes.

## The model

| Component | Precision | Notes |
|---|---|---|
| 512-expert MoE, 48 layers (6B active) | int4 GPTQ-Marlin g128, sym | abliterated, fresh scales |
| lm_head (248k vocab, shared w/ MTP) | int8 GPTQ-Marlin | untouched (read-only) |
| GDN in/out, QSA q/k/v/o, shared expert | fp8 blockwise e4m3 128×128 | writers abliterated |
| Embeddings, norms, gates, hyper-connections | bf16 | untouched |
| MTP draft layer | bf16 | abliterated |
| 51B PLE n-gram table | fp8, mmapped from NVMe | unchanged, reused |

Weights: `bidhata/Qwen3.8-Flash-Next-AutoRound-Uncensored` (71 GB, gated).
Table: `Saren/Qwen3.8-Flash-Next-ple-table-fp8` (49 GB).

## Measured (1× DGX Spark, GB10, vLLM + MTP=3)

- **Refusals gone:** 5/5 probes comply (lockpicking guide, phishing email,
  smoke bomb, exam cheating, targeted roast) — coherent, detailed output,
  zero refusal phrases.
- **Capability intact:** clean O(n) Fibonacci with tests, long-form essay,
  math reasoning with work shown.
- **Speed preserved:** decode ~32 tok/s, overall ~31 tok/s, TTFT ~230 ms —
  same as the stock hybrid within run-to-run variance.

## Quickstart

Needs: DGX Spark / GB10 box, 128 GB unified memory, Docker + NVIDIA
runtime, `hf` CLI (logged in), ~130 GB free disk.

```bash
# 1. serving image (Saren's recipe: vLLM Flash-Next + PLE/FP8/MTP patches)
git clone https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound.git
docker build -t qwen38-flash-dgx qwen3.8-Flash-DGX-AutoRound

# 2. serve (auto-downloads model + PLE table on first run, :8000)
./serve.sh
docker logs -f qwen38-uncensored   # ~4-5 min to "Application startup complete"

# 3. smoke test
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen",
  "messages": [{"role": "user", "content": "Write a haiku about a desktop supercomputer."}],
  "max_tokens": 512}'

# stop
./serve.sh --stop
```

## Configuration

Every knob is an env var on `serve.sh`:

| Var | Default | Notes |
|---|---|---|
| `MODEL_ID` | `bidhata/Qwen3.8-Flash-Next-AutoRound-Uncensored` | this fork's weights |
| `TABLE_ID` | `Saren/Qwen3.8-Flash-Next-ple-table-fp8` | unchanged PLE table |
| `MODEL_DIR` / `TABLE_DIR` | `/models/uncensored-hybrid`, `/models/ple-table-fp8` | skip download if present |
| `PORT` | `8000` | host port |
| `CTX` | `262144` | max context (500k YaRN possible, untested here) |
| `SEQS` | `8` | max concurrent sequences |
| `KV_BYTES` | `16g` | explicit KV pool (optimal; 20g works too but wastes RAM) |
| `MTP` | `3` | speculative tokens (`0` = off, lowers TTFT floor) |
| `IMAGE` / `NAME` | `qwen38-flash-dgx` / `qwen38-uncensored` | |

One model at a time per box: two residents (stock + uncensored) exceed the
128 GB pool. To A/B, stop one before starting the other.

## Tuning (1× DGX Spark, vLLM + MTP=3)

Swept by recreating the container per config and re-benching (see
`tools/measure_toks.py` — the same harness, run per-config). Results:

| config | decode tok/s |
|---|---|
| **baseline (MTP=3, bt=8192, no flashinfer autotune)** | **31.0** |
| MTP=2 | 30.0 |
| MTP=4 | 26.6 |
| max-num-batched-tokens 2048 | 30.0 |
| flashinfer autotune on | 29.8 |

MTP=3 is the peak; everything else is within noise. Under decode the GPU
sits at ~72% util, ~31 W, ~2.5 GHz (max 3.0) — you are **overhead-bound,
not clock-bound**, so locking clocks to 3.0 GHz does not help. `serve.sh`
defaults are already the winner.

**KV_BYTES**: `16g` is optimal (same 33.5 tok/s as `20g` while freeing ~3.3 GB extra RAM). The stock default `20g` is safe too — no speed penalty, just more unused KV space.

Two knobs worth knowing about:

- **Agentic / latency-sensitive workloads** (short prompts, one token at a
  time): set `MTP=0`. It removes the ~1.3 s TTFT floor (→ ~230 ms) at the
  cost of halving decode — a 5× latency win for interactive use. Pure
  single-stream bulk generation: leave it at 3.
- **Concurrency is free** — aggregate throughput climbs steeply to 8–16
  streams (recipe's own bench). If you run parallel agents you are leaving
  most of it on the table at 1 stream. One model at a time per box though:
  two residents (stock + uncensored) exceed the 128 GB pool.

## Reproduce the ablation

The build scripts live in `tools/` (needs `torch` + `safetensors` +
`numpy` + `requests`):

- `tools/quant_roundtrip.py <shard>` — proves dequant→requant is bit-exact
  at edit=0 for both the fp8-blockwise and int4-GPTQ families, so the
  pipeline preserves format by construction.
- `tools/abliterate_hybrid.py <model-dir> <refusal-vec.pt>` — the edit
  itself: direction in, edited shards out. Operates shard-by-shard
  (~1 GB RAM), ~10 min for all 68 files. **Back up the checkpoint first**
  — it edits in place. Recover the direction vector from any stock-vs-
  abliterated shard pair (diff → SVD → top left/right singular vector on
  the 2560-dim residual axis).
- `tools/measure_toks.py [max_tokens] [runs]` — streaming tok/s bench
  (TTFT / decode / overall via `usage.completion_tokens`; counts
  `reasoning` deltas — qwen3 streams thinking there, ignoring it corrupts
  both metrics).
- `tools/validate_uncensored.py` — refusal battery + capability probes
  against `http://localhost:8000`.

## Upload your own build

```bash
./upload.sh [HF_REPO_ID]   # uploads 71 GB to a gated repo (default: mine above)
```

## Credits

- Model: Qwen team, Alibaba (Qwen3.8-Flash-Next).
- int4 checkpoint: Intel (W4A16 AutoRound).
- Hybrid recipe, PLE-mmap serving stack, patches: Saren-Arterius
  ([qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound),
  forked from blazux) — the foundation this fork stands on.
- Refusal-direction source: orcarouter (Qwen3.8-Flash-Next-Uncensored).
- Engine: vLLM.

## License & disclaimer

Weights carry the Qwen community license — review it before production use
(it has a MAU/revenue clause). This is an **uncensored research artifact**
with safety refusals removed: you are solely responsible for what you
generate and for complying with all applicable laws. Provided as-is,
without warranty.

Contact: Krishnendu Paul &lt;me@krishnendu.com&gt;
