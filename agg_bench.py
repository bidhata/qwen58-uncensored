#!/usr/bin/env python3
"""Aggregate throughput: N concurrent batch-1 streams, unique prompts each.

Usage: agg_bench.py [streams=8] [tokens=256]
Honest fleet number: unique prompts (no prefix-cache help), non-streaming.
"""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://localhost:8000"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
TOK = int(sys.argv[2]) if len(sys.argv) > 2 else 256


def one(i):
    prompt = f"[agg-{i}-{time.time():.0f}] Write a very long detailed essay about the history of computing. "
    body = json.dumps({"model": "qwen", "prompt": prompt, "max_tokens": TOK,
                       "temperature": 0, "ignore_eos": True}).encode()
    req = urllib.request.Request(f"{BASE}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    d = json.load(urllib.request.urlopen(req, timeout=900))
    dt = time.perf_counter() - t0
    return d.get("usage", {}).get("completion_tokens", TOK), dt


t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=N) as ex:
    res = list(ex.map(one, range(N)))
wall = time.perf_counter() - t0
tot = sum(n for n, _ in res)
rates = sorted(n / dt for n, dt in res)
print(f"streams={N} toks/worker={TOK} total_tokens={tot} wall={wall:.1f}s", flush=True)
print(f"aggregate: {tot / wall:.1f} tok/s", flush=True)
print(f"per-worker tok/s: min {rates[0]:.1f}  median {rates[len(rates)//2]:.1f}  max {rates[-1]:.1f}", flush=True)
