---
id: M11
type: measurement
status: done
links: [P01, P03, F15, R07, T08]
date: 2026-10-28
---
# Expert reuse on real routing (Qwen3-30B-A3B)

GGML_MOE_TRACE on a real 299-token generation (database-index explanation, greedy, 6.3 tok/s), 48 layers x 128 experts, top-8. Raw trace: runs/bench/moe_trace_qwen3-30b.txt; analysis: runs/bench/analyze_trace.py.
- Token-to-token persistence 0.425; seen within last 8 tokens 0.78, 16 tokens 0.89, 32 tokens 0.95.
- Working set over 299 tokens: 62% of experts per layer (78.8 of 128).
- Hottest 5% of experts cover 33% of accesses, 10% -> 51%, 20% -> 76%, 33% -> 93%.
- Global LRU hit rate: 0.000 at cache <= 6% of experts (a token touches 384 experts across layers, so LRU evicts before reuse), 0.47 at 10%, 0.70 at 20%, 0.90 at 33%.
- Out-of-sample pinned hot set (learned on tokens 1-149, scored on 150-299): 0.16 at 2%, 0.31 at 5%, 0.345 at 6%, 0.48 at 10%, 0.73 at 20%, 0.91 at 33%. Pure LRU at 6% = 0.
- Caveat: one prompt, one model; the share of experts a cache holds is mapped to the 125B model by assumption (it has 512 experts, top-10).
- see [[P01 mmap read amplification]]
- see [[P03 The 100x claim is unproven]]
- see [[F15 wire the expert pool into ggml]]
- see [[R07 Throughput table with measured inputs]]
- see [[T08 moe-prefetch.patch]]

**CORRECTED:** the 'LRU = 0.000 at <=6% cache' is a cliff caused by the cache (368 entries) being smaller than one token's 384 accesses. In token-footprints (cap = k x 384, mean of 3 prompts, cold start included): k=0.9 0.000, k=1 0.306, k=1.5 0.454, k=2 0.549, k=3 0.686, k=4 0.793, k=7.5 0.950. The 125B has 480 accesses per token of 24,576 experts, so its cliff sits at ~2% of experts, not 6%.
