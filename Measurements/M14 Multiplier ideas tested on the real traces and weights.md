---
id: M14
type: measurement
status: done
links: [M11, M12, P03, R13]
date: 2026-10-29
---
# Multiplier ideas tested on the real traces and weights

Four candidate speed multipliers measured locally (no model calls), Qwen3-30B-A3B traces (3 prompts, 249-299 tokens x 48 layers, top-8 of 128) and its Q4_K_M weights:
(1) Multi-token verification (speculative or batched tokens of ONE sequence): experts needed per layer for k consecutive tokens = 8.0 (k=1), 12.6 (2), 16.3 (3), 19.6 (4), 24.9 (6), 29.1 (8), 35.2 (12), 39.2 (16): IO amortisation 1.00x, 1.27x, 1.47x, 1.63x, 1.93x, 2.20x, 2.73x, 3.26x. This is the best case; rejected draft positions still load their experts, so at a speculation acceptance well below k the gain per accepted token is about 1x.
(2) Batching N unrelated sequences (aligned tokens from different prompts): 1.07x (N=2), 1.13x (N=3). Different prompts share few experts, so batching unrelated streams does not amortise loads. Same-prompt branches are untested.
(3) Cross-layer prediction: a co-occurrence predictor (layer L-1 experts -> layer L experts, trained on the first half, scored on the second) gets precision@8 of 0.547 / 0.548 / 0.508 (db / story / code), vs the layer's own hot list 0.353 / 0.394 / 0.319, vs the previous token's same-layer experts 0.431 / 0.455 / 0.420, random 0.062. Best predictor available from traces alone, but ~45% of a prefetch would be wasted.
(4) Lossless compression of expert weights: Q4_K experts entropy 7.77 bits/byte; zlib 1.02x, lzma 1.02x, bz2 1.01x; Q6_K 1.06-1.09x. Compression cannot cut disk bytes.
Net: none is a free 3x. The real levers remain the read rate (mmap 0.34 -> 1.5-2.4 GB/s), staying near the 4.57 tok/s compute ceiling, fewer bytes per token (bits, top-k, pruning), and better overlap. Method: tools in tinyram-llm (trace_tools), scripts inline in the session.
- see [[M11 Expert reuse on real routing (Qwen3-30B-A3B)]]
- see [[M12 Hot set transfers poorly across prompts]]
- see [[P03 The 100x claim is unproven]]
- see [[R13 Itemised work list and test catalog]]
