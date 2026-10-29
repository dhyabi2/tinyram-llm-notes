---
id: R25
type: run
status: done
links: [100x feasibility map]
date: 2026-10-29
---
# Trace-driven combination runs

3,888 combination runs on the 3 recorded Qwen3-30B traces: 3 cache policies x 4 selection rules x 3 bit widths x 3 speculation settings x 2 layer-skip x overlap x 3 RAM sizes. Hit rates simulated, speeds from the combo time model. Result at 8 GB (base 2.50 tok/s): single levers top6 1.24, top4 1.61, cache-aware 6-of-8 1.38, 3-bit 1.29, skip25 1.47, overlap 1.26, speculation alone 0.74-0.93 (the union of experts across draft tokens exceeds the 8 GB cache). New finding: speculation becomes useful once experts per token are cut (top6+spec4 1.16 vs product 0.92; spec4+skip25 1.39 vs 1.08). At 16/32 GB no pair beats the product of its parts; policy choice (LRU/segmented/held-out pin) moves speed <3%. Quality-safe best: 1.26x at 8 GB, 1.15x at 16 GB. Cache-aware 6-of-8 never has more misses than rank-order top6 in any of 27 checks. Report: Playbook/COMBINATIONS_TRACE.md. Speculation acceptance and overlap share are still estimates.
