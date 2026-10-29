---
id: R09
type: run
status: done
links: [P03, P01, F15, M11, R07]
date: 2026-10-28
---
# Plan F cache layout for the 125B

14 stages, 49 KB, verifier 50 (50/40/50), 3 open. Resident dense ~1.8-2.1 GB; per expert ~2.7 MB (2.6 MiB), 24,576 experts, pool ~65 GB. Table (cache / pinned+managed / hit / tok/s, all [E]): 8 GB: 2.5 GiB (4%) all pinned, hit 0.26, ~1.1-2.0 tok/s (~0.1 plain mmap, ~10x); 16 GB: 9.15 GiB (14.6%), 2.29 pinned + 6.86 LRU, hit 0.64, 2.2-3.5; 32 GB: 25 GiB (40%), 4.5 pinned + 20.6 managed, hit 0.94, 4.57 (compute ceiling). 100x not supported. New catch: the 69 GB file does not fit in the Mac's 54-57 GB free disk; needs a <=46 GB (~2.9 bpw) quant or moving ~20-23 GB of other files. Everything about the 125B's own routing is unknown until its checkpoint runs.
- see [[P03 The 100x claim is unproven]]
- see [[P01 mmap read amplification]]
- see [[F15 wire the expert pool into ggml]]
- see [[M11 Expert reuse on real routing (Qwen3-30B-A3B)]]
- see [[R07 Throughput table with measured inputs]]

**CORRECTED:** the 8 GB row (hit 0.26, ~1.1-2.0 tok/s, '~10x over mmap') assumed LRU is ~0 at a 4% cache. In footprint units a 2.1-footprint cache gives LRU hit ~0.56, so plain mmap is not near 0.1 tok/s in this model (about 0.5-0.6). See [[R07 Throughput table with measured inputs]].
