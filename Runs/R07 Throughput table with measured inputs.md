---
id: R07
type: run
status: partial
links: [P03, P01, M04, M07, F14, F15]
date: 2026-10-28
---
# Throughput table with measured inputs

tok/s = min(4.57, disk GB/s / ((1-hit) x 3.23 GB)). Disk: mmap paging 0.34, madvise 0.48 (x1.41, VPS only), explicit 1 MiB 1.56, explicit 4 MiB 2.38 GB/s (Mac, measured). Hit rate ASSUMED (not measured).

| path | hit 0 | 0.5 | 0.8 | 0.9 | 0.95 |
|---|---|---|---|---|---|
| mmap | 0.11 | 0.21 | 0.53 | 1.05 | 2.11 |
| madvise | 0.15 | 0.30 | 0.74 | 1.49 | 2.97 |
| 1 MiB explicit | 0.48 | 0.97 | 2.42 | 4.57 | 4.57 |
| 4 MiB explicit | 0.74 | 1.48 | 3.69 | 4.57 | 4.57 |

Best case over mmap = disk ratio 7x while both disk-bound, ~4.3x once the fast path reaches the compute ceiling. 100x needs a baseline under 0.02 tok/s; mmap at hit 0.5 gives 0.21. 8 GB PC: ~4 GB cache vs ~66 GB experts = ~6% by size, so the hit rate hinges on token-to-token reuse (not measured); likely 0.2-1.5 tok/s.
- see [[P03 The 100x claim is unproven]]
- see [[P01 mmap read amplification]]
- see [[M04 Mac disk read-size sweep]]
- see [[M07 VPS prefetch cold A-B test]]
- see [[F14 expert prefetch patch]]
- see [[F15 wire the expert pool into ggml]]

Update with measured reuse ([[M11 Expert reuse on real routing (Qwen3-30B-A3B)]]): hit rate for a pinned hot set (out of sample) 0.345 at a 6% cache (8 GB PC), 0.73 at 20% (16 GB), 0.91 at 33% (32 GB). tok/s for the 125B target with explicit 4 MiB reads: 1.12 / 2.72 / 4.57 (compute-bound). Plain mmap behaves like LRU, so at 6% it has ~0 hit rate: ~0.10 tok/s, making the 8 GB case about 10x. Estimates, hit rates mapped from a 30B model.

**CORRECTED (footprint units, expert bytes only):** 125B target, disk bytes per token = (1-hit) x 1.30 GB (480 x 2.7 MB; the ~1.9 GB dense part is resident, not read). Serial bound 1/(io+1/4.57); overlap bound min(4.57, rate/bytes). 8 GB (4% share, 2.1 footprints): mmap as LRU(footprint) hit 0.56 -> 0.53 serial / 0.59 overlap tok/s; explicit 4 MiB + LRU 2.18 / 4.16; explicit + pinned-by-share (hit 0.26) 1.61 / 2.48. 16 GB (7.6 footprints): mmap 2.45 / 4.57; explicit 4.07 / 4.57. 32 GB: mmap 2.76; explicit 4.18. So the gain over mmap is ~4-7x at 8 GB and comes from READ SIZE (disk rate), not from pinning; pinned-by-share loses to LRU at equal bytes. The earlier '~10x at 8 GB' is withdrawn. All 125B figures are proxy extrapolations from a 30B geometry.
