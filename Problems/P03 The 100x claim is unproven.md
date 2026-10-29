---
id: P03
type: problem
status: done
links: [M04,R02]
date: 2026-10-28
---
# The 100x claim is unproven

No speedup was ever measured. Planning run B retired 100x against tuned mmap (needs ~34 GB/s disk and 126 GFLOP/s at once). Honest claim: none measured; estimated single digit to ~15x, hard cap near 48x. Measured disk-side headroom on the Mac: 4.5-7x.

Update: recomputed with measured inputs, see [[R07 Throughput table with measured inputs]]: best case over mmap is 7x (disk ratio), about 4.3x once compute-bound; 100x ruled out by arithmetic.
