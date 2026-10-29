---
id: M18
type: measurement
status: done
links: [Mac naive baseline under a wired ballast]
date: 2026-10-29
---
# Naive mmap baseline under a strict memory cap (VPS)

Qwen3-30B Q4_K_M plain mmap, cold, strict cgroup cap, 2 reps each, shared throttled VPS disk (shape only). Cap 5500 MB (file/cap 3.37x): 0.107 tok/s, 24 GiB read for 9 tokens. Cap 3700 MB (5.02x): 0.086 tok/s, 25 GiB. Cap 2200 MB (8.43x): 0.0376 tok/s, 32 GiB, 543 s wall. So the naive baseline loses 65% of its speed between 3.4x and 8.4x. Against the modelled base path at 8 GB (2.49 tok/s) the 8.4x point is 66x, but the disk here is throttled to ~120 MiB/s so the ratio is not transferable; repeat on the Mac.
