---
id: M07
type: measurement
status: partial
links: [F14]
date: 2026-10-28
---
# VPS prefetch cold A-B test

Qwen3-30B-A3B Q4_K_M on the VPS (2 cores, 5.8 GB RAM, throttled 120 MiB/s disk), `drop_caches` before every run so each starts cold, 8 tokens (9 counted), 3 runs per mode, modes rotated. Raw: runs/bench/ab-vps-qwen3-30b.json.

| Mode | tok/s median (range) | read per token | faults per token |
|---|---|---|---|
| 0 off | 0.1036 (0.1032-0.1052) | 2,741 MiB | 5,622 |
| 1 madvise prefetch | **0.1464** (0.1407-0.1617) | 2,543 MiB | 3,876 |
| 2 parallel touch | 0.1043 (0.1026-0.1090) | 2,736 MiB | 5,669 |

**madvise prefetch = 1.41x**, ranges do not overlap, faults per token -31%. Parallel touch = 1.007x (no effect).
Limits: n=3 per mode, one throttled VPS disk, 9-token cold-start runs (cold-start streaming, not a long steady state). Disagrees with the contended Mac test (1.045x, [[M05 Mac OLMoE cold A-B test (contended)]]); the Mac could not be tested cleanly ([[M09 Mac Qwen3-30B-A3B prefetch test stayed cached]]). Not close to 100x.

**CORRECTED after the playbook review ([[R10 Playbook review of 22 methods]]):** 1.41x is a tok/s ratio over the timed decode window only (about 77 s of a ~410 s run, 15-19%). Wall-clock speedup is **1.09x** (median 409.8 s vs 374.6 s). 'Read per token' is whole-process page-ins divided by 9; times tok/s it implies 286-372 MiB/s against a 120 MiB/s disk, so it is not a per-decode-token figure. n=3 per arm cannot give an exact two-sided p below 0.10. Rerun with 32-64 decode tokens, n>=5, balanced order and a pgpgin snapshot at decode start/end before citing any multiplier.
