# Without our tool vs with our tool (clean comparison, 2026-10-29)

"Without" = stock llama.cpp, plain mmap. "With" = our patched llama.cpp (madvise prefetch on the chosen experts) and, in the model rows, the explicit-read path with an expert cache (not built yet, so those rows are projections).

## A. Measured on real hardware
| # | setup | without | with | gain | notes |
|---|---|---|---|---|---|
| 1 | VPS, Qwen3-30B Q4_K_M, 5.8 GB RAM, cold, real prompt, 40 tokens, 5 reps balanced + null pair | 0.177 tok/s | 0.293 tok/s | **1.65x** generation | null pair (off vs off) 0.178 |
| 2 | same runs, wall clock (load + prompt + 40 tokens) | 702 s | 572 s | **1.23x** | older 9-token figure (1.09x) is superseded |
| 3 | same runs, prompt eval | 181 s | 139 s | 1.30x | prefill also gains |
| 4 | same runs, data read from disk | 40.7 GiB | 33.2 GiB | -18% | |
| 5 | VPS, pinned hot experts (n=2, 16 tokens) | 0.134 | 0.254 same-prompt / 0.180 cross-prompt | 1.9x / 1.3x | pin costs ~60 s of load; cross-prompt pin worse than madvise alone; no net gain on long runs |
| 6 | Mac disk, 2.7 MB expert read, 4 threads | mmap paging 0.28-0.60 GB/s | explicit read 3.11 GB/s | 5-10x raw read speed | micro-benchmark, not end-to-end |
| 7 | output correctness (tiny MoE, 300 prompts x 8 switch modes) | - | identical bytes | - | 3,325 checks: the tool changes speed, not output |

## B. Naive baseline when memory is too small (without the tool)
| machine | model / memory | tok/s |
|---|---|---|
| VPS strict cap | Qwen3-30B, file 3.4x the cap | 0.107 |
| VPS strict cap | file 5.0x the cap | 0.086 |
| VPS strict cap | file 8.4x the cap | 0.038 |
| Mac (16 GB) | Qwen3-30B fully warm in RAM | 8.0 |
| Mac | Qwen3-30B, 9 GB wired ballast (file ~8x cache, 10-14% resident) | 0.28-0.40 (median 0.33) |
With the tool at these same caps: not measured yet (the explicit-read path is not built); the VPS A/B above is the only measured "with" at a tight cap.

## C. Projection for the 125B model (compute ceiling 4.58 tok/s; per-token expert data 1.30 GB; NVMe 3.11 GB/s explicit vs 0.34 GB/s mmap paging)
Two cache hit-rate assumptions: (1) our 30B-trace curve, (2) a public benchmark on the 125B-class model itself (single-user GitHub numbers, unverified; 8 GB point extrapolated).
| RAM | without (mmap) 30B-curve | with (explicit reads + LRU) 30B-curve | without, real-model curve | with, real-model curve | + quality-safe levers (30B-curve) | + top-6 experts (quality cost +8.8% ppl on OLMoE) |
|---|---|---|---|---|---|---|
| 8 GB | 0.53 | 2.49 | 0.47 | 2.34 | 3.14 | 3.85 |
| 16 GB | 2.46 | 4.19 | 0.91 | 3.18 | 4.82 | 5.48 |
| 32 GB | 2.75 | 4.27 | 2.06 | 4.04 | 4.95 | 5.55 |
Hit rate used: 30B curve 0.56 / 0.95 / 0.96; real-model curve about 0.50 / 0.77 / 0.93. The 16 GB row is the one most affected (4.19 -> 3.18).
All of these are projections (tok/s), not measurements on the 125B; the "without" mmap column is a model of paging, only partly checked by section B.

## D. Gain of each lever over the "with" base path (trace-simulated on real routing traces, 8 GB / 16-32 GB)
| lever | 8 GB | 16-32 GB | quality |
|---|---|---|---|
| overlap I/O with compute | 1.26x | 1.03-1.04x | none |
| cache policy (LRU / segmented / pinned) | 0.98x | ~1.0x | none |
| cache-aware 6-of-8 expert choice | 1.38x | 1.13-1.16x | unmeasured |
| top-6 experts | 1.24x | 1.12-1.14x | +8.8% ppl (OLMoE) |
| top-4 experts | 1.61x | 1.28-1.30x | +45.6% ppl (OLMoE; renormalisation effect untested) |
| 3-bit experts | 1.29x | 1.02-1.04x | +10.4% ppl (OLMoE, upper bound) |
| skip 25% of layers | 1.47x | 1.33-1.36x | unmeasured |
| speculation k=4 alone | 0.74x | 1.08-1.09x | none (acceptance rate assumed) |
Levers combine sub-additively at 16-32 GB; at 8 GB speculation only pays once experts per token are cut.

## E. What is not established
- No end-to-end run of the 125B model; no measured "with" number for the explicit-read path.
- Hit rates for the 125B come from a 30B proxy; one public run suggests the 16 GB hit rate is nearer 0.77 than 0.95.
- Top-k quality cost depends on router renormalisation (one paper on a 512-expert top-10 model finds 0.55 MMLU points at 10 -> 5 with a decoupled reference vs 2.10 with plain renormalisation); our +45.6% comes from a router without renormalisation.
- Related public work already streams a 125B-class 512-expert model from flash on small machines (see papers/SUMMARY.md and deep_P05b.md).
