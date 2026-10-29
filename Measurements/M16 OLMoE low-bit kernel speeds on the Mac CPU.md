---
id: M16
type: measurement
status: done
links: [P01, R17, L02]
date: 2026-10-29
---
# OLMoE low-bit kernel speeds on the Mac CPU

llama-bench, 4 threads, model fully in RAM, 3 repetitions (runs/bench/olmoe_kernel_speeds.json). OLMoE-1B-7B requantised from Q4_K_M with --allow-requantize (speed only, not quality). Prefill 512 tokens / decode 64 tokens, tok/s: Q8_0 58.5 / 24.0; Q4_K_M 107.4 / 41.0 (279 / 107 GFLOP/s at 1.3B active); Q3_K_S 158.3 / 35.5; Q2_K 128.0 / 45.9. Relative to Q4_K_M: prefill x0.54, 1.00, 1.47, 1.19; decode x0.58, 1.00, 0.86, 1.12. Existing low-bit kernels do NOT speed up decode (0.86-1.12x) although Q2_K reads ~40% fewer bytes: decode on this CPU is limited by compute and dequantisation, not memory bandwidth. So the assumed 1.3-2.5x lookup-kernel gain has no support; the model lower bound is now 1.0x. Batched prefill reaches 279-412 GFLOP/s, 2.6-3.9x the decode rate, consistent with the 311 GFLOP/s used for TTFT.
- see [[P01 mmap read amplification]]
- see [[R17 Loop-until-dry brainstorm rounds 3-6 with an exclusion list]]
- see [[L02 Verifier scores are not stable]]
