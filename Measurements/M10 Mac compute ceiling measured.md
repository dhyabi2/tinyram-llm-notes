---
id: M10
type: measurement
status: info
links: [P03, M09]
date: 2026-10-28
---
# Mac compute ceiling measured

Fully-cached decode on the Mac (i7-1068NG7, 4 cores): Qwen3-30B-A3B Q4_K_M 8.3 tok/s x 2 x 3.3B active = 55 GFLOP/s; OLMoE-1B-7B 29 tok/s x 2 x 1.3B = 75 GFLOP/s. The old assumption (60) holds. For the 125B/6B target at 4.3 bpw: 3.23 GB touched per token, compute ceiling 4.57 tok/s.
- see [[P03 The 100x claim is unproven]]
- see [[M09 Mac Qwen3-30B-A3B prefetch test stayed cached]]
