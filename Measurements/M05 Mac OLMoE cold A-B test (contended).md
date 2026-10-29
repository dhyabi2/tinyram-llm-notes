---
id: M05
type: measurement
status: partial
links: [F14,L01]
date: 2026-10-28
---
# Mac OLMoE cold A-B test (contended)

OLMoE-1B-7B Q4_K_M, madvise prefetch off vs on, 4 samples each while a 27B run thrashed memory: 4.39 vs 4.59 tok/s = **1.045x, within noise**; read per token ~155 vs ~152 MiB.
