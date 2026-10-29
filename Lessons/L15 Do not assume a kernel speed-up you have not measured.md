---
id: L15
type: lesson
status: info
links: [M17, R18]
date: 2026-10-29
---
# Do not assume a kernel speed-up you have not measured

The combination model used a 1.3-2.5x lookup-kernel gain that no measurement supported; measuring the existing low-bit kernels (decode x0.86-1.12 vs Q4_K_M) removed it and cut the best realistic CPU-only combination from 2.5x/3.55x to 1.85x/2.4x at 8 GB. Same for predictor precision: use the held-out figure (0.31-0.43), not the in-sample 0.55.
- see [[R18 Combining the speed levers- all combinations]]
