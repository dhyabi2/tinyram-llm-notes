---
id: R21
type: run
status: done
links: [R18, M17, M18]
date: 2026-10-29
---
# Three-part sweep- second screen, local measurements, code fixes

Part 1 (screen): 26 more families, 0 pursue-now (see R18). Part 2 (measurements): existing low-bit kernels do not speed up decode (x0.86-1.12 vs Q4_K_M, M17); OLMoE perplexity vs active experts 8/6/4/2 = +0/+8.8/+45.6/+593%, decode x1.00/1.19/1.48/1.78; Q3_K_S +10.4%, Q2_K +26% (requantised upper bounds) (M18). The combination model was corrected (lookup-kernel gain lower bound 1.0x, held-out predictor precision 0.43, quality risk scores from the measured perplexity): CPU-only over our best path, quality-safe levers give x1.26-1.44 at 8 GB and ~x1.0-1.2 at 16/32 GB; x2.3 at 8 GB needs top-4 (+46% PPL). Part 3 (code): phase hook on the real decode path, schema-3 converter and contract test, shared trace loader + pin builder. Tests: 2,088 of 2,088 checks pass (95 s).
- see [[R18 Combining the speed levers- all combinations]]
- see [[M17 OLMoE perplexity and decode speed vs experts per token and bits]]
