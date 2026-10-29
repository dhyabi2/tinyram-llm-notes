---
id: M09
type: measurement
status: invalid
links: [F14, P10, L07]
date: 2026-10-28
---
# Mac Qwen3-30B-A3B prefetch test stayed cached

Qwen3-30B-A3B Q4_K_M (18.6 GB) on the 16 GB Mac, modes 0/1/2 rotated. Uncapped: 8.1 / 8.0 / 7.9 tok/s, ~0 MiB read per token after the first cold run (speedups 0.977x and 0.966x are noise between cached runs). With a verified 3 GB wired ballast (100.2%): the same, after a cold first round (0.89 / 1.03 / 0.54 tok/s, ~405 / 292 / 298 MiB per token, single samples). The first cold run is ~9x slower than the cached ones (0.93 vs 8 tok/s). Conclusion: this Mac cannot give a clean streaming test without sudo purge.
- see [[F14 expert prefetch patch]]
- see [[P10 No clean cold start on the Mac]]
