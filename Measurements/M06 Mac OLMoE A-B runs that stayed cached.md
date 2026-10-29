---
id: M06
type: measurement
status: invalid
links: [F14,L01,P10]
date: 2026-10-28
---
# Mac OLMoE A-B runs that stayed cached

Two quiet Mac A/Bs read ~0 MiB per token and ran ~29 tok/s: the 4.2 GB file stayed in the page cache (3 GB cap and eviction-by-reading did not bind). Speedups of 0.989x, 0.971x, 0.932x compare cached runs and mean nothing. Cold first run: 4.7 tok/s, ~80 MiB/token.
