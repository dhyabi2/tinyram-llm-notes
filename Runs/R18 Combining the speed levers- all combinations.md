---
id: R18
type: run
status: done
links: [R15, R16, M16, P03]
date: 2026-10-29
---
# Combining the speed levers- all combinations

tools/combo_model.py + combo_analysis.py (tinyram-llm, committed locally): 1,512 combinations of 9 levers (top-k 10/6/4, bits 4.3/3.0/2.5 or a low-rank basis, cache-aware routing, overlap, speculative k=4/8, GPU tier 0/12/24 GB, table-lookup kernels, early exit 75%) x 8/16/32 GB machines x conservative/likely/optimistic parameters, in parallel. Anchored on measured: LRU hit curve in footprints, union-of-experts table, 3.11 GB/s explicit reads, 4.58 tok/s compute ceiling. All other parameters are estimates. Quality is NOT modelled; a risk score (top6=2, top4=3, 3-bit=2, 2.5-bit=3, low-rank=3, cache routing=1, early exit=3) stands in. Results, CPU-only, x our best path (explicit reads + LRU: 2.49 / 4.19 / 4.27 tok/s at 8/16/32 GB; mmap 0.53 / 2.46 / 2.75): NO quality risk: overlap alone x1.34 at 8 GB, x1.05 at 16 GB; exact speculation x0.73-0.77 at 8 GB (cache thrash), x1.12-1.17 at 32 GB. Risk <= 2: 3-bit + overlap + table-lookup kernels x2.47 (8 GB; x1.76 conservative; Monte Carlo median 2.50, P(>=2x) 98%, P(>=3x) 0%), x1.9 at 16 GB, x2.1 at 32 GB (P(>=3x) 4%). Risk <= 4: top-6 + 3-bit + overlap + lookup kernels x3.55 at 8 GB (x2.55 conservative), x2.6 at 16 and 32 GB. Marginal effect (geometric mean when switched on): lookup kernels x1.45-1.48, top-4 x1.23-1.47, early exit x1.34-1.41, overlap x1.02-1.13, cache routing x1.01-1.11, spec k=4 x1.05-1.09, spec k=8 x0.88-1.03, low-rank x0.86-0.98. Interactions: every pair is sub-additive or about additive (0.77-0.98 of the product of the singles), none super-additive. With a 24 GB GPU tier the model reaches x4-13 (no quality risk) but that is not the small-PC target. Honest reading: combining does not multiply for free; against plain mmap the stack is 5x (best path) to 12-17x at 8 GB, against our best path the realistic ceiling is 2-2.5x without quality loss and about 3.5x only with 3-bit + top-6 whose quality cost is unmeasured.
- see [[R15 Brainstorm engine over every problem and the speed levers]]
- see [[R16 Screen of the top 22 brainstorm idea families]]
- see [[P03 The 100x claim is unproven]]
