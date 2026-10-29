---
id: R24
type: run
status: done
links: [R20, R19, P03]
date: 2026-10-29
---
# How much RAM raises max tokens per second

tools/ram_sweep.py (tinyram-llm, committed): RAM enters only through the cache hit rate (measured LRU curve in token-footprints; cache left = 0.944 x RAM - 5.05 GiB, a line through plan F's 8/16/32 GB points [E]). Base path (explicit reads + LRU, no quality cost) at a 55 GFLOP/s CPU (compute ceiling 4.58 tok/s), steady-state hit: 4-6 GB 1.58 tok/s (34% of ceiling; cache holds nothing), 8 GB 2.49 (54%), 12 GB 3.55 (77%), 16 GB 4.47 (98%), 24 GB and above 4.54 (99%); with overlap 3.13 at 8 GB, 3.95 at 12, 4.52 at 16. Value of RAM (tok/s per extra GB): 6->8 GB 0.456, 8->12 0.265, 12->16 0.231, 16->24 0.008, beyond 24 GB zero. So RAM raises the maximum speed up to about 16 GB and then buys nothing for decode: the compute ceiling binds. Plain mmap modelled with the same LRU hit: 0.25 / 0.53 / 1.25 / 3.73 / 4.22 tok/s at 4-6 / 8 / 12 / 16 / 24+ GB, so the base-path advantage over mmap is 6.3x at <=6 GB, 4.7x at 8, 2.8x at 12, 1.2x at 16, ~1.1x beyond (real mmap thrashes worse than ideal LRU, so these ratios are lower bounds on the advantage). A faster CPU moves the ceiling and makes RAM matter more: 110 GFLOP/s -> ceiling 9.17, 90% reached at 16 GB, 12->16 GB worth 0.74 tok/s per GB; 220 GFLOP/s -> ceiling 18.3, 12->16 GB worth 2.06 tok/s per GB, 16 GB gives 91% of ceiling. Caveats: the measured hit curve covers up to 20 footprints on a 299-token trace; its 0.96 plateau is cold-start (corrected by +0.033 for long runs [E]); proxy 30B routing for the 125B; the cache-vs-RAM line is a three-point fit.
- see [[R20 Second idea screen- top new families from rounds 3-7]]
- see [[R19 Evaluation of all 217 investigation items]]
- see [[P03 The 100x claim is unproven]]
