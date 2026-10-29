---
id: F30
type: fix
status: done
links: [P03, R07, R10, T09]
date: 2026-10-28
---
# Throughput model v2

tools/throughput_model_v2.py (tinyram-llm working copy, committed locally db30e60, not pushed): cache in token-footprints, expert bytes only charged to the disk (1.30 GB per token for the 125B), serial and overlap bounds, read-size rates 0.34 / 1.56 / 2.38 GB/s, LRU table recomputed from the three real traces. 8 GB: mmap 0.53 serial / 0.59 overlap; explicit 4 MiB reads 2.18 / 4.16; 16 GB: 2.46 / 4.58 vs 4.08 / 4.58; 32 GB: 2.76 vs 4.19. Still an extrapolation from a 30B geometry, no prefill term.
- see [[P03 The 100x claim is unproven]]
- see [[R07 Throughput table with measured inputs]]
- see [[R10 Playbook review of 22 methods]]
- see [[T09 tinyram-llm repo]]
