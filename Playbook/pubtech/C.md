# Published-technique families, group C

Screen 1 #16 to #21 and screen 2 #0, #1. Code: `<tinyram-llm>/tools/pubtech/f_C.py` (functions `fam_s1_16` ... `fam_s2_1`, dict `FAMILIES`). Tests: `<tinyram-llm>/tests/test_pubtech_C.py` (117 checks, about 50 s). Hygiene suite passes (171 checks).

## Setup and caveats (apply to every family)

- Data: the three real Qwen3-30B-A3B decode traces (128 experts, top-8, 48 layers, 249-299 tokens each). They stand in for the 125B (512 experts, top-10, 2.7 MB per expert). Cache sizes in token-footprints (entries = footprints x 384): 8/16/32 GB = 2.1/7.6/20.8 footprints. Measured LRU hit at these sizes (mean of 3 traces, cold start included): 0.567 / 0.951 / 0.962.
- Model constants (earlier Mac measurements, not re-measured): compute 55 GFLOP/s decode, 12 GFLOP per token = 0.218 s (4.58 tok/s ceiling); 4 MiB reads 2.38 GB/s, 1 MiB reads 1.56 GB/s, mmap paging 0.34 GB/s, 2.7 MB x 4 in flight 3.11 GB/s; prefill 311 GFLOP/s.
- Disk and model runs done now on this Mac (shared, load average 6-16, run-to-run spread about 15 %): see "Recorded runs" at the end. They are non-deterministic and live in `bench_*` functions and in the numbers recorded as `RECORDED` in `f_C.py`; the tests never run them (only dry runs on a synthetic GGUF).
- Best path = explicit large reads + LRU. At 8 GB it models to 2.20 serial / 4.24 overlap tok/s (screen: 2.18 / 4.16).

## Summary

| Family | Verdict | Key number |
|---|---|---|
| s1 #16 K-prompt lock-step batching | partly (screen's gain confirmed for 125B, kill rule not triggered on 128 experts) | 125B aggregate gain 1.07x (K=8), 1.34x (K=32), 2.72x (K=128); per user 0.042x at K=32 |
| s1 #17 B tokens together | refuted as a speedup vs best path | factor same-domain 0.70 at B=4 (screen expected 0.8-0.95); with the same LRU cache aggregate gain 0.75-1.0x |
| s1 #18 explicit pread, no cache | confirmed (loses without a cache) | 8 GB 1.7x over mmap at 1.5 GB/s (2.5x at the measured 2.48 GB/s); 0.25-0.6x of best path; below mmap at 16/32 GB |
| s1 #19 repack triple + cluster | partly | 8 GB serial gain 1.12x (measured rates) to 1.21x (screen rates); overlap 1.0-1.4x; 16/32 GB under 1.05x; clustering adds under 1.04x |
| s1 #20 co-activation reorder | refuted (as a large lever) | held-out run length 1.35; after cache 1.045x serial at 8 GB, 1.003x at 16/32 GB |
| s1 #21 draft + verify | confirmed dead on IO; compute upside not supported | bytes ratio K=1 0.97-1.00, up to 4.21 above; flat-compute speedup max 0.985 |
| s2 #0 NPU prefill | confirmed at 500 tokens, with a caveat | overlapped TTFT gain exactly 1.0x; matters above about 600-1150 tokens |
| s2 #1 computational-storage FPGA | confirmed (equals striping, no gain at 16/32 GB) | 2 NVMe 4.78 tok/s = ceiling 4.58; with imbalance 3.87 overlapped |

## [screen 1 #16] Serve K independent prompts in lock-step

Idea: one read of each expert slice serves every sequence routing to it; disk bytes per token fall roughly K-fold (claimed 2/6/100x).

Test (re-implemented): K streams = K (trace, offset) pairs, token (offset+t) of each stream at step t; mean distinct experts per layer; sharing = 8K / unique. "same" = K distinct offsets of one trace (one prompt, optimistic), "mixed" = round robin over the three traces. 12 random draws, 100 steps. Extrapolation to 125B by the independent-routing formula 512(1-(1-10/512)^K), 2.4 GB/s reads, hit 0.

| Measure | Value |
|---|---|
| 2 / 3 unrelated aligned streams (screen 1.06 / 1.12) | 1.060 / 1.119 |
| Sharing, same-domain K=2/4/8/16/32 | 1.14 / 1.42 / 1.90 / 2.83 / 4.59 |
| Sharing, mixed K=2/4/8/16/32 | 1.06 / 1.24 / 1.61 / 2.36 / 3.79 |
| Independent random routing on 128 experts, K=2/4/8/16/32 | 1.03 / 1.10 / 1.24 / 1.55 / 2.29 |
| 125B aggregate gain over one stream on the same disk, K=8/16/32/64/128 | 1.07 / 1.155 / 1.335 / 1.74 / 2.72 |
| 125B per-user speed at K=32 | 0.077 tok/s = 0.042 of single-user |

Verdict: partly. The screen's 125B numbers (1.07x at K=8, about 1.35x at K=32, 2.7x at K=128) are reproduced exactly, because they are the random-routing formula. The screen's kill rule "same-domain K=8 sharing below 1.5 on 128 experts" is NOT triggered: measured 1.90 (screen expected 1.2-1.5). That figure is an optimistic bound (one prompt at different offsets, 128 experts; nearby offsets share adjacent-token structure); the independent-random control gives 1.24, and with 3 unrelated prompts 1.12. The 100x claim is refuted (needs identical expert sets). Per-user speed collapses (0.04-0.08x), and no K-stream workload exists for a single user.

Published claims vs measured: SSD-offloading energy paper says amortisation saturates once nearly all unique experts are needed (batch 16 Mixtral, 64 DeepSeek-R1): consistent with the 125B formula where unique/layer reaches 366 of 512 only at K=64. MoE-Lightning (10.3x), MoE-Gen (needs more than 2^11 tokens per expert), ExpertFlow (10x via token grouping): offline large-batch regimes with GPU, not tested; our K<=32 table shows the disk-byte gain is only 1.07-1.34x without grouping. Dynamic Expert Sharing ("expert explosion", unique experts grow nearly linearly): matches the random-routing columns.

Unmeasured: real independent prompts at K>3 (only 3 traces exist), KV-cache memory for K contexts, grouping tokens by similarity (ExpertFlow).

Catalog rows mapped to this family test:
- [screen 1 #16] published method: SSD Offloading for LLM MoE Weights Considered Harmful in Energy Efficiency
- [screen 1 #16] published method: MoE-Lightning (batch offloading, CPU-GPU-IO pipeline)
- [screen 1 #16] published method: MoE-Gen module-based batching
- [screen 1 #16] published method: ExpertFlow (token reordering across batches to reduce expert loads)
- [screen 1 #16] published method: Dynamic Expert Sharing (diffusion LLM)

## [screen 1 #17] Process B tokens together layer by layer

Test: factor = distinct experts per layer / (8B) for B tokens at random distinct positions of one trace (same-domain), one position per trace (mixed), consecutive tokens; plus a batch throughput model (hit 0, 4 MiB reads, flat compute) and, new, a simulation with the SAME shared LRU cache for the batch and the single stream (8 GB cache = 2.1 single-token footprints).

| B | same-domain | mixed | consecutive | random 128 | random 512 |
|---|---|---|---|---|---|
| 2 | 0.863 | 0.939 | - | 0.969 | 0.990 |
| 4 | 0.704 | 0.813 | 0.611 | 0.910 | 0.971 |
| 8 | 0.525 | 0.622 | 0.445 | 0.807 | 0.934 |
| 16 | 0.355 | 0.423 | 0.306 | 0.644 | 0.866 |
| 32 | 0.217 | 0.264 | - | 0.437 | 0.749 |

Screen values reproduce (same-domain 0.86-0.88 / 0.68-0.73 / 0.49-0.55; mixed 0.94/0.81/0.61). Aggregate gain over the best path at 8 GB with the same cache, serial: B=2 1.02x, B=4 0.75x, B=8 0.90x (mixed domains 0.98 / 0.71 / 0.83); overlap 1.04 / 0.60 / 0.83. With no cache for the batch (hit 0): 0.65-0.90x serial against the cached best path. The screen's "1.3-1.7x at B=8" is DISAGREED: batching union working sets exceed the cache (B=4 pass touches about 2.8 footprints), so the lost hits outweigh the shared reads; gain stays at or below about 1.0.

Verdict: refuted as a speedup over the best path at 8 GB (single user: 1.0x by definition). The screen's expectation of "0.8-0.95 at B=4 same-domain" DISAGREES (measured 0.70, more sharing on 128 experts; the kill rule 0.85 is not triggered); the 512-expert random bound is 0.97.

Published claims vs measured: MoE-Lightning up to 10.3x (3.5x with padding) and Klotski up to 85x are GPU-memory batch-offload systems at large batches; "Mixtral offloading suits large batches, interactive much slower" matches per-user 1/B here. None reproduces on a CPU laptop with a cache.

Unmeasured: OLMoE evidence below; real batched decoding speed; compute scaling with batch (flat assumed).

OLMoE (16 layers, 64 experts, top-8, Q3_K_S, 8 prompts x 62 decode tokens recorded with the patched build): union factor same-domain (4 history prompts) B=2 0.900, B=4 0.754; different-domain (history, code, recipe, math) B=2 0.914, B=4 0.772; independent random on 64 experts 0.938 / 0.828. Domain barely matters (0.754 vs 0.772); both show sharing above random, and B=4 same-domain 0.754 is below the screen's expected 0.8-0.95.

Catalog rows mapped to this family test:
- [screen 1 #17] published method: MoE-Lightning
- [screen 1 #17] published method: Klotski
- [screen 1 #17] published method: Mixtral offloading

## [screen 1 #18] Explicit per-layer batched pread instead of mmap (no cross-token cache)

Test: (a) merging of adjacent expert ids into one read (reads per slice), (b) LRU hit below one footprint (per-layer buffer = no cache), (c) tok/s model for mmap-as-LRU vs no-cache pread at the screen's 1.5 GB/s and at the rate recorded now, (d) real replay of the db trace (8 tokens, 8 threads, F_NOCACHE preads of the real gate/up/down slices of the Qwen3-30B GGUF; the GGUF header is parsed by a small stdlib reader in `f_C.py`) and a cold mmap replay.

| Measure | Value |
|---|---|
| Reads per slice after merging adjacent ids (db/story/code) | 0.935 / 0.947 / 0.943 (saving 5.8 %; screen 5-6 %) |
| LRU hit at 0.9 / 1 / 2 / 3 / 7.5 footprints | 0.000 / 0.306 / 0.549 / 0.686 / 0.950 |
| Bytes per token, no cache | 1.296 GB |
| Slice sizes in the GGUF | gate 0.885 MB, up 0.885 MB, down 1.29 MB (mean 1.02) |
| Real pread replay, merged / unmerged (8 threads, per-layer batches) | 2.48 / 2.72 GB/s; 8583 vs 9216 reads (-6.9 %) |
| Real cold mmap replay | 0.277 GB/s |
| 1 / 4 / 8 / 16 threads, db trace layer batches | 1.1 / 3.3 / 3.7 / 3.6 GB/s |

| Machine | mmap serial | no-cache serial @1.5 | @2.48 (measured) | best path serial | no-cache / mmap @1.5 | @2.48 |
|---|---|---|---|---|---|---|
| 8 GB | 0.535 | 0.924 | 1.350 | 2.202 | 1.73x (overlap 1.91x) | 2.52x |
| 16 GB | 2.475 | 0.924 | 1.350 | 4.086 | 0.37x | 0.55x |
| 32 GB | 2.762 | 0.924 | 1.350 | 4.189 | 0.33x | 0.49x |

Verdict: confirmed. At 8 GB the idea gives 1.7x over mmap (screen 1.7x serial / 1.9x overlap, reproduced) and 2.5x with the faster read rate measured now (DISAGREES upward: layer batches of 24 reads at 8 threads reach 2.5-2.7 GB/s, not 1.5). Without a cache it stays 0.25-0.6x of the best path everywhere and below mmap at 16/32 GB. Merging saves only 6 % of reads and 0 bytes.

Published claims vs measured: llama.cpp #19825 reports stock mmap at 1.41-2.12 tok/s with 1,140-1,565 major faults and 208-312 MB SSD I/O per token on 16 GB, and an io_uring layer slightly under 2x [corrected after the paper check: these numbers are from the r/LocalLLaMA post by u/zyxciss (125B 512-expert model, RTX 3060 12GB + 16GB DDR4, 6GB cgroup cap), not from issue #19825, which has none of them; the io_uring claim is in neither (unverified); published stock-mmap gaps are 2.6-6.9x, larger than our emulation, and our 1.5-2.4x came from an mmap emulation, not stock llama.cpp]: same order as our 1.7-2.5x at 8 GB; our cold mmap replay (0.277 GB/s) is consistent with the 0.34 GB/s paging figure. Edge0's 20.4 tok/s needs a trained prerouter: not tested [corrected after the paper check: Edge0 reaches 19.9 tok/s on 24 GB without the prerouter; the prerouter gain is on a 16 GB M2 with K=4].

Unmeasured: io_uring, page-cache hit behaviour of mmap on 8/16 GB (modelled as LRU), the 125B file, mmap replay with madvise.

Catalog rows mapped to this family test:
- [screen 1 #18] published method: llama.cpp issue #19825 (and a r/LocalLLaMA --moe-direct-io post)
- [screen 1 #18] published method: Edge0 (arXiv 2609.18063)

## [screen 1 #19] Repack gate/up/down into one 2.7 MB run, optionally cluster co-firing experts

Test: (1) co-activation chain layout (greedy: start at the most frequent expert, append the unplaced expert with the highest co-count with the last placed; ties by frequency then id), reads per layer per scenario; (2) throughput model for 3 reads -> 1 read with the screen's rate expectations (1.7 -> 2.38 GB/s) and with the rates recorded now; clustering credited through a size^p rate law fitted to the two points and the run length of the reads that remain after the LRU (cross-prompt chain).

| Scenario (reads per layer of 8) | db / story / code | mean |
|---|---|---|
| identity (stock) | 7.48 / 7.58 / 7.54 | 7.54 |
| trained on another prompt (db<-code, story<-db, code<-story) | 5.75 / 6.14 / 6.48 | 6.13 (-18.7 %; screen 6.1/6.4/5.6) |
| trained on the two others | 5.68 / 6.14 / 6.02 | 5.95 |
| same prompt, first half -> second half | 4.89 / 4.91 / 5.38 | 5.06 (screen 4.6/4.6/5.1) |
| in-sample (invalid) | 4.43 / 4.32 / 4.62 | 4.46 |

| Rates (scattered -> contiguous) | 8 GB serial gain | 8 GB overlap gain | 16/32 GB serial | cluster extra (8 GB) |
|---|---|---|---|---|
| screen 1.7 -> 2.38 GB/s | 1.21x | 1.40x | 1.04 / 1.03x | 1.04x serial, 1.07x overlap |
| recorded 2.9 -> 3.8 GB/s | 1.12x | 1.00x (compute-capped) | 1.02 / 1.02x | 1.02x |

Verdict: partly. Part 1 reproduced (cross-prompt 6.1 reads; kill rule "more than 6 reads left" is triggered, marginally: 6.125). Part 2 recorded now (bench_layout, 400 random triples, 4 pairs in flight, medians of 7 interleaved repeats, two seeds): three scattered 0.885 MB reads 3.04 / 2.88 GB/s vs one contiguous 2.85 MB read from a scratch copy 3.89 / 3.71 GB/s = 1.28-1.29x (screen expected 1.2-1.4x, in range). But the idea's tok/s gain at 8 GB is 1.12x serial (not the screen's 1.2-1.4x) with measured rates because the machine is already near the compute ceiling; at 16/32 GB under 1.05x. Clustering adds under 1.04x.

Published claims vs measured: Paging the Experts (flash-backed MoE on iPhone) shows no bundling layout [corrected after the paper check: it does bundle gate/up/down per expert, one 64 KiB-aligned block]; BuddyMoE (buddy substitution on prefetch miss, up to 10 %) is not a layout method; Janas-LLM (24 vs 2.8 tok/s on 32 GB with 19 GB cache) and BigMoeOnEdge (0.94 tok/s, 284B on a 12 GB phone) are blog/news claims: our model at 32 GB is compute-capped at 4.58 tok/s for the 125B, so a 24 tok/s claim cannot come from layout.

Unmeasured: a real rewritten file (scratch-copy emulation used), 125B geometry, interleaving effects on cache granularity.

Catalog rows mapped to this family test:
- [screen 1 #19] published method: Paging the Experts: flash-backed MoE inference on iPhone (arXiv 2609.29032)
- [screen 1 #19] published method: BuddyMoE (arXiv 2511.10054)
- [screen 1 #19] published method: Janas-LLM SSD streaming engine
- [screen 1 #19] published method: BigMoeOnEdge

## [screen 1 #20] Rewrite the GGUF so co-firing experts are adjacent

Test: same chain as #19, leave-one-out over the three traces; run length = 8 / reads; merging across gaps without reorder; and the reads that remain after an LRU (8/16/32 GB), where only the missed experts need reading: miss run length stock vs reordered, converted to a rate with the 0.9 MB -> 4 MiB curve (1.5 -> 2.38 GB/s).

| Measure | Value |
|---|---|
| Reads per layer: identity / held-out / in-sample | 7.54 / 5.95 / 4.46 |
| Run length: identity / held-out / in-sample | 1.06 / 1.35 / 1.80 (screen 1.06 / 1.29-1.39 / 1.77-1.81) |
| Gap merge without reorder, gap 1 / gap 3 | run 1.12 / 1.26 (screen 1.12-1.13 / 1.25-1.26); slots read 8.41 / 10.29 |
| Miss run length after LRU, stock -> reordered (8 / 16 / 32 GB) | 1.03 -> 1.30; 1.01 -> 1.10; 1.02 -> 1.13 |
| tok/s gain, serial / overlap at 8 GB | 1.045x / 1.073x |
| tok/s gain at 16 and 32 GB | 1.003x / 1.000x |

Verdict: refuted as a lever. Reproduces the screen's reads and run lengths; the kill rule "held-out run length below 1.5" is triggered (1.35). After the cache the gain is smaller than the screen's 1.1x (1.045x at 8 GB, DISAGREES slightly) and nil above 8 GB, because missed experts are scattered.

OLMoE (64 experts, 16 layers, 8 prompts x 62 tokens, leave-one-prompt-out): reads per layer 7.15 stock -> 5.08 held-out, run length 1.12 -> 1.58. So on the smaller expert pool the held-out run length (1.58) is above 1.5 and the kill rule is not triggered there; the 125B geometry (512 experts) is expected to be worse than the 128-expert trace (screen), not measured.

Published claims vs measured: mbolt reports 1.55-1.63x faster streaming and 2.23x at storage level in a memory-squeezed setting (self-reported, compared with stock mmap; not reproducible here, mbolt not installed); our reorder-only gain over explicit reads + LRU is at most 1.07x. LLM-in-a-flash found co-activation bundling a negative result: consistent with held-out gains staying small. MoE-Infinity has no layout.

Unmeasured: mbolt itself, 125B adjacency, real disk timing of a reordered file.

Catalog rows mapped to this family test:
- [screen 1 #20] published method: mbolt - profile-guided co-activation layout optimization for GGUF MoE with explicit-read prefetch
- [screen 1 #20] published method: LLM in a flash (Apple), row-column bundling and co-activation bundling
- [screen 1 #20] published method: MoE-Infinity

## [screen 1 #21] Draft K tokens, verify in one batched pass

Test: replay a verify-cycle decoder on each real trace: a window of K+1 consecutive trace tokens is served by one pass (distinct experts per layer through an LRU cache), tokens accepted = 1 + a geometric run (acceptance a, seeded), window advances by the accepted count. Bytes ratio = misses per accepted token over plain decode with the same cache. Time per accepted token = IO at 2.38 GB/s serial + compute; compute either flat (K+1 decode computes at 55 GFLOP/s) or the unproven fast-small-batch case (12 GFLOP per position at 311 GFLOP/s).

| Cache (footprints) | K=1 bytes ratio | K=3, a=0.7 | K=7, a=0.7 | flat speedup (K=1, a=0.9) | fast-bound speedup (K=1, a=0.9) |
|---|---|---|---|---|---|
| 2.0 | 0.97-0.98 | 2.13 | 2.54 | 0.985 | 1.65 |
| 4.0 | 1.00 | 1.00 | 1.45 | 0.965 | 2.18 |
| 7.5 | 1.00 | 1.00 | 1.01 | 0.952 | 3.65 |

Ranges over all 36 cells: bytes ratio at K=1 0.973-1.002; at K>=2 0.997-4.213; flat-compute speedup 0.237-0.985 (never above 1); fast-bound speedup 0.388-3.649.

Verdict: confirmed dead on IO (screen: 0.96-1.0 at K=1, up to 4.17x above; reproduced). Flat-compute speedup never exceeds 1.0 (max 0.985). The only upside is the unproven small-batch compute bound; the llama-bench record below does not support it (batch-8 per-token cost 0.70 of decode on OLMoE, p=4 at least 0.9; the screen's kill threshold was 0.4).

Published claims vs measured: MoE-SpeQ (2.34x over the best offloading framework on Phi-MoE) uses the small draft to predict experts for prefetch, i.e. a prefetch gain, not fewer bytes; our simulation only tests the verify-union bytes, and the prefetch gain is not measured here.

Unmeasured: real draft acceptance (a is a scenario value), the experts of rejected positions (trace tokens stand in), n-gram acceptance (llama-lookup not run), prefetch-by-draft.

Catalog rows mapped to this family test:
- [screen 1 #21] published method: MoE-SpeQ

## [screen 2 #0] NPU/FPGA co-processor for prefill GEMMs

Test (arithmetic + real coverage): 500-token prefill = 6e12 FLOP; compute at 311 GFLOP/s vs I/O of the 69 GB file at 2.4 / 3.1 / 1.56 GB/s; experts touched by n consecutive real tokens; TTFT with the proposal's 900 GFLOP/s; break-even prompt length.

| Measure | Value |
|---|---|
| Compute, 500 tokens at 311 GFLOP/s | 19.3 s (at the 55 GFLOP/s decode rate: 109 s, the claim's baseline; 5.66x too slow) |
| I/O of 69 GB at 2.4 / 3.1 / 1.56 GB/s | 28.8 / 22.3 / 44.2 s |
| I/O over compute | 1.49 / 1.15 / 2.29 |
| Overlapped TTFT gain from 900 GFLOP/s | exactly 1.0x at all three rates |
| Serial (no overlap) gain | 1.36 / 1.44 / 1.25x |
| Break-even prompt length (compute = I/O) | 745 / 576 / 1146 tokens; at 2000 tokens gain 2.68x, at 8000 tokens 2.89x |
| Share of the 128 experts touched by n consecutive real tokens, n=10/50/100/200 | 0.24-0.27 / 0.39-0.48 / 0.46-0.56 / 0.55-0.66 (random model: 0.48 / 0.96 / 1.0 / 1.0) |

Verdict: confirmed for the screen's case (500 tokens: I/O-bound, NPU 1.0x), with two caveats. (1) The claim "prefill touches nearly every expert once" DISAGREES with the real traces: one prompt's tokens touch only 55-66 % of a layer's experts after 200 tokens (random routing would say 100 %). If a 500-token prompt really read only 60-75 % of the file, prefill I/O would fall to 12.8-20.7 s, below the 19.3 s compute, and an NPU could gain 1.16-1.5x (table `coverage_scenarios` in the output). This is a measurement on decode traces of one prompt (not prefill routing) and on 128 experts; the 125B prefill coverage is unmeasured. (2) Above about 600-1150 tokens the prefill is compute-bound and the gain is real (2.7-2.9x at 2000-8000 tokens with a 900 GFLOP/s device, if it exists).

Published claims vs measured: the Apple-NPU MoE paper (NPU prefill worst without its optimisations) and ProMoE (2.20x prefill speedup, GPU/PCIe) are not a laptop-disk setting; DuoServe-MoE (two-stream prefetch/compute) and FluxMoE (pipelined streaming) and ssd-llm (layer-major batch prefill) describe the overlap we already assume; Deltafin (6.3 min TTFT for 512 tokens, blog, unverified) shows prefill is the limit when streaming from NVMe, consistent with I/O-bound prefill.

Unmeasured: any NPU/FPGA throughput; prefill GFLOP/s on the 125B. The 311 GFLOP/s figure is an earlier measurement; the run now (OLMoE Q3_K_S, `llama-bench -p 512`, 4 threads, loaded machine) gave 120 t/s = about 313 GFLOP/s at an assumed 2.6 GFLOP/token (1.3B active parameters, assumption), consistent.

Catalog rows mapped to this family test:
- [screen 2 #0] published method: Efficient Mixture-of-Experts LLM Inference with Apple Silicon NPUs
- [screen 2 #0] published method: ProMoE
- [screen 2 #0] published method: DuoServe-MoE
- [screen 2 #0] published method: ssd-llm
- [screen 2 #0] published method: Deltafin fork, Kimi K3 on a MacBook (blog, unverified)
- [screen 2 #0] published method: FluxMoE

## [screen 2 #1] Expert GEMV in computational-storage FPGA drives

Test: arithmetic (1 NVMe, 2 striped NVMe, compute ceiling), per-drive in-device I/O, imbalance when each whole expert lives on one of two drives (busiest drive's bytes over mean, per token-layer, real traces and a seeded random 512/top-10 model), and the Mac NVMe rate against reads in flight (2.7 MB random reads, F_NOCACHE, on the Qwen3-30B file).

| Measure | Value |
|---|---|
| 1 NVMe / 2 NVMe / compute ceiling | 2.39 / 4.78 / 4.58 tok/s (screen 2.4 / 4.8 / 4.57) |
| Per drive: 0.648 GB per token | 0.208 s at 3.11 GB/s, just under compute 0.218 s |
| Imbalance, real 128/top-8, half / mod / hash placement | 1.267 / 1.261 / 1.266 |
| Imbalance, random 512/top-10 | 1.241 |
| CSD pair with imbalance: overlapped / serial | 3.87 / 2.10 tok/s (ideal 4.58) |
| Ideal gain over best path at 8 GB, serial / overlapped | 2.08x / 1.08x (screen 2.1 / 1.1) |
| At 16 / 32 GB best path overlapped | 4.58 (already the ceiling): CSD gain 1.0x |
| Per-drive rate 4.14 or 7.0 GB/s | ideal = ceiling 4.58 in both |
| Recorded NVMe rate, 1 / 4 / 8 / 16 reads in flight (2.7 MB) | 1.63 / 3.04 / 3.70 / 4.14 GB/s |

Verdict: confirmed (nothing beyond striping). Two ordinary NVMe drives in stripe give the same I/O time as the ideal CSD pair, and both are capped by compute at 4.58 tok/s. Whole-expert placement costs 24-27 % imbalance, so a CSD pair would deliver about 3.9 tok/s overlapped (screen: 3-4). One disagreement: the screen expected the Mac NVMe to be flat beyond 4 reads in flight at about 3.1 GB/s; it rises to 4.14 GB/s at 16 (DISAGREES), so the 3.11 GB/s figure is not a drive ceiling. That makes a plain two-NVMe stripe even less of a constraint. The screen's survival condition (7 GB/s internal per drive) does not change the ceiling: compute caps at 4.58 either way.

Published claims vs measured: InstInfer (FPGA CSD, offloads KV and attention, up to 11.1x over SSD baselines) and HillInfer (SmartSSD for KV eviction; notes SSD bandwidth remains the bottleneck) are different workloads, the second supports the internal-flash bound; Conduit (simulator, no MoE numbers), Samsung SmartSSD (scan-heavy database gains, no GEMV numbers) and DockerSSD (7.9x, self-reported) give no expert-weight evidence.

Unmeasured: a CSD's internal NAND bandwidth and FPGA GEMV rate, dispatch overhead of 480 slice calls per token, any real two-drive stripe.

Catalog rows mapped to this family test:
- [screen 2 #1] published method: InstInfer: in-storage attention offloading to computational storage drives
- [screen 2 #1] published method: HillInfer: long-context edge LLM inference with SmartSSD
- [screen 2 #1] published method: Conduit: near-data processing across SSD compute resources
- [screen 2 #1] published method: Samsung second-generation SmartSSD (Versal Prime VM1802)
- [screen 2 #1] published method: DockerSSD in-storage processing

## Recorded runs (Mac, shared, noisy)

- Disk, Qwen3-30B-A3B-Q4_K_M GGUF (18.6 GB), F_NOCACHE: `bench-pread` 8 tokens of the db trace, 8 threads: merged plan 8583 reads 2.48 GB/s, unmerged 9216 reads 2.72 GB/s; cold mmap of 2 tokens 0.277 GB/s. `bench-layout` (400 triples, 7 interleaved repeats): scattered 3.04 / 2.88, contiguous 3.89 / 3.71 GB/s. `bench-threads` (2.7 MB, 3 s each): 1.63 / 3.04 / 3.70 / 4.14 GB/s at 1 / 4 / 8 / 16. An earlier layout run was discarded: the freshly written scratch copy was served from the page cache (8.5 GB/s); fixed by F_NOCACHE on the writer and an fsync, then the first contiguous repeat is still slower (cold), the median is reported.
- llama-bench, OLMoE Q3_K_S, 4 threads, load average 7-16: decode 20.4 t/s median (13-23 across runs); p=8 29.0 t/s; p=4 9.9 median (best 22); p=1,2 4-6 t/s; p=512 120 t/s. Per-token cost of batch p relative to decode (decode/pp) at p=8 is 0.70; at p=4 between 0.9 and 2. Too noisy to claim more than "no sign that small batches are cheap".
- OLMoE traces: 8 prompts x 64 tokens (first complete block dropped = 62 decode tokens), recorded with GGML_MOE_TRACE, used for the OLMoE lines under #17 and #20 (`olmoe_study` in `f_C.py`, run by hand).

## What was not done

- No real multi-prompt batched decode (llama-parallel), no llama-lookup acceptance, no mbolt, no NPU/FPGA/CSD hardware, no 125B model.
- All 125B numbers are the 30B traces mapped through token-footprints plus constants listed above.


## Corrections from the deep paper check (papers/DEEP_SUMMARY.md)
- Paging the Experts does bundle gate/up/down per expert (one 64 KiB-aligned block); the earlier "no bundling layout" line is wrong.
- Published stock-mmap gaps are larger than our emulation (2.6-6.9x on GPU/phone/Apple setups); our 1.5-2.4x came from an mmap emulation, not stock llama.cpp.
- Mac naive baseline at about 8x file-to-cache: 0.33 tok/s (9 GB wired ballast), fully warm 8.0 tok/s.
