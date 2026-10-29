# Paper check summary (228 catalog rows, 12 batches)

Method: each paper/page was opened with a fetch tool (which often returns a model-written digest, not raw text; where only abstracts or search snippets were read the row says so in P01..P12.md). Counts: **127 accurate, 76 partly accurate, 18 wrong, 7 unverified** catalog summaries; 219 pages opened, 9 not.

Transfer to our regime (CPU-only, 8-32 GB RAM, expert weights streamed from NVMe, batch 1): **no for 89 rows, partly for about 45, the rest unclear or unverified**. Every headline speedup found is on a GPU, a phone, a simulator, or with the model in DRAM; none is CPU-only streamed at batch 1 on a 125B-class model.

## Catalog rows that are wrong (fix before quoting)

| row | batch | title | what is wrong |
|---|---|---|---|
| P-0025 | P01 | Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster | ERROR ALSO IN pubtech/A.md line 105 ('large drops ... when k is lowered to 3'). Correct reading: the paper is evidence for batch-aware reuse, not for or against a plain top-k cut. Read via html version. |
| P-0075 | P03 | SMIDT: High-Performance Inference Framework for MoE Models with Dynami | AAAI 40(27), 2026-03-14 |
| P-0084 | P04 | Fiddler: CPU-GPU Orchestration for Fast Inference of MoE Models (arXiv | 8.2x/10.1x not on the page opened; possibly an earlier version (unverified) |
| P-0094 | P04 | gpt-oss-120b on an 11.3 GB Phone (BigMoeOnEdge docs) | O_DIRECT was applied to dense weights, not experts or 'routing weights' |
| P-0101 | P04 | llama.cpp Discussion #19883 Why does it takes the same amount of RAM a | also checked issue #19825 (swap death spiral on M4 Pro), no maintainer statement either |
| P-0113 | P05 | llama.cpp issue #19825 'Managed SSD offloading for MoE to prevent macO | Closest public precedent for our model class; competitor claim unverified |
| P-0146 | P07 | A Multi-Engine Dataflow for MoE Decoding on AWS Trainium (CARDAN) (Ma, | Full results tables not read (unverified beyond abstract). Notable: the closest published combination of our families 2 and 5, gains modest (1.15-1.31x). |
| P-0205 | P09 | OpenAI forum: Prefill-only requests that consume input tokens without  | hosted prefix warm, not typing-time |
| P-0207 | P09 | A Photonic-CXL Memory Appliance for Scalable KV Cache Management in LL | abstract only; pdf unreadable (abstract-level only) |
| P-0214 | P10 | Understanding MoE Offloading (SomeOddCodeGuy, DEV Community, 13 Oct 20 | Url does not support the catalog summary. A real source for llama.cpp RPC (docs or repo) is needed. |
| P-0217 | P10 | OD-MoE (the catalog url points to the wrong paper: it is FIRM-MoE, Che | Transport not confirmed on the AAAI page; the arXiv OD-MoE page gives 1 Gbps Ethernet. |
| P-0219 | P10 | Multi-node local LLM inference guide 2026 (fungies.io, 1 Jul 2026) | Replace by an actual benchmark source for RPC over 10GbE before citing. |
| P-0230 | P10 | Effective MoE-based LLM Compression by Exploiting Heterogeneous Inter- | Rank allocation fuses effective rank and routing frequency (xi about 0.7-0.8). |
| P-0233 | P11 | Run distributed inference with llama.cpp on Arm-based AWS Graviton4 in | catalog URL truncated; opened learn.arm.com path ending distributed-inference-with-llama-cpp/ |
| P-0242 | P11 | DIMoE-Adapters: Dynamic Expert Evolution for Continual Learning in VLM | no numbers in text |
| P-0261 | P12 | NVIDIA blog: Boosting MoE Training Throughput with Advanced Fusion Ker | 15-20% Mixtral and 0.999x OLMoE figures not on this page; page is training not inference |
| P-0271 | P12 | BentoML: Structured Decoding in vLLM, a gentle introduction (Aaron Pha | page does not discuss batching of forced tokens; 5x is mask overhead |
| P-0272 | P12 | Earley-Driven Dynamic Pruning for Efficient Structured Decoding (Forma | method paper, no DOMINO or jump-forward; gains from mask cost |

## Unverified or unopened

- P-0058 (P03): Which Expert Misses Are Worth Waiting For? Substitution and Caching for MoE Inference on a
- P-0097 (P04): Apple LLM in a flash + HN comment (Roxxik) on io_uring layer in llama.cpp fork
- P-0119 (P05): BigMoeOnEdge (Helldez; repo docs/limitations.md; news article url 404)
- P-0137 (P06): Samsung Electronics Develops Second-Generation SmartSSD (BusinessWire, 20 Jul 2022)
- P-0144 (P07): llama.cpp IQ2 codebook-grid quants (opened the originating PR #4773 'SOTA 2-bit quants', i
- P-0150 (P07): diskllm (PyPI)
- P-0151 (P07): llama : save and restore kv cache for single seq id (PR #6341, kaetemi; opened 27 Mar 2024
- P-0178 (P08): VPTQ: Extreme Low-bit Vector Post-Training Quantization for LLMs (Microsoft; EMNLP 2024); 
- P-0257 (P12): Mini-SGLang chunked prefill with radix cache (docs mirror)
- P-0265 (P12): ASAP: A Disaggregated and Asynchronous Inference System for MoE Prefill (arXiv 2606.22541,

## Per-family accuracy of the catalog summaries

| family | accurate | partly | wrong | unverified |
|---|---|---|---|---|
| s1 #0 | 3 | 4 | 0 | 0 |
| s1 #1 | 2 | 3 | 0 | 0 |
| s1 #2 | 5 | 2 | 0 | 0 |
| s1 #3 | 3 | 3 | 1 | 0 |
| s1 #4 | 3 | 1 | 0 | 0 |
| s1 #5 | 3 | 2 | 0 | 0 |
| s1 #6 | 3 | 2 | 0 | 0 |
| s1 #7 | 3 | 2 | 0 | 0 |
| s1#8 | 2 | 1 | 0 | 1 |
| s1#9 | 3 | 2 | 0 | 0 |
| s1#10 | 5 | 3 | 0 | 0 |
| s1#11 | 4 | 1 | 1 | 0 |
| s1#12 | 0 | 3 | 1 | 0 |
| s1#13 | 4 | 1 | 0 | 0 |
| s1#14 | 2 | 1 | 1 | 1 |
| s1#15 | 1 | 1 | 1 | 0 |
| screen1 #16 | 3 | 2 | 0 | 0 |
| screen1 #17 | 2 | 1 | 0 | 0 |
| screen1 #18 | 0 | 1 | 1 | 0 |
| screen1 #19 | 2 | 1 | 0 | 1 |
| screen 1 #20 | 2 | 1 | 0 | 0 |
| screen 1 #21 | 1 | 0 | 0 | 0 |
| screen 2 #0 | 4 | 2 | 0 | 0 |
| screen 2 #1 | 3 | 1 | 0 | 1 |
| screen 2 #2 | 4 | 2 | 1 | 0 |
| screen 2 #3 | 6 | 0 | 0 | 1 |
| screen 2 #4 | 0 | 4 | 0 | 0 |
| screen 2 #5 | 4 | 1 | 0 | 0 |
| screen 2 #6 | 3 | 1 | 0 | 0 |
| screen 2 #7 | 1 | 0 | 0 | 0 |
| screen 2 #8 | 6 | 1 | 0 | 0 |
| screen 2 #9 | 1 | 2 | 0 | 0 |
| 10 | 2 | 3 | 0 | 0 |
| 11 | 2 | 3 | 0 | 0 |
| 12 | 4 | 2 | 0 | 0 |
| 13 | 3 | 0 | 2 | 0 |
| screen2 #14 | 1 | 2 | 1 | 0 |
| screen2 #15 | 0 | 2 | 2 | 0 |
| screen2 #16 | 4 | 1 | 0 | 0 |
| screen2 #17 | 4 | 0 | 1 | 0 |
| screen2#18 | 1 | 1 | 1 | 0 |
| screen2#19 | 3 | 2 | 1 | 0 |
| screen2#20 | 3 | 1 | 0 | 0 |
| screen2#21 | 2 | 3 | 0 | 0 |
| 22 | 2 | 1 | 0 | 1 |
| 23 | 3 | 1 | 1 | 1 |
| 24 | 3 | 1 | 2 | 0 |
| 25 | 2 | 1 | 0 | 0 |

## What the papers say about our families (from the batch verdicts)

- Confirmed by outside work: explicit large reads/pread beat mmap (Flash-MoE ~5x, SGLang expert pack, llama.cpp slot-pool RFC), but they are known; co-activation reorder adds ~5% over prefetch (mbolt) matching our 1.045x; cross-layer prefetch gives ~1.0-1.08x in disk-bound regimes (three independent sources); static pins lose to LRU; LRU vs ARC differ by <0.01 hit; batching/large-K gains are GPU-offline numbers; naive speculation under offload loses to plain decoding; low-rank/shared-basis and VQ at ~1 bit lose real quality; lookup-table kernels lose to int8-dot on AVX2.
- Novelty threats (known work): router bias for cache hits (Qualcomm 2412.00099, llama.cpp CPU, batch 1); SliceMoE (bit-sliced expert caching, up to 1.81x decode); HOBBIT; FlashMoE (ML cache policy on SSD experts); SpecMD (Least-Stale eviction); ExFold and Elbow routing (prefill/adaptive top-k); MoE-Spec/EcoSpec/DraftExpert/SpecMoE (speculation with union cost); MAFDET/LiveMind/PredGen (prefill while typing); llama.cpp discussion #23324 (pread slot pool + LRU); 2605.08575 (gate-first sparsity inside experts).
- Challenges to our assumptions: SpecMD and FlashMoE argue LRU fits expert access poorly (their gap to optimal matches our Belady-LRU gap); one paper warns recency policies are inflated 27-29% by inconsistent replay (check our simulators); Flash-MoE reaches 17.5 GB/s SSD reads and found prefetch hurt (-73%) on Apple silicon; prefill touches 80-98% of experts on 32-64-expert models (our decode-trace proxy gives 55-61% for 128 experts), so our prefill numbers may be optimistic.
- Errors in our own reports found by the check: pubtech/A.md misreads P-0025 (the 43.0->22.2 figures are experts per layer, not accuracy) and mislabels a Cascade row as P-0039 (it is P-0010); P-0113's numbers come from a Reddit post, not the issue it cites; several catalog links point at different papers (P-0183, P-0184, P-0217, P-0218, P-0146, P-0242).
- Closest public precedent for our model class: a 125B-class 512-expert top-10 MoE run from disk with direct I/O plus prefetch (20 tok/s warm vs ~2 stock mmap, RTX 3060, unverified Reddit claim) and Flash-MoE (397B, 512 experts, 48 GB Mac, 4.36 tok/s). Neither is CPU-only on 8-16 GB.
