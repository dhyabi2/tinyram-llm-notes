# deep_P01b: deep re-check from raw text, rows P-0017..P-0022, P-0024..P-0030 (families s1 #2 and s1 #3)

Method: every row read from raw text (arXiv PDFs via pdftotext, GitHub API/raw files, curl'd markdown). Work folder: scratchpad deep_P01b. All 13 rows raw_text_read=true, 0 unread.


## P-0017  Fate: Fast Edge Inference of Mixture-of-Experts Models via Cross-Layer Gate (Zhiyuan Fang, Zicong Hong, Yuegui Huang, Yufeng Lyu, Wuhui Chen, Yue Yu, Fan Yu, Zibin Zheng; Sun Yat-sen Univ / HKUST / Huawei / Peng Cheng Lab; arXiv 2502.12224v2, cs.AI, 7 May 2025)

- Read: arXiv 2502.12224v2 (PDF, pdftotext)
- Setup: Models: DeepSeekMoE-16.4B (top-6 of 64 routed) and Qwen1.5-MoE-14.3B (top-4 of 60), both with shared experts. High-end PC: RTX 3090 24 GB, i9-9900X, 128 GB host RAM, PCIe 3.0 x16 (16 GB/s quoted). Low-end PC: GTX 1080Ti 11 GB, E5-2650 v4, 64 GB host RAM, PCIe 1.0 x16 (4 GB/s). Experts live in host RAM (no SSD). Prefill prompt lengths 128/256/512 (ChatGPT-prompts, HumanEval, GSM8K); decode input <=64, output <=1024. Baselines: Load-on-Demand (LoD) and Expert-Activation-Path (EAP, same cache as Fate but path-based prefetch). Batch size not stated. Cache: shallow-favoring ARC. Quantisation: HQQ, INT4 for cache, INT2 for the least-popular experts during prefill transfers.
- Numbers (what | value | baseline | ratio | where):
  - Prefill speedup, high-end PC, DeepseekMoE/Qwen1.5-MoE, prompt 128-512 | up to 4.5x over LoD, 1.9x over EAP | LoD, EAP | as stated (bars in Fig 11 not machine-readable) | Sec 5.2.1, Fig 11
  - Prefill throughput, DeepseekMoE, 512-token prompt, high-end PC | 804 tok/s | LoD/EAP (values not extractable) | n/a | Sec 5.2.1
  - Prefill speedup, low-end PC | up to 1.8x over LoD, 1.5x over EAP | LoD, EAP | as stated | Sec 5.2.1
  - Decode, high-end PC | Qwen1.5-MoE 14 tok/s; DeepseekMoE 3.7x over LoD, 2.2x over EAP | LoD, EAP | as stated | Sec 5.2.2, Fig 12
  - Decode, low-end PC | up to 2.3x over LoD, 1.6x over EAP | LoD, EAP | as stated | Sec 5.2.2
  - Abstract decode claim | up to 4.1x and 2.2x | LoD, EAP | 4.1x not found in body text (body says 3.7x/2.3x) | Abstract and Intro vs Sec 5.2.2
  - Average experts activated per layer in prefill, MMLU, Qwen1.5-MoE | 55.9 of 60 (93%) | - | - | Sec 4.6 (hybrid quantization paragraph)
  - Next-layer expert prediction accuracy using Gate_in_i (no training) | 78.79% overall; prefill ~>99%; decode lowest 76.94%; with the extra similar-routing-weights step 97.15% | other two input positions (83.99%, 88.83%, 93.05% in Table 1) | - | Table 1, Sec 4.2-4.3
  - Accuracy of Fate (INT4 cache + INT2 least-popular during prefill transfer) vs BF16 | MMLU 59.7/44.3 (high-end) and 58.4/42.0 (low-end) vs 59.6/44.4; GSM8K 60.2/22.3 and 59.9/20.2 vs 62.4/21.2; HumanEval 32.3/26.8 and 34.1/24.6 vs 34.8/25.6 (Qwen/Deepseek) | BF16 | average degradation 0 / -0.55 / -0.15 (high-end), -1.8 / -1.75 / -0.85 (low-end) | Table 3
- Catalog / first-pass numbers checked:
  - [yes] Prefill activates a large number of experts and is heavy on I/O: 55.9 of 60 experts per layer on MMLU for Qwen1.5-MoE; paper calls prefill I/O intensive
  - [no] No sorting scheme: Fate reorders the prefill computation by predicted expert popularity so popular (longer) experts hide the I/O of others, and sends the least popular as INT2 (Sec 4.6 and 5.2.1). It is ordering by popularity, not expert-major grouping of tokens, but 'no sorting scheme' is wrong.
- Transfers to our regime: **partly**. Confirms that a prompt of a few hundred tokens touches nearly all experts of a layer (93% here, 60 experts, top-4) so a prefill cannot be saved by sparsity alone. Breaks for us: GPU+PCIe pipeline, experts in 64-128 GB host RAM (no SSD), 60-64 experts per layer rather than 512, top-4/6 instead of top-10, and the gains are against load-on-demand and a path predictor, not against llama.cpp. The INT2 trick is lossy (loss measured as <=1.8 points average, searched to stay within 1% by heuristic).
- Quality loss: Table 3: average degradation 0 to -0.55 points on the high-end PC and -0.85 to -1.8 on the low-end PC (MMLU, GSM8K, HumanEval, two models). A heuristic search fixes the INT2 share p so that loss stays within 1% (p=25% for Qwen1.5-MoE).
- Ablation: Fig 14 (decoding ablation, Qwen1.5 high-end): LoD, +Prefetch, +Cache, +Quantization; text: prefetching alone gives about 30% average speedup, caching up to 2x, quantization adds more (3.7x total decode). No ablation isolating the popularity reordering from the INT2 transfer in prefill.
- Limitations (authors): No dedicated limitations section found. Authors note low-end PCIe 1.0 limits overlap; shared experts and fine-grained routing assumed (cosine similarity of adjacent gate inputs > 83%).
- Contradicts / bears on our findings: No contradiction. Its 55.9/60 prefill union (93%) supports the screen's assumption that a real prefill union is high (0.75-0.95) and is higher than our decode-trace proxy (0.55 at 128 tokens); different model (60 experts, top-4) so it is not a measurement for 512-expert top-10.

## P-0018  DAOP: Data-Aware Offloading and Predictive Pre-Calculation for Efficient MoE Inference (Yujie Zhang, Shivam Aggarwal, Tulika Mitra; NUS; arXiv 2501.10375v3 cs.DC, dated 21 Aug 2026 on the page; DATE 2025 per footer)

- Read: arXiv 2501.10375v3 (PDF, pdftotext)
- Setup: Models: Mixtral-8x7B (32 blocks, 8 experts, top-2, 45.1B) and Phi-3.5-MoE (32 blocks, 16 experts, top-2, 40.3B). Hardware: i9-10980XE (18 cores), 130 GB host RAM, RTX A6000 48 GB (768 GB/s), PCIe 4.0 (64 GB/s quoted). Batch size 1 (Sec V-A c). Experts on CPU memory; expert cache ratio (ECR) = share of expert slots on GPU, 46.9% for the headline runs. Baselines: MoE-OnDemand, DeepSpeed-MII, Mixtral-Offloading, Fiddler. Implementation on HF Transformers. Calibration of dominant experts on ShareGPT. Input/output configs e.g. [256,512].
- Numbers (what | value | baseline | ratio | where):
  - Mixtral 8x7B, [256,512], ECR 46.9% | 4.52 tok/s | Fiddler (about 3.22 tok/s implied by +40.4%) | 1.404x (stated +40.4%) | Sec V-B, Fig 9
  - Phi-3.5-MoE, [256,512], ECR 46.9% | 8.21 tok/s | Fiddler | not extractable | Sec V-B, Fig 9
  - Average improvement over Fiddler across cache ratios, input/output 256 | +35.4% | Fiddler | 1.354x | Fig 10
  - Mixtral at 25% ECR / Phi at 25% ECR | 3.23 / 5.03 tok/s | - | - | Sec V-B
  - Other baselines (MoE-OnDemand, DeepSpeed-MII, Mixtral-Offloading) | each below 1 tok/s on Mixtral (Table IV energy rows imply 0.59-2.13 tok/kJ vs 10.06 Fiddler) | - | DAOP/DeepSpeed-MII from tok/kJ: 14.37/0.59 = 24x energy | Sec V-B, Table IV
  - Energy efficiency tok/kJ, Mixtral / Phi | 14.37 / 27.07 | Fiddler 10.06 / 17.15 | 1.43x / 1.58x (text says average 1.50x) | Table IV
  - Abstract: up to 8.20x vs caching/prefetching, 1.35x vs offloading | 8.20x / 1.35x | not itemised in text | cannot be reproduced from the body numbers (4.52/1.404=3.22 for Fiddler; 1.35x does not match 1.40x or 1.354x exactly) | Abstract, Intro, Conclusion
  - Accuracy, prefill-only metric (first output token), Mixtral at ECR 25% | HellaS 66.80, Arc-e 84.39, Arc-c 63.82, PIQA 82.59, WinoG 81.77, MMLU 70.47 | official 66.96/83.10/63.74/83.60/81.69/70.60 | about -0.1 to +1.3 points | Table V
  - Accuracy over whole generation, Mixtral, ECR 100/62.5/50/37.5/25% | GSM8K EM 58.91/51.48/48.07/41.77/33.51; TriviaQA 71.59/70.98/70.60/70.13/69.08; BBH 49.36/47.63/47.10/47.14/46.61 | ECR 100% | GSM8K at ECR 25%: 33.51/58.91 = 0.57x | Table VI
- Catalog / first-pass numbers checked:
  - [yes] Counts tokens per expert during prefill to set expert activity: Algorithm 1 counts tokens assigned to each expert and swaps the most active CPU experts with the least active GPU experts (threshold SwapInOut 1.05)
  - [yes] Limits migration to the prefill stage to minimize I/O: 'expert migration is restricted to the prefill stage for cache adjustments, with this configuration maintained throughout decoding' (Sec III-B)
- Transfers to our regime: **no**. The prefill token counts decide which of 8-16 experts per layer sit on a 48 GB GPU while the rest are computed on the CPU (the Fiddler idea). Assumptions that break for us: a GPU decides, 130 GB RAM holds all experts (no SSD), top-2 models, 8-16 experts instead of 512. The only transferable idea is 'use the prompt's per-expert counts to choose what to keep resident for decode', which we already have as a cache-warm-up idea (also SliceMoE PCW, Fig 10).
- Quality loss: Pre-calculation is approximate (graceful degradation: a predicted CPU expert result used in place of the true one). Tables V/VI: negligible on commonsense tasks down to ECR 25%, but GSM8K exact match drops 58.91 -> 41.77 (ECR 37.5%) -> 33.51 (ECR 25%) on Mixtral and 86.88 -> 74.07 on Phi-3.5.
- Ablation: None isolating the prefill-count placement from the approximate pre-calculation; Fig 10 varies ECR only.
- Limitations (authors): Authors discuss the GSM8K drop (Section VI); models limited to two top-2 MoEs; GPU memory must hold the cache.
- Contradicts / bears on our findings: None. The reported prefill-driven placement is a decode-reuse effect, not a prefill-speed effect, so it does not bear on our s1 #2 (prefill as expert-major work) measurement.

## P-0019  Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models (Keisuke Kamahori, Tian Tang, Yile Gu, Kan Zhu, Baris Kasikci; Univ. of Washington; ICLR 2025 per page header; arXiv 2402.07033v3, 1 May 2025)

- Read: arXiv 2402.07033v3 (PDF, pdftotext)
- Setup: Mixtral-8x7B 16-bit (>90 GB, 32 layers x 8 experts = 256 experts, top-2). Env 1: Quadro RTX 6000 24 GB, PCIe Gen3 x16 (32 GB/s quoted), Xeon Gold 6126 (48 cores), 56/256 experts fit on GPU. Env 2: RTX 6000 Ada 48 GB, Gen4 x16, Xeon Platinum 8480+ (112 cores), 125/256 experts on GPU. Host RAM not stated in Table 1. Single request; baselines DeepSpeed-MII v0.2.3 (ZeRO-Infinity), Mixtral-Offloading (16-bit extension, offload_per_layer 7/5), llama.cpp b2956 (ngl 8/16). Scenarios: (a) 15 input/output length pairs (in 32-256, out 64-512), (b) long prefill TTFT with 512/1024/2048/4096-token inputs, (c) beam search width 4-16.
- Numbers (what | value | baseline | ratio | where):
  - Single-request end-to-end speed (a), avg of 15 configs x 2 envs | 1.26x faster | llama.cpp (best baseline) | as stated | Sec 4.2, Fig 4
  - Long-prefill TTFT (b), avg of 4 lengths | 1.07x better than DeepSpeed-MII, 1.65x better than Mixtral-Offloading (abstract: 1.30x) | DeepSpeed-MII, Mixtral-Offloading | 1.30x is not derivable from the two quoted ratios (arithmetic mean 1.36, geometric 1.33); llama.cpp was worse than both offloading baselines on long prefill | Sec 4.2, Fig 5
  - Beam search (c), avg of widths 4-16, input 32, output 64 | 11.57x | llama.cpp (only baseline supporting beam search) | as stated | Sec 4.2, Fig 6
  - Expert size | 3 matrices 4096x14336, >300 MB at 16-bit | - | - | Sec 3.2
  - CPU activation copy cost | <1% of total latency | - | - | Sec 3.3, Appendix A
- Catalog / first-pass numbers checked:
  - [yes] Reports 1.30x on long prefill: abstract and Sec 4.2 (average over the offloading baselines; not derivable exactly from the quoted per-baseline ratios)
  - [yes] Reports 1.26x on single-batch decode: 1.26x avg vs llama.cpp (Fig 4)
  - [no] Avoids moving weights during prefill by computing on the CPU instead: Opposite for long prefill: Sec 3.2 says CPU expert compute is advantageous when an expert's input is small, but for large inputs (prefill reaching ~a thousand tokens) the CPU compute latency exceeds the weight-transfer latency and the paper uses transfer + GPU compute. Algorithm 1 chooses per expert from a latency model (gpu_lat constant, cpu_lat linear in tokens). CPU-vs-copy is a decode/small-batch decision.
- Transfers to our regime: **no**. CPU compute of an expert is only preferred over copying when the PCIe copy of ~300 MB is slower than CPU compute; needs a GPU and all experts in host RAM (>90 GB), so there is no SSD in the loop. Our regime is already CPU-only. The prefill claim is the opposite of the catalog line: for long prompts Fiddler copies weights to the GPU.
- Quality loss: None (exact computation, 16-bit weights).
- Ablation: Appendix covers latency-model constants and activation-sparsity (Table 2: SiLU outputs) and sensitivity to datasets; no cache vs placement ablation in the main text.
- Limitations (authors): No limitations section located; the paper assumes non-expert layers fit on the GPU (<2B params) and that popularity profiling is available for expert placement.
- Contradicts / bears on our findings: None. It does contradict the catalog summary mechanism (see above) and the first-pass note, which could not verify the prefill mechanism.

## P-0020  FlashMoE: Reducing SSD I/O Bottlenecks via ML-Based Cache Replacement for Mixture-of-Experts Inference on Edge Devices (Byeongju Kim, Jungwan Lee, Donghyeon Han, Hoi-Jun Yoo, Sangyeob Kim; KAIST / Chung-Ang / Yonsei; arXiv 2601.17063v2 cs.LG, dated 5 Oct 2026 on the page)

- Read: arXiv 2601.17063v2 (PDF, pdftotext)
- Setup: Models: OLMoE-1B-7B (64 experts/layer, top-8) and Qwen3-30B-A3B (128 experts, top-8). Desktop: Ryzen 5 9600X, RTX 5070 Ti 16 GB (expert cache lives in VRAM, 14 GB), 4x16 GB DDR5-6000 with all but ~1 GB locked, SK hynix P51 PCIe5 NVMe 7.4 GB/s read, Ubuntu 24.04, PyTorch 2.7. Experts stored as individual .pt files; one expert load from SSD ~3 ms vs FFN 158 us; expert loading >70% of decode time (Fig 5). Cache sizes 12-24 experts per layer (OLMoE of 64; Qwen3 of 128 = 8.8-14.4 GB). Baselines: Fiddler, DAOP, llama.cpp, HF Transformers; policies LIFO/LFU/LRU/ARC/LeCaR. ML policy: tiny per-layer FFN (~113 KB) fed recency/frequency, trained to imitate Belady on 512 TriviaQA samples; evaluation on a held-out TriviaQA split. Decode batch size not stated.
- Numbers (what | value | baseline | ratio | where):
  - Initial model load, non-expert weights only | 4x faster than llama.cpp, 6.8x faster than Fiddler/DAOP | llama.cpp / Fiddler / DAOP | as stated | Sec 4.2, Fig 7(a,d)
  - Load + prefill latency | total 2.5x over llama.cpp, 4.1x over Fiddler and DAOP | same | as stated | Sec 4.2, Fig 7(a,d)
  - Share of experts loaded in prefill vs prompt length | 47% at 32 tokens, 58% at 64, 64% at 128, 67% at 256 | - | model not named in that sentence | Sec 4.2
  - Cache hit rate, OLMoE-1B-7B, ML policy | +21% vs LRU, +51% vs LFU (relative), up to +28% vs ARC, +21% vs LeCaR; I/O reduction 22% vs LRU, 35% vs LFU | LRU, LFU, ARC, LeCaR | as stated | Sec 4.3, Fig 6
  - Decode token rate, ML vs LRU | +22% OLMoE, +7% Qwen3-30B | LRU | 1.22x / 1.07x | Sec 4.4, Fig 7(b,e)
  - Abstract: hit rate +51% over LRU/LFU; up to 2.6x vs existing systems | 51% / 2.6x | - | 51% is the LFU figure (LRU: 21%); 2.6x is not stated in the body (Fig 7c,f bars only) | Abstract
- Catalog / first-pass numbers checked:
  - [yes] SSD-offloaded MoE on edge devices (OLMoE, Qwen3-30B-A3B): as read
  - [yes] Compares with llama.cpp, Fiddler and DAOP: also HF Transformers
  - [yes] Reports loading 4x faster than llama.cpp: but it is start-up of non-expert weights only, not streaming throughput
  - [absent] Prefill figures (catalog text truncated): load+prefill 2.5x over llama.cpp and 4.1x over Fiddler/DAOP, Fig 7
- Transfers to our regime: **partly**. Closest published system on storage: real PCIe5 NVMe, RAM locked to ~1 GB free, batch-1 style decode, per-layer expert cache. Breaks for us: the cache is in 16 GB VRAM of a GPU, CPU DRAM is not the cache, SSD 7.4 GB/s vs our NVMe, models of 64-128 experts and 7-30B, ML policy trained on in-distribution traces with a Belady oracle, and the gain shrinks with the larger model (+7% decode over LRU on Qwen3-30B). The prefill method is simply: collect the experts needed by the whole token batch and load each once per iteration (Sec 2), i.e. the same expert-major idea as our s1 #2 with no custom sorting.
- Quality loss: None (exact; policy only changes what is cached).
- Ablation: Cache policy sweep (Fig 6, 7b,e) isolates replacement policy at equal cache size; no ablation of prefill grouping vs cache.
- Limitations (authors): Not stated as a section. Visible: trained only on TriviaQA traces; DAOP/Fiddler baselines are DRAM-based; Fiddler/DAOP initialised with popular experts from the same training traces.
- Contradicts / bears on our findings: Consistent with our numbers: a prefill window union of 47-67% at 32-256 tokens vs our decode-trace proxy 0.39 (32), 0.46 (64), 0.55 (128): the real prefill union looks 0.08-0.12 absolute higher than the decode-window proxy, so our proxy is a lower bound (models differ; theirs unstated). Also LRU being close to the learned policy on the larger model matches our finding that eviction policy has small leverage.

## P-0021  llama : add Mixtral support (llama.cpp PR #4406, author slaren, with ggerganov; opened 2023-12-11, merged 2023-12-13; 47 commits, 195 comments)

- Read: GitHub API: pull 4406 body and all 195 issue comments (2 pages)
- Setup: PR description and discussion only: Q4_0 Mixtral-8x7B on an M2 Ultra (Metal, -ngl 999). One timing log: prompt eval 9.00 ms/token (111.17 tok/s), generation 18.67 ms/token (53.57 tok/s).
- Numbers (what | value | baseline | ratio | where):
  - Q4_0 Mixtral-8x7B prompt eval, M2 Ultra | 9.00 ms/token = 111.17 tok/s (20-token prompt) | - | - | PR body log
  - Q4_0 Mixtral-8x7B generation, M2 Ultra | 18.67 ms/token = 53.57 tok/s | - | prompt/generation = 2.07x per token only | PR body log
- Catalog / first-pass numbers checked:
  - [yes] The indirect matmul op (ggml_mul_mat_id): 'a new indirect matrix multiplication operator' (PR body, signature ggml_mul_mat_id(ctx, as[], n_as, ids, id, b))
  - [no (for the PR as merged); yes for later code] already processes batched tokens per expert; expert-grouped compute is the existing baseline: PR body: 'efficient for BS=1, but not so much for BS>1, since each token in the batch is evaluated separately. Improvements will follow'; slaren, 11 Dec: prompt processing 'not much better than during generation' and 'We can improve this by grouping the tokens by the experts that they use ... not yet implemented'. In our local llama.cpp checkout (eec18f5, dated 3 Oct 2026) the CPU backend ggml_compute_forward_mul_mat_id groups rows per expert (matrix_row_counts, MMID_MATRIX_ROW, tiled path), so the catalog statement is true of today's code but not of this PR.
- Transfers to our regime: **partly**. Not a paper: it documents the operator our patched build traces. Today's grouped implementation (read in the local checkout, not on the PR page) means expert-major compute already exists; whether it also reads each expert's weights once from disk in prefill depends on page cache/mmap, not on this PR. The 111 tok/s prompt rate is Metal on an M2 Ultra and says nothing about CPU-only 512-expert models.
- Quality loss: None for the op (exact). The PR thread reports early-EOS/typo issues on Metal that were bugs being chased.
- Ablation: None.
- Limitations (authors): Author-stated: BS>1 not efficient; quantisation mixtures 'very likely to change'.
- Contradicts / bears on our findings: Our plan to remove the ids->ne[1]!=1 filter assumes batch >1 reaches the grouped path; consistent with the current code. The PR itself shows the pre-grouping design, so any result citing 'PR #4406 baseline' for expert-grouped compute is wrong for that commit.

## P-0022  Flash-MoE: Running a 397B-parameter model on a 48GB MacBook (lilting channel article by Ikesan, 23 Mar 2026, translated from Japanese) and its primary sources: github.com/danveloper/flash-moe (CLAUDE.md/README and paper/flash_moe.pdf 'Flash-MoE: Streaming a 397B Parameter Mixture-of-Experts Model from NVMe at 5.7 Tokens/Second on Consumer Hardware', authors Claude Opus 4.6 and Daniel Woods; no arXiv id)

- Read: lilting.ch article HTML; GitHub raw CLAUDE.md; GitHub raw paper/flash_moe.pdf (pdftotext)
- Setup: Qwen3.5-397B-A17B: 60 layers (45 GatedDeltaNet + 15 full attention), 512 routed experts/layer plus one shared expert, default top-10 (paper abstract), project runs K=4 (the 'pruned configuration'). Apple M3 Max 48 GB unified memory (~400 GB/s), 1 TB Apple SSD measured 17.5 GB/s sequential read, macOS 26.2, batch 1. 4-bit experts 6.75 MB each (209 GB total) or 2-bit 3.93 MB (120 GB). Parallel pread() with GCD, OS page cache as the only cache (~71% hit reported, ~35 GB). Non-expert 5.5 GB resident. Custom Metal kernels.
- Numbers (what | value | baseline | ratio | where):
  - Decode, 4-bit experts, K=4, FMA kernel | 4.36 tok/s | same engine before FMA: 3.90 tok/s | 1.12x | CLAUDE.md results table; article table
  - Decode, 2-bit experts, K=4, trust OS | 5.74 tok/s (peak 7.05) | 4-bit 4.36 | 1.32x; breaks JSON/tool calling (\name\ instead of "name") | CLAUDE.md; paper Table 1
  - Per-layer time at 4-bit | 4.28 ms = 1.22 + 0.55 GPU + 0.01 CPU + 2.41 ms SSD I/O (+deferred expert compute) | - | 60 x 4.28 ms = 257 ms = 3.9 tok/s, consistent with baseline 3.90 | CLAUDE.md pipeline; article
  - I/O per token at K=4 | 4 x 6.75 MB = 27 MB per layer at 4-bit (1.62 GB per token); 943 MB per token at 2-bit | - | 27 MB in 2.41 ms is 11 GB/s apparent; with 71% page-cache hits the physical read is ~7.8 MB/layer = ~3.2 GB/s, below the 17.5 GB/s peak | paper Sec 5.1; article
  - Top-K sweep (2-bit, F_NOCACHE pipeline) | K=10 1.20 tok/s (2,359 MB/tok); K=6 1.91 tok/s (1,415 MB); K=4 3.15 tok/s (943 MB); K=3 and K=2 collapsed (EOS only) | K=10 | K=6: 1.59x tok/s, 1.67x fewer bytes; K=4: 2.6x tok/s, 2.5x fewer bytes | paper Table 3, Sec 4.5
  - Quality statement for K sweep | K=6 'Same as K=10'; K=4 'Excellent (no degradation)' | - | judged by 200-token generations over 5 prompts, no benchmark | paper Table 3 caption
  - Routing statistics at K=4 over 200 tokens | 43-57% of 512 experts used per layer over 30 tokens; consecutive-token overlap 8-34%; LRU per-layer hit 49/59/68/71% at 16/32/64/128 entries; cross-layer correlation near zero | - | - | paper Sec 4.4 (statistics box)
  - Things that failed (58-90 experiments) | LZ4 -13%; mmap about 5x slower (article x0.2); F_RDADVISE prefetch net 0%, SSD DMA slows GPU by 73%; temporal expert prediction -18% (25% hit); MLP routing predictor 31% accuracy; expert file clustering 0%; dispatch_io -70% | - | - | CLAUDE.md; paper Table 4
  - Prefill | 'batch prefill' skips expert I/O and compute for all prompt tokens except the last (discard_deferred_experts); TTFT 5.6 s -> 2.6 s for a 16-token prompt | - | 2.15x; changes the computation, no quality test | paper Sec 5.5
- Catalog / first-pass numbers checked:
  - [yes] NVMe expert streaming with parallel pread(): parallel pread with GCD dispatch groups
  - [yes] Reported I/O of 2.41 ms per layer: at 4-bit K=4, out of 4.28 ms per layer; the paper's 2-bit breakdown gives 1.37-1.49 ms
  - [no] No prefill benchmark found: the paper (not the article) has one prefill number: TTFT 5.6 s -> 2.6 s for a 16-token prompt, obtained by skipping expert work for N-1 prompt tokens, so it is not a faithful prefill and is not an expert-major measurement
- Transfers to our regime: **partly**. Nearest in shape (512 experts, SSD streaming, 48 GB, batch 1, OS page cache) and it supplies the only K=10 -> 4/6 throughput sweep in an I/O-bound regime. Breaks: Apple unified memory and a 17.5 GB/s Apple SSD (3-7x a typical laptop NVMe), GPU does all compute, 2-bit experts, GatedDeltaNet hybrid, CPU-only is not tested, and quality is self-assessed by coherence on five prompts. Its own negative results (prefetch hurts because SSD DMA contends with GPU on the shared memory bus; mmap 5x slower than pread) are Apple-specific for the first and general for the second.
- Quality loss: No benchmark. Claims K=4 'no degradation'; independent measurement on the same model (see P-0030/prior-art section, arXiv 2609.04575) finds K 10 -> 5 costs 2.10 MMLU points and 4.20 GSM8K points under default renormalisation. The 2-bit variant is admitted to break JSON/tool calling.
- Ablation: Table 3 isolates K (bytes and tok/s) at fixed 2-bit; Table 1 isolates cache choices (custom LRU slower than OS cache). No cache-size sweep with the OS cache.
- Limitations (authors): Stated: Apple-specific constraints; 2-bit unsuitable for tool use. Not stated: the K=4 quality evidence is anecdotal.
- Contradicts / bears on our findings: Supports our direction: K 10 -> 6 gives 1.59x tok/s and 1.67x fewer bytes in an I/O-bound run (our simulated K=6 vs K=8 byte gain at 8 GB is 1.43x-1.78x), and 'temporal prediction / prefetch does not pay' echoes our weak prefetch gains. It undercuts any 'K=4 is free' belief: the quality evidence is weaker than it looks. Its 71% OS-cache hit at ~35 GB cache (~17% of the 209 GB) is a useful reference point for our 16/32 GB hit-rate simulations.

## P-0024  Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models (Xudong Lu, Qi Liu, Yuhui Xu, Aojun Zhou, Siyuan Huang, Bo Zhang, Junchi Yan, Hongsheng Li; CUHK MMLab / SJTU / Salesforce / Shanghai AI Lab; arXiv 2402.14800v2 cs.CL, 30 May 2024; ACL 2024)

- Read: arXiv 2402.14800v2 (PDF, pdftotext)
- Setup: Mixtral-8x7B and Mixtral-8x7B-Instruct only (8 experts, top-2). bf16 needs 2 A100-80G; pruned versions fit on one. Calibration: 128 C4 sequences of 2048 tokens (domain data for MATH/GSM8K variants). Pruning enumerates expert subsets per layer (r=6 keeps 6 of 8, r=4 keeps 4). Dynamic skipping: if the second expert's weight < beta x first's weight, drop it, beta = median of w1/w0 on calibration set per layer. Speed from a script adapted from an AutoGPTQ-style generation test; batch size not reported.
- Numbers (what | value | baseline | ratio | where):
  - Average 8-task zero-shot accuracy, Mixtral-8x7B, r=8 / r=6 / r=4 | 67.58 / 64.22 / 59.57 | unpruned 67.58 | -3.36 / -8.01 points (text says -2.9 / -7.1 for another comparison) | Table 5 (and Table 2)
  - Speedup, pruning only | r=6 1.19x (text 1.20x), r=4 1.27x | r=8 | mostly from needing 1 GPU instead of 2 (less inter-GPU communication) because each token still uses 2 experts | Table 5; Sec 4.1 'Memory Usage and Generation Speed'
  - Dynamic skipping alone (r=8) | 66.37 accuracy, 1.08x | 67.58, 1.00x | -1.21 points, 1.08x | Table 5
  - Pruning + skipping | r=6: 62.91, 1.23x; r=4: 57.91, 1.31x (Instruct: 66.04 1.27x; 62.33 1.33x) | pruning only | +0.04x / +0.04x over pruning only | Table 5
  - Memory | r=6 68,383 MB (76%), r=4 46,879 MB (52%) | 89,926 MB | - | Table 9
  - Comparison with Wanda 2:4 | accuracy 62.80, 0.92x speed, 51,210 MB | ours r=4: 59.57, 1.27x, 46,879 MB | - | Table 1
- Catalog / first-pass numbers checked:
  - [yes] Skips lower-weighted experts per token on Mixtral: skips the second expert when its weight is below beta x the first
  - [partly] Pruning plus skipping gives speedup with limited loss: 1.23x/1.31x for -4.7/-9.7 points average (62.91, 57.91 vs 67.58); 'limited' only for r=6 with skipping; the first pass quoted 1.20x/1.27x/1.23x/1.31x correctly
  - [yes (and only for top-2)] Works for top-2: the algorithm and 'Limitations' state design for 4-8 experts, top-2; pruning enumeration is infeasible for 32+ experts
- Transfers to our regime: **no**. Only evidence is top-2 of 8: skipping the 2nd expert for roughly half the tokens saves 1.08x. The speedups quoted for pruning are an artefact of fewer GPUs, not of less compute. Nothing about 10 -> 6 on 512 experts, nothing about CPU or SSD. The pruning enumeration scales badly (authors' own limitation).
- Quality loss: Table 5 above; Table 2: random/frequency pruning worse; GSM8K needs domain calibration data (Table 3-4: C4 calibration gives large GSM8K loss, MATH calibration reduces it).
- Ablation: Table 5 separates pruning, skipping and both (skipping alone 1.08x). Table 6 layer-wise vs progressive pruning. Isolating a fixed top-k reduction (e.g. 2 -> 1) is not reported.
- Limitations (authors): Authors: expert-combination enumeration infeasible beyond 4-8 experts; only Mixtral tested.
- Contradicts / bears on our findings: None. The 1.08x for skipping alone is a useful upper-level reality check against our 1.19x Amdahl ceiling for K cuts (their cut is 2 -> ~1.5 on average, i.e. -25% expert compute, and returns 1.08x on a multi-GPU box).

## P-0025  Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining (Costin-Andrei Oncescu, Qingyang Wu, Wai Tong Chung, Robert Wu, Bryan Gopal, Junxiong Wang, Tri Dao, Ben Athiwaratkun; Together AI; arXiv 2511.02237v1 cs.LG, 4 Nov 2025)

- Read: arXiv 2511.02237v1 (PDF, pdftotext)
- Setup: Qwen3-30B-A3B (48 layers, 128 experts, top-8, bf16) on one H100 80GB; Qwen3-235B-A22B (96 layers, 128 experts, top-8) with TP over 8 H100 NVLink; SGLang; batch up to 16 (CUDA graphs captured to 16); temperature 0.6, top-p 0.95, up to 32768 generated tokens; AIME24, GPQA, LiveCodeBench, MATH-500, 4 runs (3 for 235B). Baseline: vanilla top-8; also 'pruned' = plain top-k0 per token (Phase 1 only). OEA = top-k0 base experts per token, then each token piggybacks on experts already loaded for other tokens in the batch.
- Numbers (what | value | baseline | ratio | where):
  - MoE layer latency, Qwen3-30B, k0=3 (OEA) | 106.8 us | vanilla 175.7 us | 1.645x faster (-39%) | Table 3
  - MoE layer latency, Qwen3-30B, k0=5 | 136.0 us | 175.7 us | 1.29x (-23%) | Table 3
  - MoE layer latency incl. all-reduce, Qwen3-235B, k0=5 / k0=3 | 101.4 / 87.7 us | 119.4 us | 1.18x / 1.36x (text: 15% at k0=5) | Table 5
  - Average activated experts per layer, Qwen3-30B, AIME24 and GPQA at k0=3 | 22.2 and 26.5 | vanilla 43.0 and 51.6 | 0.52x and 0.51x | Table 4
  - Qwen3-30B accuracy, OEA k0=3 | AIME 80.0, GPQA 58.6, LCB 61.2, MATH 93.5 | vanilla 80.4, 60.2, 62.1, 92.8 | within standard error | Table 1/6
  - Qwen3-30B accuracy, PLAIN top-k0 (pruned) k0=3 / 4 / 5 / 6 | AIME 51.2 / 75.8 / 80.6 / 80.2; GPQA 45.7 / 54.3 / 56.2 / 58.3; LCB 37.4 / 58.2 / 63.2 / 63.1; MATH 91.1 / 92.7 / 92.6 / 93.1 | vanilla 80.4, 60.2, 62.1, 92.8 | k0=3: AIME -29.2, GPQA -14.5, LCB -24.7 points; k0=5: GPQA -4.0; k0=6: GPQA -1.9 | Table 1 / Table 7
  - Qwen3-235B accuracy, PLAIN top-k0 k0=3 / 4 / 5 / 6 | AIME 17.5 / 69.4 / 81.9 / 82.8; GPQA 43.8 / 56.4 / 60.6 / 64.1; LCB 5.7 / 27.4 / 53.5 / 60.8; MATH 80.9 / 93.3 / 94.5 / 94.5 | vanilla 85.0, 68.4, 68.5, 94.7 | k0=6: AIME -2.2, GPQA -4.3, LCB -7.7; k0=5: GPQA -7.8, LCB -15.0 | Table 2 / Table 9
- Catalog / first-pass numbers checked:
  - [partly] Qwen3-30B-A3B quality drop when base k is lowered: The 'pruned' (plain top-k0) columns do show large drops at low k0 (k0=3: AIME 80.4 -> 51.2, GPQA 60.2 -> 45.7). The catalog's specific numbers are not those.
  - [no (as accuracy); yes as experts] AIME24 43.0 -> 22.2: 43.0 and 22.2 are the average number of activated experts per layer (Table 4: vanilla vs k0=3 OEA); AIME accuracy is 80.4 vs 80.0 (OEA) or 51.2 (plain top-3)
  - [no (as accuracy); yes as experts] GPQA 51.6 -> 26.5: 51.6 and 26.5 are average activated experts (Table 4); GPQA accuracy is 60.2 vs 58.6 (OEA) or 45.7 (plain top-3)
- Transfers to our regime: **no**. OEA's speedup and its quality recovery both come from batch>1: extra experts are free because other tokens already loaded them. At batch 1 there is nothing to piggyback on (authors: batch size 1 makes piggybacking redundant) so only the plain top-k0 rows apply, and those are a quality measurement for a pure top-k cut. GPU HBM-bound latency model (latency linear in activated experts, R2>0.99) does not describe CPU-DRAM or NVMe.
- Quality loss: For plain top-k0 (the cut we would use): Qwen3-30B (k=8) no significant change down to k0=5 except GPQA -4.0; k0=4 AIME -4.6, GPQA -5.9, LCB -3.9; k0=3 collapse. Qwen3-235B (k=8): k0=6 already LCB -7.7, GPQA -4.3; k0=5 LCB -15, GPQA -7.8; k0=4 LCB -41. Long-generation thinking benchmarks, so errors compound.
- Ablation: Yes: pruned (Phase 1 only) vs OEA at every k0 (Tables 1, 2, 6-9); cross-entropy sweeps of k0, k_max, p, maxP (Fig 2-3); batch size effect in Fig 5.
- Limitations (authors): Authors: OEA depends on token diversity in the batch; effectiveness limited to moderate batch sizes; padding artefact in SGLang CUDA graphs; latency measured per MoE layer, not end to end.
- Contradicts / bears on our findings: Yes, twice. (1) The 'correction' appended to pubtech/A.md and the first-pass note ('at k0=3 accuracy barely moves; no evidence for or against a plain top-k cut') is itself incomplete: accuracy barely moves only for OEA at batch 16; the same paper's plain top-k0 rows (Tables 1, 2, 7, 9) are direct evidence AGAINST a plain top-k cut, so the original A.md sentence ('large drops ... when k is lowered to 3') is substantively right for Qwen3-30B even though the catalog's numbers are experts-per-layer. (2) Our 6-of-10 plain cut sits at 60% of k, between k0=5 (62.5%, GPQA -4.0 / LCB -15 on 235B) and k0=6 (75%) of this paper's k=8 models: expect a measurable loss on reasoning benchmarks, not 'free'.

## P-0026  LExI: Layer-Adaptive Active Experts for Efficient MoE Model Inference (Krishna Teja Chitty-Venkata, Sandeep Madireddy, Murali Emani, Venkatram Vishwanath; Argonne National Laboratory; arXiv 2509.02753v1 cs.LG, 2 Sep 2025; 'Preprint. Under review')

- Read: arXiv 2509.02753v1 (PDF, pdftotext)
- Setup: Six MoEs: Mixtral-8x7B, Qwen1.5-MoE-A2.7B, OLMoE-1B-7B, MiniCPM-MoE-8x2B, DeepSeek-V2-Lite, DeepSeek-VL2-Tiny. NVIDIA H100 80GB, 4 GPUs for LLMs (2 for DeepSeek-V2-Lite/VL2-Tiny), tensor parallel, vLLM with FusedMoE, batch 16, throughput = (input+output tokens)/latency. Baselines: inter-expert pruning (NAEE) and intra-expert pruning (MoE-I2) at 12.5/25/50%. Method: Stage 1 per-layer sensitivity = Frobenius norm of output change for each candidate top-k using random Gaussian inputs (data-free, Monte Carlo); Stage 2 search assigns per-layer k under a total 'active expert budget' B. Search space: every integer from 1 to the pretrained top-k.
- Numbers (what | value | baseline | ratio | where):
  - OLMoE-1B-7B, budget B=100 | same throughput as 50% intra-pruning with +10% accuracy; +15% accuracy vs 50% inter-pruning | pruning baselines | not absolute (percent may be points or relative; figures only) | Sec 5, Fig 4a
  - Qwen1.5-MoE | at least +5.1% throughput vs both pruning types, +0.5% accuracy | pruning | - | Sec 5, Fig 4
  - DeepSeek-V2-Lite | +6.5% throughput at equal accuracy vs inter-pruning; +6 accuracy points vs intra | pruning | - | Sec 5, Fig 4
  - Mixtral-8x7B perplexity at ~2.4k throughput | ~23 on C4, ~10 on WikiText | pruned models worse | figure read | Sec 5, Fig 7
  - Abstract: Qwen1.5-MoE same throughput with 10% better accuracy than pruning | 10% | pruning | body attributes the 10% to OLMoE (Fig 4a); abstract names Qwen1.5-MoE | Abstract vs Sec 5
- Catalog / first-pass numbers checked:
  - [yes] Per-layer top-k: LExI assigns k per layer from 1 to base k under a total budget
  - [partly] Small accuracy loss claimed: 'negligible accuracy loss' relative to the unpruned model is claimed in words; every plotted comparison is against pruning, and there are no tables of loss versus the unmodified model
  - [partly] Fixed top-k is wasteful: the paper's claim is that uniform k is redundant and that pruning does not speed up vLLM; it shows throughput gains of reduced-k, not that fixed k is wasteful in quality terms
- Transfers to our regime: **no**. GPU, vLLM, batch 16, tensor parallel; throughput measured as tokens/s. The selection of k from random Gaussian inputs is data-free and cheap, so the allocation step is reusable: per-layer sensitivity profiles differ by model (Fig 3 shows Mixtral and Qwen1.5-MoE with opposite early/late patterns). No number applies to a CPU-only SSD-bound model, no 512-expert model tested; OLMoE (top-8 of 64) is the only model close to the one we test locally.
- Quality loss: Figures only (LM-Eval 9 tasks, Qasper F1, passkey, perplexity C4/PTB/WikiText, VLM suites); no tabulated numbers vs unmodified model in text.
- Ablation: None for the sensitivity-estimation method (random input vs real data); only budgets B swept.
- Limitations (authors): Authors: does not reduce memory footprint; inapplicable to top-1 models (Llama-4).
- Contradicts / bears on our findings: None directly. It states expert pruning gives no consistent vLLM speedup (load imbalance), which agrees with batch-size sensitivity found in OEA; our per-layer K idea (spend more k in early/sensitive layers) is exactly this paper's idea, so the idea is not novel, only the CPU/SSD regime.

## P-0027  REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression (Mike Lasby, Ivan Lazarevich, Nish Sinnadurai, Sean Lie, Yani Ioannou, Vithursan Thangarasa; Cerebras Systems / Univ. of Calgary; ICLR 2026; arXiv 2510.13999, v1 15 Oct 2025, read also v3 13 May 2026)

- Read: arXiv 2510.13999v1 and v3 (PDF, pdftotext)
- Setup: Seven SMoE models 21B to ~1T: ERNIE-4.5-21B (64 experts top-6), Qwen3-30B-A3B (128, top-8), Mixtral-8x7B (8, top-2), GLM-4.5-Air (128, top-8), Llama-4-Scout (16, top-1), Qwen3-Coder-480B-A35B-FP8 (160, top-8), Kimi-K2-Instruct-W4A16 (384, top-8, 1.03T). Prune or merge 25% or 50% of experts per layer, one shot, no fine-tuning. Calibration: evol-codealpaca for code (domain), C4, plus xlam-function-calling and SWE-smith for big models; 12,228 samples up to 16k tokens for >=110B models. Baselines: frequency pruning, EAN, M-SMoE, HC-SMoE. No latency or throughput measured.
- Numbers (what | value | baseline | ratio | where):
  - Qwen3-Coder-480B, 50% pruned, code average (Eval+/LiveCode) | v1: 0.644 (0.873/0.415); v3: 0.619 (0.822/0.415) | baseline v1 0.660; v3 0.636 | v1 0.976x; v3 0.973x | Table 3 (v1 and v3)
  - Kimi-K2, 50% pruned, code average | v1: 0.646; v3: 0.624 | baseline v1 0.659; v3 0.631 | v1 0.980x; v3 0.989x | Table 3
  - Qwen3-Coder-480B 50%: MC average / tool-use overall / SWE-Bench-Verified | 0.692 / 0.674 / 0.522 | 0.750 / 0.690 / 0.540 | 0.923x / 0.977x / 0.967x | Table 3
  - Kimi-K2 50%: MC / tool-use overall / multi-turn / SWE-Bench | 0.643 / 0.564 / 0.164 / 0.576 | 0.780 / 0.666 / 0.355 / 0.554 | 0.824x / 0.847x / 0.46x / 1.04x | Table 3
  - Qwen3-30B-A3B 50%, REAP, code avg / MC avg | v3: 0.541 / 0.503 | baseline 0.558 / 0.721 | 0.97x / 0.70x | Table 2
  - GLM-4.5-Air 50% REAP: code avg / math avg / MC | 0.515 / 0.857 / 0.554 | 0.580 / 0.882 / 0.747 | 0.89x / 0.97x / 0.74x | Table 2
  - Calibration sensitivity | C4 calibration collapses code accuracy (Fig A8) | evol-codealpaca calibration | 'stark' | Sec F 'importance of domain-specific calibration'
- Catalog / first-pass numbers checked:
  - [yes] One-shot 50% expert pruning near-lossless on code: code average within 1-3% (v1: -2.4% Qwen3-Coder, -2.0% Kimi; v3 -2.7%, -1.1%; the paper says 0.20%/1.4% mean decrease in v1, 0.16%/1.2% in v3, which uses a different metric/average)
  - [yes] for Qwen3-Coder-480B and Kimi-K2: abstract also says tool calling; but multiple-choice and tool-use drop clearly, Kimi multi-turn tool use 0.355 -> 0.164
- Transfers to our regime: **partly**. It shrinks the number of experts (file size) by 50% rather than the number activated, which is a different lever: with a 512-expert model it would halve the stored set at the price of calibrating on the target domain; it makes no speed claim and saves no per-token reads. For an NVMe-streaming target, halving the pool could lift cache hit rates (twice the share of experts fit in RAM) but would need the domain to be known up front. The hard requirement is domain-matched calibration (C4 calibration collapses code accuracy).
- Quality loss: Table 3 above; Table 2: at 50% on Qwen3-30B MC avg 0.721 -> 0.503. Near-lossless is true for code on the two largest models only.
- Ablation: Pruning criteria (frequency, EAN, REAP) and merging methods compared at equal compression; calibration set ablation (C4 vs evol-codealpaca, Fig A8). No ablation about runtime.
- Limitations (authors): Authors flag domain-specific calibration need, one-shot setting, no runtime study; ethics note about deploying cheaper models.
- Contradicts / bears on our findings: None. Note the first pass quoted v1 numbers (0.660 -> 0.644 etc.), which are correct for v1; v3 revised the Eval+ scores down (0.841/0.828 baselines), so the same table in the camera-ready reads 0.636 -> 0.619.

## P-0028  SliceMoE: Bit-Sliced Expert Caching under Miss-Rate Constraints for Efficient MoE Inference (Yuseon Choi, Sangjin Kim, Jungjun Oh, Gwangtae Park, Byeongcheol Kim, Hoi-Jun Yoo; KAIST; arXiv 2512.12990, cs.AR; read v5 dated 8 Sep 2026, first pass read v3)

- Read: arXiv 2512.12990v5 (PDF, pdftotext)
- Setup: DeepSeek-V2-Lite and Qwen1.5-MoE-A2.7B on a MODELED edge system: XPU 16.4 TOPS (8,192 8-bit PEs at 1 GHz), LPDDR4 ~104 Gbps ~8 GB, UFS 3.1 flash 10 Gbps (103 pJ/bit), expert-cache capacities 1.8/2.4/3.6 GB, batch 1, GSM8K 5-shot (prefill ~500, decode >100 tokens), plain LRU cache shared across layers, miss-rate constraint active after 10 decode steps. Experts stored as nested Matryoshka bit slices (AMAT: truncation-based asymmetric quantisation, e.g. 8/4, 6/3, 4/2 bit); critical experts (0-2 per token) request MSB+LSB slices, others MSB only. Baselines: Cache-Prior routing (high-bit) and Cumsum routing. Predictive Cache Warmup re-orders the cache at the end of prefill by accumulated frequency.
- Numbers (what | value | baseline | ratio | where):
  - Decode energy reduction, DeepSeek-V2-Lite / Qwen1.5-MoE | up to 2.37x / 2.85x | high-bit Cache-Prior at matched accuracy | as stated | Sec 6.3, Fig 9
  - Decode speed-up | up to 1.81x / 1.64x | same | as stated | Sec 6.3, Fig 9
  - Predictive cache warm-up vs empty cache after prefill | 2.31x lower decode energy, 1.96x speed-up | empty cache | as stated | Sec 6.3, Fig 10
  - Accuracy | GSM8K 5-shot vs high-bit-normalised miss rate; high-bit caching collapses under tight capacity; uniform low-bit has a ceiling; DBSC+AMAT on the Pareto front | - | figure only | Sec 6.2, Fig 8
  - Table 1 AMAT PPL | 7.08 for 8/4-bit at one setting; plain symmetric truncation 19.67 and 3.6e10 (broken) | - | - | Table 1
- Catalog / first-pass numbers checked:
  - [yes] More low-bit experts cached beats fewer full-precision ones: central claim; but it combines low-bit slices with cache-aware routing (Cache-Prior) and only high-bit for 'critical' single-head experts
  - [yes (as motivation)] Miss rate set by how many experts fit: miss-rate-constrained design; 1.8 GB holds at least one high-bit expert per layer, 3.6 GB fewer than half
- Transfers to our regime: **partly**. Right regime on the storage side (flash to small DRAM expert cache, batch 1, 1.8-3.6 GB cache) and it uses prefill to warm decode cache. But results are simulated energy/latency on a custom accelerator model, two ~16B 64-expert models, GSM8K only, LRU only, baselines are their own re-implementations, and the benefit mixes three levers (slice caching, quantisation, cache-aware routing which swaps experts). Not CPU/llama.cpp.
- Quality loss: Qualitative curves (Fig 8) at matched miss rates; AMAT PPL Table 1; 'near-high-bit accuracy' claimed. Cache-aware routing underlying Cache-Prior is itself lossy.
- Ablation: Fig 9 compares cache-aware routing schemes at fixed cache sizes; Fig 10 isolates warm-up. No clean 'low-bit slices vs same-bit more-experts' ablation separated from routing.
- Limitations (authors): Not given as a section; evaluation is on a modeled system and one benchmark.
- Contradicts / bears on our findings: None. Consistent with our 2/3/4-bit family direction (more cached entries improve hit rate); first pass numbers are right and unchanged between v3 and v5 (checked 2.37x/2.85x/1.81x/1.64x/2.31x/1.96x in v5).

## P-0029  DyMoE: Dynamic Expert Orchestration with Mixed-Precision Quantization for Efficient MoE Inference on Edge (Yuegui Huang, Zhiyuan Fang, Weiqi Luo, Ruoyu Wu, Wuhui Chen, Zibin Zheng; Sun Yat-sen Univ; arXiv 2603.19172v1 cs.LG, 19 Mar 2026)

- Read: arXiv 2603.19172v1 (PDF, pdftotext)
- Setup: Mixtral-8x7B-Instruct and Qwen3-30B-A3B. AMD EPYC 7542 + RTX 3090 24 GB, PCIe Gen3 x16; VRAM capped by a software allocator to 12-24 GB; experts in host memory (the figure mentions 'CPU Memory / SSD', no SSD experiment). Batch size 1, ShareGPT inputs. Baselines: Accelerate, Mixtral-Offloading, MoE-Infinity, Fiddler. GPTQ for the quantisation. Retention ratio r = share of 'critical' experts kept at High precision (4-bit; 8-bit variant exists); the rest run at Low (2-bit, '4/2') or are skipped ('4/0'). Critical set chosen by heavy-hitter token load in prefill and gate score in decode; depth-adaptive r(l) (shallow layers keep more); look-ahead prefetch via adjacent-layer gate similarity.
- Numbers (what | value | baseline | ratio | where):
  - TTFT, Mixtral, DyMoE(4/0), 24 GB / 16 GB | 22.7x / 15.7x lower | Fiddler | as stated | Sec 6.2, Fig 10
  - TTFT vs Accelerate / Mixtral-Offloading | 7.2-8.3x / 4.2-14.58x | same | as stated | Sec 6.2
  - TTFT, Qwen3-30B-A3B, 12 GB | 3.44x | Accelerate | as stated | Sec 6.2
  - TPOT, Mixtral 16 GB | 14.58x vs Fiddler; 8.31x vs Accelerate; up to 2.27x vs Mixtral-Offloading | same | as stated (14.58x appears for TTFT vs Mixtral-Offloading and for TPOT vs Fiddler) | Sec 6.3
  - TPOT, Qwen3-30B-A3B 12 GB | 2.86x | Accelerate (stated 'same 12 GB environment') | as stated | Sec 6.3
  - Ablation, Mixtral TPOT 16 GB / 24 GB: Load-on-demand | 0.2795 s / 0.2795 s | - | - | Table 3 row 1
  - Ablation: Cache | 0.1489 / 0.1273 s | LoD | 1.88x / 2.20x | Table 3 row 2
  - Ablation: Cache+Dyquant(4/0)+Prefetcher (full) | 0.1048 / 0.0656 s | LoD; and vs Cache row | 2.67x / 4.26x vs LoD; 1.42x / 1.94x vs cache-only | Table 3 row 6
  - Dyquant alone (no prefetch) vs cache-only | 0.1307 vs 0.1489 (16 GB); 0.0796 vs 0.1273 (24 GB) | cache-only | 1.14x / 1.60x | Table 3 rows 2,4
  - Accuracy, Mixtral 4/0, r=0.75 / 0.9 / 1.0 | MMLU 0.6714 / 0.6807 / 0.6795; CMMLU 0.4822 / 0.4940 / 0.5044; GSM8K 0.6209 / 0.6558 / 0.6467 | Int4 uniform (r=1.0) | r=0.75: -0.8, -2.2, -2.6 points | Table 2
  - Accuracy, Qwen3-30B-A3B 4/0, r=0.75 / 0.9 / 1.0 | MMLU 0.7698 / 0.7737 / 0.7705; CMMLU 0.7960 / 0.8019 / 0.8044; GSM8K 0.9151 / 0.9174 / 0.8908 | Int4 | GSM8K +2.4 / +2.7 points above Int4 (authors call it regularisation) | Table 2
- Catalog / first-pass numbers checked:
  - [yes] Dynamic precision plus skipping of experts: configs '4/2' (non-critical at 2-bit) and '4/0' (non-critical skipped) at retention r (0.75-1.0)
  - [partly] 8/4-bit/skip per expert: the High/Low configurations in the paper are 4/0, 4/2 and 8/0; no per-expert choice among 8, 4 and skip
  - [yes (but GPU+host RAM)] mixed-precision cache on edge: LRU extended with no-duplication, precision-promotion, conservative-reuse rules; evaluation is VRAM-limited 12-24 GB with experts in host memory
- Transfers to our regime: **no**. Needs a GPU with a VRAM cache and PCIe-fed host memory; baselines are weak (Fiddler, Accelerate) and the strongest, Mixtral-Offloading, is beaten by 2.27x on decode; ablation shows plain caching accounts for 1.88-2.20x of the gain and Dyquant+prefetch adds 1.4-1.9x on top. 'Skip' means skipping 25% of the expert slots at r=0.75, which on Mixtral costs 2-3 points on GSM8K/CMMLU. Title says SSD edge; no SSD number.
- Quality loss: Table 2 above (Mixtral r=0.75 4/0: GSM8K -2.6, CMMLU -2.2, MMLU -0.8 vs Int4; Qwen3-30B no loss). Fig 11 gives accuracy-vs-r curves.
- Ablation: Table 3 (Mixtral only): load-on-demand, +cache, +prefetch, +Dyquant (4/2), +prefetcher, +Dyquant (4/0): the cleanest cache vs layout (precision) vs prefetch isolation among my rows.
- Limitations (authors): Not a separate section; GPTQ dependency contradicts 'no calibration' wording (GPTQ uses calibration data); only two models.
- Contradicts / bears on our findings: Its ablation (cache alone 1.88-2.20x over load-on-demand, prefetch a further 1.13-1.15x) matches our finding that residency/caching dominates and prefetch is a small extra.

## P-0030  MoE top_k override (--moe-top-k) guide, vllm-mlx docs (author and date not stated on the page; vllm-mlx.is-a.dev/guides/moe-top-k.md)

- Read: raw markdown fetched with curl (4,672 bytes)
- Setup: Qwen3-30B-A3B-4bit (128 experts, top-8) on Apple M4 Max 128 GB, MLX backend, batch 1, greedy decode; whole model resident in RAM; flag sets top_k on every .mlp.switch_mlp layer. Quality section is a table of TBD placeholders (MMLU 200 samples, GSM8K 100 samples planned).
- Numbers (what | value | baseline | ratio | where):
  - Decode tok/s, top_k 8 / 6 / 5 / 4 | 126.5 / 136.1 / 140.3 / 147.3 | k=8 126.5 | 1.076x / 1.109x / 1.164x (stated +7.6 / +10.9 / +16.5%) | 'Decode throughput' table
  - Implied expert-compute share of decode time | about 28% (k=6: 1/(1-f/4)=1.076 gives f=0.28; k=4: 1/(1-f/2)=1.164 gives f=0.28) | - | consistent across k=6 and k=4 | derived by me
  - 3-bit experts + top_k | 3-bit k=8: 138.6 (+9.6%); 3-bit k=6: 147.1 (+16.3%); 3-bit k=4: 157.3 (+24%) with broken output | 4-bit k=8 126.5 | - | 'Stacking' list
  - Output parity | first 16 greedy tokens identical to baseline at top_k=4 on probe prompts | - | - | 'Greedy output parity'
- Catalog / first-pass numbers checked:
  - [yes] Qwen3-30B-A3B 4-bit decode 126.5 -> 147.3 tok/s going from 8 to 4 experts: 126.5 -> 147.3
  - [yes] 1.16x: 147.3/126.5 = 1.164
  - [yes] compute-bound: everything is in unified RAM; the page itself says it is bound by expert gather bandwidth on Apple Silicon
  - [yes] quality table empty: all cells TBD; page says 'verify on your model'
- Transfers to our regime: **partly**. Gives the only direct batch-1 decode timing of a plain top-k cut on a 128-expert top-8 model, but with every byte in RAM: that measures only the Amdahl part (expert compute is ~28% of step time), i.e. the 1.19x ceiling in our A.md, not an I/O effect. Quality unmeasured and the guide cites LExI/Lynx for 'preserves most of the quality', which does not hold for a plain top-k cut per OEA's Qwen3-30B tables (k0=4: AIME -4.6, GPQA -5.9, LCB -3.9 points).
- Quality loss: No data (TBD). Author warns thinking-mode generations may fall off a cliff and that 3-bit + top_k=4 breaks output.
- Ablation: None (single knob sweep).
- Limitations (authors): Stated: opt-in, verify before production, floor of top_k=4 recommended.
- Contradicts / bears on our findings: None; it independently supports the compute-ceiling number. Its statement that LExI/Lynx show preserved quality is not supported for a plain cut (Lynx and OEA need batch>1).

---

# Per-family sections

## Family s1 #2: prefill as expert-major work (rows P-0017 .. P-0022)

What the raw texts establish
- Prefill touches most experts: Fate 55.9 of 60 per layer on MMLU prompts (Qwen1.5-MoE, top-4); FlashMoE 47% / 58% / 64% / 67% of experts at 32 / 64 / 128 / 256 tokens (model not named in that sentence). Our decode-trace proxy gave 0.39 / 0.46 / 0.55 at 32 / 64 / 128 tokens, so the real prefill union is probably 0.08-0.12 above the proxy (different models; the 512-expert value is still unmeasured).
- Loading each needed expert once per prefill iteration is already the published baseline design: FlashMoE Sec 2 ("loads each required expert file on-demand from the SSD exactly once per inference iteration"), and SSD-LLaMA (arXiv 2609.18110, read below) "groups all prompt rows assigned to the same expert ... each expert's weights loaded once and reused for all rows". llama.cpp's CPU mul_mat_id (current code in our checkout) groups rows per expert, while PR #4406 as merged did not.
- Fiddler does NOT avoid weight movement in long prefill: it copies weights and computes on the GPU for large per-expert inputs (the catalog line is the opposite). DAOP uses prefill token counts only to decide which experts stay on the GPU for decode.
- Flash-MoE's only prefill number (TTFT 5.6 s -> 2.6 s, 16-token prompt) comes from skipping expert compute and I/O for N-1 prompt tokens, an unvalidated change to the computation.

Further prior art found and read (same depth, text of arXiv PDFs/GitHub)
- SSD-LLaMA: SSD-Native Inference for Trillion-Parameter MoE at 1+ Token/s on a Consumer PC (Liang et al., HKUST/WiCi; arXiv 2609.18110v1, 16 Sep 2026). RTX 5090 + <=32 GB RAM (also a 16 GB target), PCIe5 SSD (9 GiB/s physical limit in Fig 11), batch 1, Kimi/GLM-class models. Prefill 1.90-3.61x over llama.cpp and 1.06-2.92x over KTransformers (GLM-5.2: 1.44-1.90x over Colibri); abstract 1.52-4.19x prefill, 2.10-15.58x decode; Kimi-K3 (2.8T) prefill 1.217 tok/s (24.1x over llama.cpp), decode 0.465 tok/s (17.2x). Explicit expert-grouped prefill and a three-tier SSD/RAM/VRAM hierarchy, exact (no pruning or substitution). Threatens novelty of the "read each expert once in prefill" part; the GPU is required.
- Also seen as search hits, NOT read: llama.cpp discussion #27149 (expert-aware SSD streaming proposal), HOBBIT (arXiv 2411.01433), PIPO (2504.03664), Colibri, KTransformers, MoE-Infinity. From memory: MoE-Infinity, Pre-gated MoE, ProMoE, EdgeMoE, HOBBIT, MoE-Lightning all load per-expert in batch-aware order; none tested with CPU-only 8-32 GB and a 512-expert model.

What is new in OUR variant: nothing in the idea itself. The existing literature covers expert-grouped prefill (FlashMoE, SSD-LLaMA, llama.cpp mul_mat_id). What remains open and unpublished among what I read: the measured prefill union at 512 experts/top-10 and the CPU-only, mmap-versus-explicit-read wall time for it. Closest papers, read in full: FlashMoE (P-0020) for SSD plus prefill, SSD-LLaMA for expert-grouped prefill with SSD.

## Family s1 #3: route fewer experts per token (rows P-0024 .. P-0030)

What the raw texts establish
- The only rigorous evidence for a plain top-k cut on fine-grained models is inside P-0025 (OEA) "pruned" columns and in arXiv 2609.04575 (below): Qwen3-30B k0=5 -4.0 GPQA, k0=4 -4.6 AIME / -5.9 GPQA; Qwen3-235B k0=6 -4.3 GPQA / -7.7 LiveCodeBench, k0=5 -15 LCB; and for the 512-expert top-10 Qwen3.5-397B, 10 -> 5 costs 2.10 MMLU / 4.20 GSM8K under default renormalisation (0.55 / 0.80 with the denominator anchored to the native top-10).
- Speed evidence: 1.08x for skipping alone on Mixtral (NAEE), +16.5% for 8 -> 4 in RAM on M4 Max (vllm-mlx; implies expert compute is ~28% of step time), 1.59x (10 -> 6) and 2.6x (10 -> 4) in an SSD-bound 2-bit run on Flash-MoE's own paper.
- NAEE, LExI, OEA, DyMoE gains come from GPU batch effects or from baselines that are weak; none runs CPU-only.

Further prior art found and read
- Training-Free Halving of Activated Experts in Fine-Grained MoE Models (Chen, Yao; arXiv 2609.04575v1, 4 Sep 2026). Qwen3.6-35B-A3B (256 experts, top-8) and Qwen3.5-397B-A17B (512 experts, top-10, the nearest model shape to ours). Activate k1 experts but renormalise by the mass of top-k2 (k2 = trained k). 35B: 8 -> 4 costs 4.65 MMLU points with standard renormalisation, 0.35 with k2=16 (p=0.66); 397B: 10 -> 5 costs 2.10 MMLU vs 0.55 at k2=10 (p=0.24); GSM8K 4.20 vs 0.80; C-Eval k1=8,k2=8 -2.92; perplexity optimum differs from MMLU optimum. No latency measured. This is the closest paper to our "6 of 10": it says a plain 10 -> 6 cut is not free and gives a zero-cost fix we have not tried (k2=10). It strengthens the idea (quality can be recovered) and weakens novelty of the plain cut.
- Don't Skip the Baseline: A Controlled Comparison of Dynamic Expert Skipping in MoEs (Steimle et al.; arXiv 2610.28369v1, 20 Oct 2026; Gemma4-26B-A4B, Qwen3-30B-A3B, Qwen3-VL): static reduced k at matched compute is hard to beat by NAEE-style/learned/dynamic skipping; recommends calibration-free static k. Supports using a static K as the baseline for any dynamic scheme.
- ACE: Adaptive Calibration-Free Expert Skipping (Xu et al.; arXiv 2609.05228v1, 4 Sep 2026): token-adaptive skipping from weights/router geometry, always keeps top-1; at 50% skipping on Qwen3.6-35B-A3B -7.96% WikiText-2 perplexity and +4.15 points average accuracy vs the best competitor; up to 2.25x prefill / 1.41x decode (A100). Only the abstract and table index were read in detail.
- Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference (Skliar et al., Qualcomm; arXiv 2412.00099v2, TMLR June 2025): training-free cache-aware routing, batch 1, miss rate cut by >50%, perplexity +0.1-3%, task accuracy <0.1% loss, 2x speed on phone-class devices; models DeepSeek-V2-Lite, Qwen1.5-MoE, Phi-3.5-MoE, Mixtral. Published prior art for the cache-aware 6-of-8 variant in our simulator (read abstract, setup lines).
- Lynx (Gupta et al.; arXiv 2411.08982v3): batch-aware dynamic expert selection for GPU serving, needs moderate batch. Does Mixture-of-Experts Actually Help Inference on Consumer and Edge Hardware? (Alfarizy et al.; arXiv 2606.21428v3; OLMoE-1B-7B on M2 Pro and Jetson via llama.cpp; routing <9% of MoE block time): inference cost tracks total, not active, parameters on bandwidth-bound devices. Both read at abstract level.
- Not opened: Ada-K, AdapMoE, MC-MoE, DynMoE, MoE-Pruner, MoDES (arXiv 2511.15690), DiEP; labelled from memory or search listing only.

What is new in OUR variant: a lower fixed K (e.g. 6 of 10) is published prior art in method (OEA 'pruned', LExI, 2609.04575, Flash-MoE K=4, vllm-mlx --moe-top-k). Not published in what I read: (a) a measured K cut on a 512-expert model on CPU-only with expert weights streamed from NVMe at 8/16/32 GB, (b) the interaction between K and cache hit rate (bytes per token) in that regime (Flash-MoE gives the only I/O-bound sweep, on Apple GPU with 2-bit), (c) a quality gate for 125B-class with a denominator-anchored 6-of-10. Closest papers, read in full: arXiv 2609.04575 and the OEA paper (P-0025); closest system: Flash-MoE (P-0022).

## Corrections to the first pass (P01.md / P01.json)
1. P-0017: 'no sorting scheme' is wrong; Fate orders prefill compute and transfers by predicted expert popularity (the first pass said 'not quite right', now itemised). Abstract's 4.1x decode is not in the body (3.7x/2.3x).
2. P-0019: the mechanism is resolved: for long prefill Fiddler copies weights and computes on the GPU, the opposite of the catalog line; the catalog's mechanism is wrong. 1.30x is not derivable from the quoted 1.07x/1.65x.
3. P-0021: first-pass setting is right; additionally, slaren's own comment (11 Dec) says token grouping 'is not yet implemented'; our local checkout (eec18f5) does group rows per expert.
4. P-0022: the first pass relied on the article only. The primary repo/paper shows the model default is top-10 (confirmed), that Table 3 has a K sweep (K=6: 1.59x, K=4: 2.6x tok/s, K=3 collapse), that the prefill number comes from skipping expert work for N-1 tokens, and that quality is judged on 5 prompts. 'No prefill benchmark found' is wrong for the paper (one 16-token number).
5. P-0024: the first pass omitted that pruning speedups are mostly from needing 1 GPU instead of 2, and that skipping alone gives only 1.08x with -1.21 points.
6. P-0025 (most important): the first pass and the correction note in pubtech/A.md are right that 43.0/22.2 and 51.6/26.5 are expert counts, but wrong that 'the paper is no evidence for or against a plain top-k cut': its Tables 1, 2, 7, 9 'pruned' columns are exactly a plain top-k0 cut and show large losses at k0<=4 (30B) and k0<=6 (235B, LCB -7.7). The pubtech/A.md sentence 'large drops ... when k is lowered to 3' is therefore substantively correct for Qwen3-30B (AIME 80.4 -> 51.2).
7. P-0027: first-pass v1 numbers are correct for v1; v3 (ICLR camera-ready) revised baselines (Qwen3-Coder code avg 0.636 -> 0.619 at 50%; Kimi 0.631 -> 0.624).
8. P-0028: read v5 (8 Sep 2026) and v3 numbers agree; no correction.
9. P-0020: abstract's '+51% hit rate over LRU and LFU' is the LFU figure (LRU +21%); '2.6x' is not traced in the body.
10. P-0018/P-0029/P-0030: no number corrections. P-0029: the 14.58x appears for both TTFT (vs Mixtral-Offloading) and TPOT (vs Fiddler); ablation shows caching alone gives 1.88-2.20x of the gain.
