# deep_P04a: raw-text re-check of P-0082..P-0091 (excluding P-0086)

Date 2026-10-29. All rows read from raw text (PDF via pdftotext, HTML stripped, GitHub API JSON). Treated as untrusted data. Page numbers: printed paper pages where I could verify (MoDES p7), otherwise section/table/figure numbers. Remembered facts are labelled 'from memory'.

## P-0082: llama.cpp MoE Offload: --n-cpu-moe vs the New Expert Cache (Developers Digest, no author named, 9 Oct 2026) relaying llama.cpp PR #29887 'llama : add a GPU cache for MoE experts kept in host memory' (author am17an, opened 3 Oct 2026, merged 7 Oct 2026) and PR #30112 (multi-GPU, merged 8 Oct 2026)

- ID read: blog HTML (curl) + GitHub API JSON of PR #29887 body, its 64 comments, PR #30112 body; no arXiv; raw text read: True
- Setup: Model Qwen3.8-Flash-Next Q4_0, 93.7 GiB total, 65.4 GiB experts (blog calls it 125B params, ~6B active; arch string in a comment 'qwen4exp'). EPYC 7742, 16 threads, one NUMA node, PCIe 4.0 x16, experts in HOST RAM (not disk), llama-server -fa on -c 32768 -b 2048 -ub 2048 -t 16, temp 0, no MTP. Benchmark: SPEED-Bench 'qualitative', 11 categories x 3 samples, --osl 512, concurrency 1 (single-GPU table, PR #29887); 1 sample/category = 11 prompts (2x4090, PR #30112). Baseline = master with default --fit (no cache). Cache = per-layer-bank LRU of experts in VRAM, only misses uploaded; applies only to ubatch <= 32 (decode); hit stats counted separately for ubatch<=8 (decode) and >8.
- Numbers (value | baseline | ratio | where):
  - RTX 4090 decode, --fit --moe-cache-mib 6544 (10.4 GB layers + 6.4 GB cache): 39.4 tok/s, hit 72% | 25.0 (--fit) | 1.576x | PR #29887 table / blog 'Benchmarks, labelled'
  - RTX 4090 decode, -cmoe --moe-cache-mib 11000 (10.7 GB cache): 40.7 tok/s, hit 77% | 25.0 | 1.628x | same
  - RTX 5090 decode, --fit + 6544: 54.5 tok/s, hit 77% | 30.8 | 1.769x | same
  - RTX 5090 decode, -cmoe + 19000 (18.6 GB cache = 28.4% of 65.4 GiB experts): 67.8 tok/s, hit 89% | 30.8 | 2.201x | same
  - 2x RTX 4090 decode, -cmoe + 15000 per GPU: 64.39 tok/s, hit 95.8% | 34.78 | 1.851x | PR #30112 table
  - 2x RTX 4090 decode, --fit + 6544: 60.38 tok/s, hit 85.9% | 34.78 | 1.736x | PR #30112 table
  - 2x RTX 4090 prompt processing, -cmoe + 15000: 321.5 tok/s | 440.7 | 0.730x (prefill gets slower) | PR #30112 table
  - Radeon AI PRO R9700 32GB, Ryzen 7 8700F, 32 GB RAM, Windows, -cmoe, DIFFERENT model (GSQ-RCO Coder IQ1_M, 24 GB experts): 8000 MiB (33% of experts) / 16000 MiB (67%): 14.5 / 27.6 tok/s; decode hit 85.0% / 94.6% | 11.0 (no cache) | 1.318x / 2.509x | PR #29887 comment by LeoBorcherding 3 Oct
  - Same box re-run at later commit: 11.9 / 23.2 tok/s, hit 85.3% / 94.4% | 11.8 no cache | 1.01x / 1.97x (author: cache only breaks even with CPU path at 85% hit); all-on-GPU 43.3 | PR comment 7 Oct
  - RTX 3090 vs tuned --n-cpu-moe 44: 1-5% faster (18.80 vs 18.62 at 7300 MiB) | tuned manual split | 1.01-1.05x | PR comment Volunteer-1 7 Oct
  - RTX 5060 Ti 16GB, 62 GB RAM, UD-IQ3_XXS 78 GB (fork port): 19.26 tok/s at 8000 MiB | 14.95 | 1.288x | blog, citing review thread (not re-read at source)
  - Offline LRU trace study on real Qwen3.8-Flash-Next routing (48 layers, 512 experts, 10 prompts, per-layer LRU): hit % at cache share of 512 experts per layer: 7.8%->62.1, 14%->75.9, 25%->87.6, 31%->91.2, 62.5%->97.4 (LFU 29.8/49.6/73.9/82.7/97.3) | - | - | PR comment by lich-77 7 Oct (single commenter, unverified)
  - GLM-5.3-Flash (288 experts, top-8) 1x RTX 5090 PCIe4, 10.5% cache: 14.23 / 12.9 tok/s, hit 47.0%, PCIe RX 24.7 GB/s avg | 17.75 / 18.47 no cache | 0.80x / 0.70x (slower) | PR comment paulp83 7 Oct
- Catalog / first-pass numbers:
  - RTX 4090 decode 34.8 to 64.4 tok/s with -cmoe + 15000 MiB cache: **no**. True: 34.78->64.39 is TWO RTX 4090s (layer split), PR #30112, 15000 MiB per GPU, hit 95.8%. Single 4090 best row is 25.0->40.7 (1.63x) with 11000 MiB. First pass flagged this correctly.
  - simulated 8 GB card 11.0 tok/s no cache: **partly**. 11.0 is a 32 GB R9700 with -cmoe on a different, tiny-quant model; 'simulated 8 GB card' is the catalog's gloss, not in the source. Blog says 'to simulate a smaller card'.
  - 14.5 at 8000 MiB, 27.6 at 16000 MiB: **yes (numbers) / misleading (meaning)**. Numbers correct but 8000 MiB = 33% and 16000 MiB = 67% of that model's 24 GB of experts, not a small cache. Later re-run gave 11.9/23.2.
  - decode hit rates (truncated in catalog): **absent**. 85.0% at 8000, 94.6% at 16000 (R9700); 72/77/77/89% single-GPU table
- Transfers to our regime: **partly**. Speed: no. Experts sit in host RAM over PCIe 4.0 (24-32 GB/s), decode is GPU compute from a VRAM cache, a 4090/5090/R9700 is required, and the 2.2x needs a 18.6 GB cache plus 128+ GB host RAM. Hit-rate evidence: YES and important. This is the only source with real-routing hit rates for a 512-expert top-10 125B-class model (same family as our target): PR author 72% at 13% of host experts (--fit row: 6.4 GB of 48.9 GB host-resident), 77% at 16%, 89% at 28%; lich-77 offline per-layer LRU 75.9% at 14%, 87.6% at 25%. Caveat: workload mix (33 mixed prompts) and cold/warm state change hit by 10+ points (72% vs 85.9% at similar share in the 1-sample 2xGPU run).
- Quality loss: Method is exact in principle (LRU cache only changes where experts are computed). But comments report greedy output diverging from the non-cache path in a port (antomicblitz) and a repetition loop ('the the the') with cache on in 4 of 4 runs on a 20k-word essay prompt (insunaa), unconfirmed, vanished after changing top-k; no benchmark of quality given.
- Ablation: Partial: cache vs --fit vs -cmoe vs tuned --n-cpu-moe are compared (so cache vs static placement is isolated: 1.01-1.05x vs tuned --n-cpu-moe on 3090). No layout/prefetch ablation. Policy: lich-77 offline LRU/FIFO/LFU/heat-decay; frequency-aware only wins below ~10% share, LRU best >=25%. Other commenters (Volunteer-1, pashko-k) claim LRU 'performs very poorly' without data.
- Limitations stated: Author/PR: batch <= 32 only; single device at first; blog: 'We have not run these configurations ourselves'; competes with whole expert layers for VRAM; prompt processing falls (0.73x); regressions: GLM-5.3-Flash 33->16 tok/s on 2 GPUs at 43.5% hit; break-even with CPU-only at ~85% hit on PCIe4 x8 (Volunteer-1 claims ~74% of experts cached).
- Contradicts our findings: YES, partly. pubtech E.md says hit 0.95/0.96 at 16/32 GB local caches and A/B.md map cache bytes to 'token footprints' on a 128-expert, top-8 Qwen3-30B trace. By my arithmetic (computed by me, not in sources) a 16 GB machine cache of 9.15 GiB = 3.6k experts = 14.8% of the 512x48 experts, whereas on the 30B proxy the same footprint count is 2911 of 6144 slots = 47%. Real-model hit at ~14% share is 0.72-0.77 (PR) / 0.76 (offline trace), not 0.95. At 32 GB (~41% share) interpolated real hit ~0.93 vs our 0.96 (close). At 8 GB (~4% share) real hit is probably under 0.6 (62% at 7.8%) vs our 0.56 at 2.07 footprints (13% share on proxy). Our 29%-share figures (0.856 dense / 0.789 4x-sparse) bracket real 0.876-0.89, fine.

## P-0083: Accelerating Hybrid Inference in SGLang with KTransformers CPU Kernels (KVCache.AI and Approaching AI, LMSYS blog, 22 Oct 2025)

- ID read: https://www.lmsys.org/blog/2025-10-22-KTransformers/ raw HTML stripped (74 KB, 14 KB text); no arXiv id; raw text read: True
- Setup: Single-GPU: dual-socket Xeon Platinum 8452Y (36 cores x2), 1 TB DDR5 x2, A100 40GB (full-precision models) or RTX 4080 16GB (quantized). Multi-GPU: 8x L20 + dual Xeon Gold 6454S, DeepSeek-V3 int4 (avg 128 in / 512 out); ShareGPT run: DeepSeek-R1-0528 FP8/AMXINT4 experts on 8xL20, 1000 requests. Flags: --kt-num-gpu-experts 200, --kt-amx-method AMXINT4, --kt-cpuinfer 60-80. Experts in host DRAM (500GB+), computed on CPU with AMX; hot experts optionally in VRAM.
- Numbers (value | baseline | ratio | where):
  - Prefill speedup, single GPU + CPU: up to 20x | two baselines not named in text (figures are images) | not computable | 'Benchmark Results (Preview)' section
  - Decode speedup: up to 4x | same unnamed baselines | not computable | same
  - Multi-GPU, 1-concurrency 8 GPUs vs 1 GPU: +26% | 1 GPU | 1.26x | text; table is an image
  - Multi-GPU, 8-way concurrency 8 GPUs vs 1 GPU: +264%, ~20 tok/s per request | 1 GPU | 3.64x | text
  - ShareGPT 1000 reqs, 8xL20: 227.85 tok/s total, 87.58 output tok/s, median ITL 299.18 ms, mean 431.61, P99 1935.13 | - | - | text table
  - AMX kernels: 21.3 TFLOPS per socket, 3.9x PyTorch native | PyTorch | 3.9x | Section 1
  - NUMA-aware TP: up to +63% decode on dual socket | - | 1.63x | Section 2
  - Expert Deferral (lossy reorder): up to 1.45x decode, accuracy variation below 0.5% | - | 1.45x | Section 3
- Catalog / first-pass numbers:
  - --kt-num-gpu-experts keeps hot experts in VRAM to relieve CPU compute: **yes**. Text: more GPU experts reduce CPU compute pressure; also 'storing frequently used hot experts on GPUs'. No selection policy or ablation is given (static count, 200 in the examples).
  - DeepSeek-V3 Int4 about 16 tok/s decode (arXiv 2606.10493): **absent on this page**. 16 tok/s is not on the blog; it is KTransformers' number cited inside arXiv 2606.10493 (Sec 1, 2.1/2.2.1) while the same paper also says 22 tok/s (intro, Fig 17 measured on its platform).
- Transfers to our regime: **no**. Needs AMX Xeons and 500 GB+ DDR5; routed experts computed on CPU from RAM, no disk path; GPU required; 'hot experts on GPU' is a static count, not a streaming cache; our regime has 8-32 GB RAM and NVMe-streamed weights.
- Quality loss: Expert Deferral: accuracy variation below 0.5% (no benchmark named on page). Rest is lossless apart from int4 weights.
- Ablation: none for hot experts; Expert Deferral and NUMA gains are stated but charts are images
- Limitations stated: Labelled Preview / proof of concept, roadmap for full integration; table values for multi-GPU only as images; baselines unnamed; own tests.
- Contradicts our findings: No.

## P-0084: Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models (Kamahori, Tang, Gu, Zhu, Kasikci; University of Washington; ICLR 2025)

- ID read: arXiv 2402.07033 v3 (1 May 2025, 'Published as a conference paper at ICLR 2025') and v1 (10 Feb 2024, preprint, authors Kamahori, Gu, Zhu, Kasikci) both read as PDF text; raw text read: True
- Setup: Mixtral-8x7B 16-bit (>90 GB). v3 Table 1: Env 1 Quadro RTX 6000 24GB, PCIe3 x16 (32GB/s), Xeon Gold 6126 (48 cores), 56/256 experts on GPU; Env 2 RTX 6000 Ada 48GB, PCIe4 x16 (64GB/s), Xeon Platinum 8480+ (112 cores), 125/256 experts. v1: Env 2 was an L4 GPU. ShareGPT prompts. Baselines v3: DeepSpeed-MII 0.2.3 (ZeRO-Infinity), Mixtral-Offloading (extended to fp16, offload_per_layer 7 / 5), llama.cpp b2956 (ngl 8 / 16). Metrics: tok/s end-to-end (in 32-256, out 64-512), TTFT (in 512-4096), beam search width 4-16 (vs llama.cpp only). Software PyTorch, AVX512_BF16 CPU kernel. Experts in host RAM; no disk.
- Numbers (value | baseline | ratio | where):
  - v3 single-batch end-to-end, average over 15 length configs x 2 envs: 1.26x | best baseline llama.cpp | 1.26x (stated; bar values only in Fig 4) | Abstract, Sec 4.2, Fig 4 (p8-9)
  - v3 long-prefill TTFT: 1.30x avg (1.07x vs DeepSpeed-MII, 1.65x vs Mixtral-Offloading) | those baselines | as stated | Sec 4.2, Fig 5
  - v3 beam search: 11.57x avg | llama.cpp | 11.57x | Sec 4.2, Fig 6
  - v3 appendix TTFT / ITL across baselines: 1.13x / 1.43x avg | all baselines | as stated | App F
  - v3 Phi-3.5-MoE: 6.5x | DeepSpeed-MII (only supporting baseline) | 6.5x | App E, Fig 10
  - v1 single-batch latency: 8.2x (Env1 Quadro RTX 6000), 10.1x (Env2 L4) | Mixtral-Offloading (vs DeepSpeed-MII 19.4x / 22.5x) | as stated; >3 tok/s on a 24GB GPU | v1 abstract, Sec 4.2, conclusion
  - v3 hit rate from popularity placement, Env 1 (56/256): best 25.2%, random 21.9%, worst 18.7% | random | +3.3 pts | App (expert popularity), Fig 8 text
  - v3 hit rate Env 2 (125/256): best 53.0%, random 48.8%, worst 44.6% | random | +4.2 pts | same
- Catalog / first-pass numbers:
  - 8.2x (Quadro RTX 6000) and 10.1x (L4) single-batch latency vs weight-swapping baselines: **yes for v1 only; absent in v3**. These are in arXiv v1 (10 Feb 2024): vs Mixtral-Offloading specifically, Env 2 = L4. v3/ICLR replaced them with 1.26x/1.30x/11.57x vs llama.cpp etc. and Env 2 = RTX 6000 Ada. The catalog URL (ar5iv) probably serves v1. First pass said 'wrong, possibly earlier version, unverified': confirmed earlier version.
- Transfers to our regime: **no**. GPU required; experts in host RAM and executed on the CPU with AVX512_BF16; fp16 model; the gain is that CPU compute beats PCIe weight copy at batch 1; with NVMe-resident weights and CPU-only there is no GPU to orchestrate. Cache share of experts is 22-49% on GPU, popularity placement adds only 3-4 points.
- Quality loss: None (exact computation, 16-bit).
- Ablation: none for placement vs execution strategy, only the hit-rate calculation; sensitivity to datasets in App D.
- Limitations stated: Authors tested one main model plus Phi-3.5-MoE; baselines weak at batch 1; popularity profile assumed 'almost universal across datasets'.
- Contradicts our findings: No. 3-4 point gain of popularity-pinning over random agrees with our finding that offline hot lists are far worse than LRU (E.md: 0.30 vs 0.57 at 8 GB) in the sense that static placement gives little.

## P-0085: Bundle: (a) HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference (Zhong, Sun, Liang, Wang, Huang, Li; Peking U; arXiv 2504.05897v1, 8 Apr 2025, DAC 2025); (b) HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference (Tang, Liu, Hou, Pu, Wang, Heng, Li, Guo; SJTU/CUHK; arXiv 2411.01433v2, 6 Nov 2024); (c) MoE-Infinity (Xue, Fu, Lu, Sun, Mai, Marina; arXiv 2401.14361v3, 12 Mar 2025); (d) DAOP (Zhang, Aggarwal, Mitra; NUS; arXiv 2501.10375v3, 21 Aug 2026)

- ID read: all four PDFs read as text (pdftotext). Catalog url points only to 2504.05897; first pass opened (a) and (b) only; raw text read: True
- Setup: (a) RTX A6000 + Xeon Gold 5220R limited to 10 cores; Mixtral-8x7B (32 layers, 8 experts, top-2), Qwen2-57B-A14B (28 layers, 64 routed + 1 shared, top-8), DeepSeek-V2-Lite (26 layers, 64 + 2 shared, top-6); 4-bit Marlin; GPU expert cache ratio 25/50/75%; baselines llama.cpp, AdapMoE, kTransformers; prefill 32-1024 tokens, TTFT/TBT; host RAM size not stated. (b) RTX 4090 24GB + 256 GB RAM + 64 cores, PCIe4 32GB/s; Jetson AGX Orin 32GB unified, Samsung 980 PRO NVMe (7000 MB/s nominal, ~3000 practical); Mixtral-8x7B fp16/int8, Phi-MoE; batch 1; 60 Alpaca prompts; in/out [16,32],[16,128],[128,32],[128,128]; on Jetson baselines run int8 while HOBBIT runs int8 with int2 replacements (4090: fp16 with int4); built on llama.cpp. (c) single RTX A5000 24GB over PCIe4 (24GB/s), host RAM 128GB (Mixtral)/1TB (Arctic); DeepSeek-V2-Lite 31GB, Mixtral, NLLB-MoE, Switch, Arctic; prompt 512 decode 32. (d) i9-10980XE 18 cores, 130GB RAM, RTX A6000 48GB PCIe4; Mixtral-8x7B, Phi-3.5-MoE; HF Transformers; batch 1; ECR 25-46.9%.
- Numbers (value | baseline | ratio | where):
  - (a) prefill TTFT avg: 1.33x | kTransformers | 1.33x | Sec VI-B.1, Fig 7
  - (a) decode throughput avg: 1.70x | kTransformers | 1.70x | Sec VI-B.2, Fig 8
  - (a) ablation, Qwen2, 25% cache, prefill latency (s): baseline 1.47; +scheduling 1.17 (1.26x); +prefetch 1.39 (1.06x); all 1.13 (1.31x) | kTransformers | as shown | Table III
  - (a) ablation decode latency (s): baseline 0.21; +scheduling 0.14 (1.46x); +prefetch 0.18 (1.15x); +caching (MRS) 0.15 (1.38x); all 0.11 (1.86x) | kTransformers | 0.21/0.15=1.40, 0.21/0.11=1.91 (printed speedups 1.38/1.86; rounding) | Table III
  - (a) hit rate MRS vs LRU at 25% cache: Mixtral 36.2 vs 30.2; DeepSeek 52.7 vs 47.7; Qwen2 52.8 vs 45.0; at 75% Mixtral 83.3 vs 80.6 | LRU | +6.0 / +5.0 / +7.8 pts; +2.7 pts at 75% | Sec VI-D.1, Fig 9
  - (b) Jetson vs llama.cpp decode: 13.0x Mixtral, 18.9x Phi-MoE; prefill latency -79% / -80% | llama.cpp (mmap, page-faulting) | as stated | Sec 5.2, Fig 14a,b
  - (b) Jetson vs MoE-Infinity decode: 3.64x Mixtral, 9.93x Phi-MoE; prefill -60% / -72% | MoE-Infinity | as stated | Sec 5.2
  - (b) RTX 4090 vs MoE-Offloading / MoE-Infinity: 3.21x, 3.29x / 2.30x, 3.92x | - | as stated | Sec 5.2, Fig 14c,d
  - (b) RTX 4090 + CPU vs llama.cpp / Fiddler: 1.31x, 1.42x / 0.99x, 1.46x | - | as stated | Sec 5.4 (CPU assistant), Fig 15
  - (b) accuracy fp16 vs fp16+int4 (GSM8K / TruthfulQA): Mixtral 0.52 vs 0.51; 0.49,0.94 vs 0.49,0.93; Phi-MoE 0.83 vs 0.82 | fp16 | <=1 pt | Table 3
  - (b) ablation dynamic loading: 1.19x-1.57x (highest on Jetson, lowest on 4090+CPU) | without dynamic loading | as stated | Sec 5.5.1, Fig 16
  - (b) cache policy vs LRU / LFU: 4.69-8.68% / 2.13-4.19% miss-penalty reduction | LRU / LFU | as stated | Sec 5.5.3, Fig 18
  - (b) prefetch gain: ~10% prefill latency; decode 1.01x fp16, ~1.05x with int4 | no prefetch | as stated | Sec 5.5.2
  - (c) DeepSeek-V2-Lite TPOT, A5000: MoE-Infinity 155 ms avg (169 tail); vLLM 485; DeepSpeed 737; Mixtral-Offloading 1250; Ollama 2590 | - | 3.1x / 4.8x / 8.1x / 16.7x | Table 1 (p1)
  - (d) DAOP vs Fiddler, [256,512]: 4.52 tok/s Mixtral, 8.21 Phi-3.5; +40.4% | Fiddler | 1.40x | Sec V-B, Fig 9
  - (d) DAOP at ECR 25%: 3.23 tok/s Mixtral, 5.03 Phi-3.5 | - | - | Sec V-B
  - (d) headline: up to 8.20x vs caching/prefetching, 1.35x vs offloading | MoE-OnDemand / Mixtral-Offloading / DeepSpeed-MII / Fiddler | as stated; baselines <1 tok/s on Mixtral | Abstract, Sec V-B
- Catalog / first-pass numbers:
  - hybrid CPU-GPU scheduling and expert cache management (HybriMoE): **yes**. -
  - HOBBIT up to 9.93x decode vs offloading baselines: **partly**. 9.93x is vs MoE-Infinity, Phi-MoE, Jetson AGX Orin int8 vs int8+int2; vs llama.cpp on the same Jetson it is 18.9x (the true maximum); 4090: 1.31-1.42x vs llama.cpp, 0.99x vs Fiddler. 'Offloading baselines' conceals that HOBBIT replaces missed experts by int4/int2 copies (lossy, <=1 pt on 2 benchmarks).
  - first-pass: HybriMoE 1.33x/1.70x vs kTransformers: **yes**. figures only per model; averages as stated
  - first-pass: score cache beats LRU by 3-8 points: **yes**. 2.7-7.8 pts (HybriMoE MRS); HOBBIT is 4.7-8.7% miss-penalty (a different metric)
  - first-pass: HybriMoE host RAM / HOBBIT NVMe ~3GB/s: **yes**. HybriMoE does not state RAM; HOBBIT 3000 MB/s practical
- Transfers to our regime: **no**. HybriMoE: A6000 + kTransformers, host-RAM experts, 25-75% GPU cache ratio, 8-64 experts per layer. HOBBIT: closest regime (Jetson, NVMe) but the 18.9x/9.93x come from a baseline that page-faults (llama.cpp mmap on shared-memory Jetson) or a GPU-server design (MoE-Infinity), plus int2 substitution which is a quality change; on the RAM-rich 4090 it is 1.3x. MoE-Infinity: expert cache + prefetch in GPU with all experts in host RAM and large PCIe bandwidth. DAOP: A6000, Mixtral/Phi, expert substitution ('graceful degradation').
- Quality loss: HOBBIT: <=1 pt on GSM8K/TruthfulQA (fp16 vs fp16+int4; int8 vs int8+int2) for 2 models (Table 3). DAOP: Table V/VI says minimal impact, GSM8K the exception (discussion section not read in detail). HybriMoE, MoE-Infinity: exact.
- Ablation: HybriMoE Table III isolates scheduling/prefetch/caching on Qwen2 only (caching alone 1.38x decode vs kTransformers at 25% cache). HOBBIT Sec 5.5 isolates dynamic loading (1.19-1.57x), prefetch (<=1.05x decode), cache policy. No SSD-layout ablation.
- Limitations stated: HOBBIT: prefetch useless in decode because of mispredict penalty; HybriMoE: scores stop mattering at 75% cache; MoE-Infinity: needs host RAM for the whole model.
- Contradicts our findings: No contradiction; HOBBIT's finding that decode prefetching gives ~1.01-1.05x agrees with our prefetch results.

## P-0087: LExI: Layer-Adaptive Active Experts for Efficient MoE Model Inference (Chitty-Venkata, Madireddy, Emani, Vishwanath; Argonne National Laboratory)

- ID read: arXiv 2509.02753v1 (2 Sep 2025) PDF text; raw text read: True
- Setup: Mixtral-8x7B-Instruct, Qwen1.5-MoE-A2.7B, OLMoE-1B-7B-0924, MiniCPM-MoE, DeepSeek-V2-Lite-Chat, DeepSeek-VL2-Tiny. 4x H100 80GB (2 for DeepSeek models), vLLM FusedMoE, tensor parallel, batch 16, throughput = (input+output tokens)/end-to-end latency. Method: per-layer perturbation (Frobenius norm of output change between top-k_base and top-k) on Gaussian random inputs (data-free), then an evolutionary search for per-layer k under a global budget B (sum of k). Baselines: inter-pruning NAEE 12.5/25/50% and intra-pruning (Yang et al.) 12.5/25/50%. Tasks: 9 LM-Eval (ARC-c/e, BoolQ, HellaSwag, MMLU, OBQA, RTE, WinoGrande), Qasper F1, passkey, perplexity C4/PTB/WikiText, VLM MME/MMMU/ScienceQA.
- Numbers (value | baseline | ratio | where):
  - OLMoE, budget B=100: same throughput as 50% intra pruning, +10% accuracy; +15% vs 50% inter pruning | 50% intra / inter pruning | not a speedup over full k | Sec 5, Fig 4a (figure only)
  - Qwen1.5-MoE: at least +5.1% throughput vs both prunings, +0.5% accuracy | inter/intra pruning | 1.051x | Sec 5, Fig 4
  - Abstract version for Qwen1.5-MoE: 'same throughput on H100 with 10% better accuracy than traditional expert pruning' | pruning | - | Abstract (inconsistent with Sec 5 text, which gives the 10% to OLMoE)
  - Mixtral-8x7B perplexity: ~2.4k throughput at ~23 ppl on C4; ~2.2k at ~10 ppl on WikiText | inter pruning worse | - | Fig 7
  - DeepSeek-V2-Lite: +6.5% throughput at equal accuracy vs inter pruning; +6 accuracy points vs intra | pruning | 1.065x | Sec 5
  - Qasper (Qwen/DeepSeek): 35.5 F1 at ~4.1k throughput vs inter 34 at ~3.9k, intra 30 at ~3.75k; +5.1-9.3% throughput | pruning | 1.05-1.09x | Fig 5 text
- Catalog / first-pass numbers:
  - Static per-layer expert counts without retraining or data: **yes**. data-free: only weights and random Gaussian input; evolutionary search.
  - first-pass: OLMoE matches 50% intra-pruning throughput with ~10% higher accuracy (Fig 4): **yes**. Sec 5 says OLMoE; the abstract attributes the 10% to Qwen1.5-MoE, internal inconsistency in the paper. 'Accuracy' units (points vs relative) not stated.
- Transfers to our regime: **partly**. Idea (per-layer non-uniform k, chosen from weights alone) transfers; numbers do not: GPU H100, batch 16, all weights resident, throughput compared to pruned models rather than to the unreduced model, so the absolute speedup vs full top-k is not reported in any text table. The method reduces expert count per token and hence bytes per token in our setting, but the benefit there is untested. Search space k=1..k_base. No CPU or disk experiment.
- Quality loss: Accuracy/perplexity 'close to the base model' per figures only (Fig 4-8); no table with deltas to full-k; Mixtral perplexity ~10 on WikiText at ~2.2k throughput (base value not in text).
- Ablation: none isolating anything relevant to cache/layout; the pruning-vs-top-k comparison is the only comparison
- Limitations stated: Does not reduce memory footprint (so little use when memory-bound); inapplicable if k_base=1 (Llama-4); can be combined with pruning.
- Contradicts our findings: Paper's claim that per-layer k gives 'negligible accuracy loss' is not tested at 125B and sits against our OLMoE k=4 result (+45.6% relative ppl); LExI never shows a uniform k=4 of 8 baseline, so no direct contradiction.

## P-0088: Certain Head, Uncertain Tail: Expert-Sample for Test-Time Scaling in Fine-Grained MoE (Chen, Wang, Zeng, Shao, Qiu, Li, Liu, Cheng; CAS/others; ICML 2026 proceedings, 43rd ICML)

- ID read: arXiv 2602.02443v2 (1 May 2026) PDF text; raw text read: True
- Setup: Table 1: Qwen3-30B-A3B (128 experts, top-8), GPT-OSS-20B (32 experts, top-4), Ling-Lite-1.5 (64 experts, top-6), Qwen3-Next-80B-A3B (table shows 256 experts, top-10; from memory the real model has 512 routed experts, so the table entry may be wrong or I mis-aligned the scrambled column). Benchmarks AIME-120/2024/2025, HMMT-2025, MATH500-Hard, GPQA-Diamond, LiveCodeBench-v6-Lite; LightEval 0.9.1 + vLLM 0.10.2 on 8x A800-80GB, tensor parallel 8. Overhead tests: random WikiText-2 prompts, prompt 1024, batch 8, output 1024 (Table 3); prefill 256-1024 x batch 1-32 (Tables 6,7).
- Numbers (value | baseline | ratio | where):
  - Greedy accuracy when activating k/2: 'remarkably stable' on 5 models, 2 benchmarks | default top-k | no numbers in text | Sec 2.1, Fig 2a (figure only)
  - Reduced experts (k/2+1) pass@n: 'drops substantially' (64 samples, T=0.7) | default | no numbers in text | Fig 2b
  - Router score rank gap: rank1-rank5 gap > 3.9%; rank5-rank32 cumulative diff < 1.5% | - | - | Sec 2.2, Fig 2c
  - Recommended k_keep: floor(k/2)+1 or +2; below k/2 stability and exploration collapse | - | - | App B.1, Fig 5-6
  - Headline (not about speed): GPQA-Diamond pass@32 85.4 -> 91.9; BoN 59.1 -> 62.6 (Qwen3-30B-A3B-Instruct) | token sampling | +6.5 / +3.5 pts | Abstract; avg +4.28% (BoN), +3.15% (WMV) over 20 combos
  - Overhead of Expert-Sample, Qwen3-30B-A3B: prefill 137121 vs 136754 tok/s (+0.27%); decode 662.1 vs 666.7 (-0.70%) | no ES | 1.003x / 0.993x | Table 3
  - Overhead Ling-Lite-1.5: prefill 213463 vs 214802 (-0.62%); decode 1028 vs 1024 (+0.40%) | no ES | 0.994x / 1.004x | Table 3
- Catalog / first-pass numbers:
  - Greedy accuracy stays stable with activated experts halved: **yes, with qualifiers**. 'half' means k/2+1 (5 of 8 on Qwen3-30B, 6 of 10 on the 10-expert model), only on GPQA/AIME greedy, shown as a figure; no table value anywhere in the paper.
  - sampling-based metrics degrade more: **yes**. pass@n falls at k/2+1 (Fig 2b), no number given in text
  - first-pass: speed table is overhead only (<1%): **yes**. Tables 3,6,7: ratios ~100% for batch 1-32
- Transfers to our regime: **partly**. Evidence that halving active experts leaves greedy reasoning accuracy roughly intact on top-8 to top-10 fine-grained MoEs is directly on-topic for lowering bytes per token; but it is accuracy on GPQA/AIME only (not perplexity, long-form generation or tool use), at k/2+1 not k=4 of 10, GPU-served, and speed from smaller k is never measured. Our k=4 of 8 OLMoE perplexity +45.6% concerns a different metric.
- Quality loss: Not measured numerically as a table; figure says negligible for greedy at k/2+1 (about 5/8 or 6/10 of the experts retained, i.e. 62%/60%, not 50%).
- Ablation: k_keep sweep (App B.1) on AIME-120 for Qwen3-30B and Ling-Lite: stable region k/2 <= k_keep < k; no speed ablation.
- Limitations stated: Authors discuss diversity/stability, not inference cost; the paper's own aim is test-time scaling, k reduction is a probe.
- Contradicts our findings: Weakly: suggests half-k keeps greedy accuracy, our OLMoE perplexity at k=4 says +45.6%; the two are different metrics and different models. See family #13 prior art arXiv 2609.04575 for the reconciliation (renormalisation gain).

## P-0089: MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping (Huang, Wang, Yuan, Gong, Ding, Guo, Liu, Zhang; HKUST/Beihang/PKU; CVPR 2026)

- ID read: arXiv 2511.15690v2 (22 Feb 2026) PDF text; Fig 6 and Table 3 read as a rendered page image (page 7); raw text read: True
- Setup: Kimi-VL-A3B-Instruct (64 routed experts, top-6), Qwen3-VL-MoE-30B-A3B-Instruct (128 experts, top-8), InternVL-3.5-30B-A3B-HF (128 experts), InternVL-3.5-GPT-OSS-20B-A4B (32 experts, top-4); 13 image/video benchmarks; metric = average per-benchmark score as % of the original model's score. Training-free: globally-modulated local gating + dual-modality thresholds (text/vision) + frontier search; calibration 20 min to ~4 h on 8xH200. Efficiency on ONE H200, HF-transformers-style eager runs (prefill batch 8, decode sequence length 1024, batch 1/2/4), skipping ratios 83% (Kimi) and 88% (Qwen3-VL).
- Numbers (value | baseline | ratio | where):
  - Qwen3-VL-MoE-30B, skip 88% (rho=0.85), 13 benchmarks avg: MoDES 97.33% | k=8 original 100; k=1 60.11; NAEE 80.60; MC-MoE 86.66; DiEP 85.30 | 97.33/86.66 = 1.123x relative; +10.67 pts | Table 3 (p7)
  - Qwen3-VL prefill TTFT (ms), seq 256/512/1024: 193.6 / 303.2 / 691.8 | 371.1 / 616.3 / 1438.6 | 1.92x / 2.03x / 2.07x | Fig 6 lower (p7)
  - Qwen3-VL decode TPOT (ms/token), batch 1/2/4: 90.6 / 109.7 / 131.7 | 112.0 / 134.4 / 159.2 | 1.24x / 1.23x / 1.21x | Fig 6 lower (p7)
  - Kimi-VL (83% skip) prefill TTFT seq 256/512/1024: 81.2 / 140.0 / 297.2 | 165.5 / 278.0 / 642.1 | 2.04x / 1.99x / 2.16x | Fig 6 upper
  - Kimi-VL decode TPOT batch 1/2/4: 36.0 / 45.9 / 58.3 | 45.2 / 54.7 / 69.4 | 1.26x / 1.19x / 1.19x (printed 1.19x at bs 2,4) | Fig 6 upper
  - Abstract/intro statements: abstract: prefill 2.16x, decode 1.26x; intro: Qwen3-VL 2.03x and 1.24x | - | see above | Abstract, Sec 1
- Catalog / first-pass numbers:
  - 97.33% performance retained at 88% skipped (Qwen3-VL-MoE-30B): **yes**. 97.33% of original average score over 13 benchmarks at 88% skipped; the competitor 86.66% is MC-MoE; plain k=1 gets 60.11%.
  - prefill 2.16x, decode 1.26x (Qwen3-VL-MoE-30B): **no**. 2.16x and 1.26x are the best points of KIMI-VL-A3B (83% skip, seq 1024 prefill; batch-1 decode). For Qwen3-VL-MoE-30B Fig 6 gives prefill 1.92-2.07x and decode 1.21-1.24x; the intro itself says 2.03x/1.24x. The abstract sentence is ambiguous. First pass accepted 2.16x/1.26x for Qwen3-VL: wrong.
  - (GPU-resident, not disk-bound): **yes**. single H200, 128 experts in HBM
- Transfers to our regime: **no**. Vision-language, modality-specific thresholds, calibration search; GPU-resident H200 where decode is eager-HF memory/launch bound (112 ms/token for a 3B-active model), so 1.2x in decode is about compute reduction only; an 88% skip is an average over mostly vision tokens (text tokens skip fewer), not top-k cut to 1 of 8; no disk or CPU experiment.
- Quality loss: 13-benchmark average 97.33% of original at 88% skipped (Qwen3-VL), 96.25% at 83% (Kimi-VL); InternVL-3.5-30B 97.03%; GPT-OSS-based 97.89% at 75%. Quantisation combos in Table 2.
- Ablation: Table 4 ablates each component (thresholding single vs dual etc.) and Table 5 calibration datasets; nothing about caching.
- Limitations stated: Future work: combine with pruning/distillation; low-bit quantisation costs more accuracy (17.3% drop at 1.5 bits on Kimi).
- Contradicts our findings: No.

## P-0090: llama.cpp issue #19528 'Feature Request: --expert-override flag for MoE models' (author muscleriot; opened 2026-02-11; closed 2026-03-28 by github-actions stale bot, state_reason not_planned)

- ID read: GitHub REST API JSON for issue and comments (3 comments); raw text read: True
- Setup: Nemotron-3-Nano-30B-A3B (default 6 experts). No hardware, no benchmark. Maintainer CISC: 'There already is such an option; --override-kv <arch>.expert_used_count=int:8'. Author: 'I have confirmed --override-kv nemotron_h_moe.expert_used_count=int:8 works'.
- Numbers (value | baseline | ratio | where):
  - Quality claim in the request: '8-10 experts significantly improves summarization quality for long-context tasks while 4-6 is sufficient for chat' | default 6 | qualitative, no data | issue body, 'Motivation'
- Catalog / first-pass numbers:
  - top-k is a GGUF key: **yes**. LLM_KV_EXPERT_USED_COUNT in GGUF metadata
  - override via --override-kv llama.expert_used_count=int:N: **no**. Key is per architecture: nemotron_h_moe.expert_used_count (a later PR comment on #29887 uses qwen4exp.* keys for another override). First pass noted this.
  - no published quality benchmark found: **yes**. none in the thread
- Transfers to our regime: **yes (mechanism only)**. Confirms a runtime top-k override exists in stock llama.cpp with no rebuild; says nothing about speed or quality.
- Quality loss: none measured (assertion only that more experts help summarisation)
- Ablation: none
- Limitations stated: Closed because inactive for 14 days (stale bot), not because a maintainer rejected the flag; request for a friendlier flag unanswered.
- Contradicts our findings: No.

## P-0091: Achieving Cloud-Grade SLOs for Local Mixture-of-Experts Inference through CPU-GPU Hybrid Design (Wang, Hou, Ji, Qu, Zhang; Tsinghua / Xingyun Integrated Circuits; accepted OSDI '26)

- ID read: arXiv 2606.10493v1 (9 Jun 2026) PDF text; raw text read: True
- Setup: Own platform: 2x AMD EPYC 9355, 24x DDR5-6400 48GB (1.15 TB, 1228 GB/s theoretical), 2x RTX 5090 32GB. DeepSeek-R1/V3 671B (INT4/Q4_K_M 404GB, FP8), Kimi-K2 1T, DeepSeek-V3.1. Baselines KTransformers (own platform measured; plus its published numbers cited), ik_llama.cpp, llama.cpp. Contributions: stream-loading prefill (SLP), distributed SLP, intra-node prefill/decode disaggregation, AVX-512 FP8 GEMV, fine-grained CPU parallelism.
- Numbers (value | baseline | ratio | where):
  - Own prefill, SLP FP8: up to 1,200 tok/s; 32K prompt within 30 s | KT AMX INT4 'over 500 tok/s at 2048 tokens' (KT's published number, other platform) | 2.4x vs 500 (not same platform); 2.8x vs estimated KT AMX at 20-32K | Abstract, Sec 4 prefill, Fig 13/14
  - Own DSLP on 2x5090: 1,800 tok/s; 45K prompt in 30 s; 1.64x SLP | SLP | 1.64x | Abstract, Sec 4
  - Own decode, Q4_K_M / INT4 DeepSeek-R1, 1K-8K context: 28 tok/s | KTransformers 22 tok/s (measured on own platform), ik_llama.cpp 14 tok/s | 1.27x vs KT, 2.0x vs ik_llama | Fig 17; text says 25% / 1.25x
  - Own decode FP8 R1 671B: 21.5 tok/s short, ~20 up to 32K; Kimi-K2 22.4 | - | - | Fig 15, Sec 4.5
  - KT cited decode: ~16 tok/s (Intro, Sec 2.2.1: 'e.g. 16 tokens/s on DeepSeek-V3 INT4 [8]') and 22 tok/s (abstract-intro: 'acceptable 22 tokens/s'; Fig 17 measured) | - | - | Sec 1, 2.1, 2.2.1
  - KT implied memory bandwidth: ~221 GB/s effective = ~50% of nominal DDR5 (from 74% CPU time and 20.4B active params) | - | - | Sec 2.1
  - KT cited prefill: over 500 tok/s at 2048-token prompt; 8K prompt ~20 s TTFT | - | - | Sec 2.2.2
  - Quality, FP8 vs official (Exact Match): DeepSeek-V3.1 MMLU-Redux 91.00 vs 91.8; MMLU-Pro 83.41 vs 83.7 | official | -0.8 / -0.3 pts | Table 2
  - Concurrency latency: worst case 1.67x increase vs KT 2.53x | own baseline | - | Fig 18
- Catalog / first-pass numbers:
  - KTransformers reaches over 500 tok/s prefill at 2048 tokens with AMX: **yes, but it is KT's published number cited by this paper (Sec 2.2.2), not measured here**. paper's own prefill is 1,200 tok/s (FP8, SLP) and is the title claim
  - streaming experts overlaps transfer with GPU compute: **yes**. it is the paper's own SLP (weights streamed host->GPU overlapped with GPU GEMM at sub-layer granularity), not KT
  - catalog row via P-0083: KT ~16 tok/s decode: **partly**. the paper quotes 16 (published KT) and 22 (measured / intro); both are KT, not this paper (28)
- Transfers to our regime: **no**. 1 TB DDR5 across 24 channels, EPYC 9355 pair, 2x5090, DeepSeek-scale; decode is memory-bandwidth-bound from host RAM (28 tok/s needs ~hundreds of GB/s); prefill is compute-bound on GPU with weights streamed over PCIe. None of 8-32 GB RAM, NVMe, CPU-only batch 1. One useful general point: decode on large MoE is memory-bandwidth-bound and engines reach only ~50% of nominal bandwidth.
- Quality loss: FP8 keeps quality (Table 2: within 0.3-0.8 pts of official); INT4 comparisons are about speed only; KT 'rerouted' model is called capacity-reduced.
- Ablation: Per-technique results (SLP vs AVX, DSLP vs SLP, kernel Table 1) but nothing on cache/layout/prefetch of experts from disk.
- Limitations stated: Authors: results cover a dual-socket, PCIe-attached commodity node; lower-end bandwidth/PCIe discussed in Sec 6 (not itemised).
- Contradicts our findings: No.

# Per-family sections

## Family s1 #12 (GPU / fast tier as resident expert cache; rows P-0082..P-0085)

Further prior art, read in raw text unless marked:
- Eliseev and Mazur, Fast Inference of MoE Language Models with Offloading, arXiv 2312.17238v1 (28 Dec 2023), read: LRU cache of k experts per layer plus speculative loading plus mixed quantisation, Mixtral-8x7B on T4/3060/3080 Mobile, 2-4 tok/s (Table 2). Establishes the small-cache LRU idea.
- MoE-Infinity (2401.14361v3), DAOP (2501.10375v3): read, see P-0085.
- SpecMD, Hoang, Jaiswal, Samragh, Cho (Apple), arXiv 2602.03921v1 (3 Feb 2026), read abstract and setup: benchmarks cache policies on OLMoE-1B-7B, Mixtral-8x7B, Phi-3.5-MoE, Qwen1.5-MoE on one A100 with emulated hardware; claims access is not LRU/LFU-temporal; Least-Stale 88-92% hit at 5% cache on OLMoE, TTFT -10.7 to -34.7%. Caveat: small expert counts (64) and emulated; a 7-8 point LRU hit gap vs real 512-expert traces is not shown.
- FlashMoE, arXiv 2601.17063v2 (5 Oct 2026), read: closest regime to ours (SSD-resident experts on a desktop): Ryzen 5 9600X, 64 GB DDR5, RTX 5070 Ti 16GB, SK hynix P51 7.4 GB/s NVMe, OLMoE-1B-7B and Qwen3-30B-A3B; ML cache beats LRU hit by 21% and LFU by 51% (OLMoE) but token speed gain over LRU only about 22% (OLMoE) and 7% (Qwen3-30B-A3B). The "up to 2.6x" is versus other systems, not versus an LRU cache. Reports LRU-evicted experts reused within 5 steps 34.2% of the time vs 0.1% for Belady.
- llama.cpp PR #29887 and #30112 (this batch, P-0082) and an RFC #23324 (mentioned by the first pass, not re-read).
- MoEpic 2509.08342, ExpertFlow 2410.17954, ProMoE, DALI 2602.03495: seen in search snippets only (from search results: MoEpic Qwen1.5-MoE 20/60 cached hit RND 33.0, LRU 38.9, LFU 43.7, LCP 45.4); not opened here, UNREAD.

What is new in OUR variant: caching per se is not new (Eliseev 2023, MoE-Infinity, HOBBIT, DAOP, HybriMoE, llama.cpp PR #29887). Closest: FlashMoE (SSD experts, desktop, OLMoE/Qwen3-30B-A3B, DRAM left at ~1 GB) and PR #29887 (real 125B-class routing, host RAM). Nothing read measures a CPU-only 8-32 GB box streaming a 512-expert top-10 model from NVMe, which is the regime difference; the mechanisms are shared.

Result that changes what we should believe: the real-model hit rates in PR #29887 and the offline trace study in its comments (62.1/75.9/87.6/91.2/97.4% LRU at 7.8/14/25/31/62.5% of experts) imply our simulated 16 GB hit of 0.95 is optimistic (real about 0.72-0.77 at the equivalent 14% share), because the 30B proxy maps bytes to footprints and ends with a cache of 47% of the experts. The 32 GB figure (~0.93-0.96) and the 29% points hold. Recommend re-running the 8/16/32 GB sweeps on traces of the actual 512-expert model. Caveat: comments are single-user, unverified GitHub posts, and workload mix moves hit by 10+ points.

## Family s1 #13 (cut active experts k, training-free; rows P-0087..P-0091)

Further prior art:
- Training-Free Halving of Activated Experts in Fine-Grained MoE, Chen and Yao, arXiv 2609.04575v1 (4 Sep 2026), read: key finding that renormalisation over the top-k_1 survivors inflates expert-branch gain; normalising by top-k_2 mass fixes it. Qwen3.6-35B-A3B (256 experts, k=8): 8->4 costs 4.65 MMLU points under standard renormalisation but 0.35 with k_2=16 (p=0.66); Qwen3.5-397B-A17B (512 experts, k=10, 60 layers): 10->5 costs 2.10 points at k_2=k_1 but 0.55 at k_2=10 (p=0.24); perplexity at k_1=8 of 10 on 397B: +1.06% default vs +0.46% at k_2=10 (Table 4); removing renormalisation is catastrophic (+128% ppl). Perplexity and MMLU prefer different k_2. This is the same architecture as ours (512 experts, top-10). No speed measured, GPU/eager evaluation.
- Elbow-Based MoE Routing, Pan et al., arXiv 2608.04401v1 (5 Aug 2026, ICLR 2026 TTU workshop), read: per-token adaptive k on OLMoE only (cap 8), latency -5.3%, six benchmarks, accuracy maintained. Small gain, one model.
- Eliseev (above), MoE-Spec/ASET/SERE/Ada-K/NAEE: snippets only (ASET and ExFold 2608.24938 and a NeurIPS 2026 listing 'Don't Skip the Baseline' appeared in search; not opened, UNREAD). The NeurIPS listing snippet reports that static k reduction is a strong baseline.
- DAOP/HOBBIT (read) replace missed experts by approximations rather than dropping them.

What is new in OUR variant: lowering k is known and shipped (llama.cpp --override-kv, P-0090; LExI; Expert-Sample; MoDES; 2609.04575). What none of the rows isolate is the bytes-per-token argument for disk-bound batch-1 decode (k reduced -> misses per token reduced); the closest are 2609.04575 (quality at 512 experts, top-10 to 5) and BigMoeOnEdge (first-pass, not re-read). Our measured k=4 quality cost (+45.6% relative ppl on OLMoE, naive renormalisation) is plausibly the renormalisation artefact described by 2609.04575; their fixed-compute k_2 trick is untested by us and costs nothing at inference. For a 125B-class top-10 model, 10->5 with k_2 at k looks to cost about 0.5 MMLU points on the 397B sibling (p=0.24, 2000 questions), so the quality side may be far less severe than our OLMoE number.
