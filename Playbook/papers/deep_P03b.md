# deep_P03b: raw full-text re-check of P-0069..P-0080 (11 rows)

Raw text read for all 11 rows; none unread. Working copies of texts are in the scratch folder only. Row P-0074 is not mine.

## P-0069  Fate: Fast Edge Inference of Mixture-of-Experts Models via Cross-Layer Gate (Fang, Hong, Huang, Lyu, Chen, Yu, Yu, Zheng; Sun Yat-sen Univ / HKUST / Huawei / Peng Cheng Lab)

- Read: arXiv:2502.12224v2 (7 May 2025); catalog URL points to v1, not read
- Setup: DeepseekMoE 16.4B (top-6 of 64 + shared) and Qwen1.5-MoE 14.3B (top-4 of 60 + shared). High-end PC: RTX 3090 24GB, i9-9900X, 128GB host RAM, PCIe3 x16 (16 GB/s). Low-end PC: GTX 1080Ti 11GB, E5-2650v4, 64GB RAM, PCIe 1.0 x16 (4 GB/s). All experts live in host RAM (no SSD). Prefill inputs 128/256/512 tokens; decode: input <=64, output <=1024; datasets ChatGPT-prompts, HumanEval, GSM8K. Batch size not stated (single request implied). Baselines: Load on Demand (LoD) and Expert-Activation-Path (EAP) prefetch. Experts quantized INT4 (INT2 for non-popular experts in prefill).
- Numbers:
  - prefetch accuracy using Gate_in_i for layer i+1 (decode, no training): 78.79% (Table 1 text; lowest of the three input positions 76.94%); baseline -; ratio -; [Sec 4.2, Table 1]
  - prefetch accuracy claimed in contributions: 97.15% (with similar-routing-weight refinement, per intro); baseline -; ratio -; [Sec 1 contributions]
  - expert hit rate with shallow-favoring cache: 99.08% - obtained by caching ALL experts of layers 0-3 of Qwen1.5-MoE; ambiguous which accesses are averaged; baseline -; ratio -; [Sec 4.4 (p.~7)]
  - prefill speedup high-end PC: up to 4.5x vs LoD, 1.9x vs EAP (804 tok/s at prompt 512, DeepseekMoE); baseline LoD / EAP (absolute baselines only in Fig 11); ratio as stated; [Abstract; Sec 5.2.1, Fig 11]
  - prefill speedup low-end PC: up to 1.8x vs LoD, 1.5x vs EAP; baseline LoD/EAP; ratio as stated; [Sec 5.2.1]
  - decode speedup: up to 4.1x vs LoD, 2.2x vs EAP (abstract); high-end 3.7x (ablation); low-end 2.3x vs LoD, 1.6x vs EAP; baseline LoD/EAP; ratio as stated; [Abstract, Sec 5.2.2, 5.4, Fig 12-14]
  - ablation decode: prefetch alone / +cache / +quant: prefetch alone avg +30%; caching on top up to 2x; quantization adds >2x on top in places; baseline LoD; ratio -; [Sec 5.4, Fig 14]
  - MMLU Qwen1.5-MoE: 59.6 (BF16) -> 59.7 high-end / 58.4 low-end; baseline BF16; ratio -0.1 / -1.2 pts; [Table 3]
  - MMLU DeepseekMoE: 44.4 -> 44.3 high-end / 42.0 low-end; baseline BF16; ratio -0.1 / -2.4 pts; [Table 3]
- Catalog / first-pass numbers:
  - Reports a 99% expert hit rate -> partly: 99.08% only for the experiment that fully caches layers 0-3 (Qwen1.5-MoE, 60 experts/layer); not a global hit rate at a small cache share
  - treats prefill and decode as different stages -> yes: prefill: popularity-reordered compute + INT2 for unpopular experts; decode: cross-layer prefetch + shallow cache (INT4 only)
  - first-pass: prefill up to 4.5x vs LoD, 1.9x vs EAP; decode 4.1x/2.2x -> yes: matches abstract; per-figure baselines tok/s are only in Fig 11/12 (not extractable)
  - first-pass: batch not given -> yes: confirmed, not stated
  - first-pass: 128GB / 64GB RAM -> yes: confirmed
- Transfers to our regime: no
- Why: GPU (3090/1080Ti) + PCIe from host RAM that holds ALL experts, 14-16B models with 60-64 experts, top-4/6; our regime is CPU-only, NVMe streaming, 512 experts top-10. Shallow-favoring cache needs a memory budget that holds whole shallow layers (layers 0-3 = 4*60 experts); prefetch gains come from overlapping PCIe with GPU compute. Decode speedups are mixed with INT4/INT2 re-quantization (lossy) so they are not cache-only. Cross-layer gate prefetch accuracy 77-79% (decode) is a usable idea in principle but at 512 experts/top-10 accuracy is unmeasured.
- Quality loss: Lossy part is INT2/INT4 experts: MMLU -0.1 (Qwen high-end), -1.2 (Qwen low-end), -0.1 / -2.4 (Deepseek). Table 3. GSM8K/HumanEval also listed (figures).
- Ablation: Fig 14 decode breakdown (LoD -> +prefetch -> +cache -> +quantization): prefetch avg 1.3x, caching up to 2x on top, quantization further; cache effect separable from layout; no layout study. Figure-only values.
- Stated limitations: No explicit limitations section found; related-work says compression is orthogonal. Not stated: batch size, SSD, models >16B.
- Versus our findings: No. 'Prefill touches many experts' is consistent with our window-union proxy. It does not test pinned-vs-LRU; its 99% is a full-layer cache, which does not conflict with our finding that a small static pin (10% share) is below LRU.

## P-0070  MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache (Xue, Fu, Lu, Sun, Mai, Marina; Univ. Edinburgh)

- Read: arXiv:2401.14361v3 (12 Mar 2025)
- Setup: Single RTX A5000 24GB, PCIe 4.0 (Table 1 caption says 24GB/s, Sec 5 says 32GB/s - internal inconsistency). Batch size 1 (stated design point). Host RAM sized to fit the whole model: 64GB Switch, 256GB NLLB, 32GB DeepSeek-V2-Lite (31GB), 128GB Mixtral-8x7B, 1TB Arctic. No SSD tier. Prompt 512, decode 32 tokens, 290 tasks (BIGBench, FLAN, MMLU). Baselines: DeepSpeed-Inference, Llama.cpp/Ollama, Mixtral-Offloading, BrainStorm* (own re-implementation, not open source), vLLM (where the model fits).
- Numbers:
  - DeepSeek-V2-Lite TPOT avg, MoE-Infinity: 155 ms (tail 169 ms; GPU idle 51 ms); baseline DeepSpeed 737, Mixtral-Offloading 1250, Ollama 2590, vLLM 485 ms; ratio 4.75x, 8.06x, 16.7x, 3.13x (computed); [Table 1, p.1]
  - abstract range: 3.1-16.7x per-token latency vs systems; baseline -; ratio computed 485/155=3.13, 2590/155=16.7: abstract matches Table 1; [Abstract]
  - conclusion range: 2.7-13.7x; baseline -; ratio differs from abstract (internal inconsistency); [Sec 6]
  - Mixtral-8x7B decode latency: 836 ms; 1.4x vs BrainStorm / Mixtral-Offloading (worst case: 8 big experts, high activation ratio); baseline -; ratio 1.4x; [Sec 5.1, Fig 7]
  - Switch / NLLB / DeepSeek latency: 155 / 531 / 155 ms, 'comparable to fully-on-GPU'; baseline -; ratio -; [Sec 5.1]
  - EAMC capacity needed: ~3% of request count (<=120 EAMs) suffices; baseline -; ratio -; [Sec 5.3, Fig 9]
- Catalog / first-pass numbers:
  - Per-request activation matrix used to predict reuse and drive cache or prefetch -> yes: EAM (expert activation matrix) per request, clustered into an EAM collection; matched online to guide prefetch + replacement
  - closest 'learn per-prompt hot set' prior work -> yes: per-request tracing, but the cache is a GPU cache above host RAM; no calibrate-then-pin on a first-W-token window per se
  - first-pass: TPOT 155 vs 737/1250/2590 -> yes: confirmed (also vLLM 485)
  - first-pass: 3.1-16.7x abstract vs 2.7-13.7x conclusion -> yes: confirmed
  - first-pass: Mixtral ~836 ms (1.4x) -> yes: confirmed
- Transfers to our regime: partly
- Why: Batch 1 and per-request sparsity are exactly our premise; the idea (trace activations per request, cache by them) transfers. Speedups do not: baselines are weak GPU-offload engines (DeepSpeed, Ollama, an own BrainStorm port) and nothing is compared to a plain LRU over explicit reads; host RAM holds all experts; GPU compute, PCIe 4.0. No SSD. The authors' own fine-grained win is on DeepSeek-V2-Lite (64 experts top-6); the 'worst case' Mixtral gain over a good prefetch/cache baseline is 1.4x.
- Quality loss: None (lossless caching/prefetch).
- Ablation: No clean cache-vs-prefetch ablation found in the text; Sec 5.3 studies EAMC capacity and workload-change robustness (Table 3 hit pattern counts).
- Stated limitations: Long context shrinks GPU buffer for experts (latency 50->160 ms from 4K to 128K ctx, OOM at 2^17). Requires whole model in host memory. Workload switches degrade (Table 3).
- Versus our findings: No direct contradiction: our static frequency pin matched a warmed LRU within -0.02..+0.06 over 20 tokens and lost 0.05-0.09 afterward; MoE-Infinity is not compared with explicit reads + LRU, so its multiples are not evidence against our measurement (consistent with pubtech note).

## P-0071  In-depth Analysis on Caching and Pre-fetching in Mixture of Experts Offloading (Lin*, He*, Chen*; Carnegie Mellon Univ.)

- Read: arXiv:2511.05814v1 (8 Nov 2025)
- Setup: Mixtral-8x7B-Instruct with HQQ 4-bit attention and 2-bit experts, built on the mixtral-offloading code (baseline = its LRU cache). GPUs: A100, A6000, L40, RTX 3090. Batch 1, cache size = 4 experts/layer, ONE 50-word prompt ('Introduce yourself...') for the cache traces and speed comparison; MMLU subset of 57 questions (1 per subject), 4-shot CoT for the offload-count sweep. 32 MoE layers x 8 experts, top-2.
- Numbers:
  - LRU, offloads per layer 4/5/6 (A6000): MMLU 63.16/61.40/59.65; 4.23/4.78/7.16 tok/s; peak mem 11148/9146/7128 MB; baseline -; ratio -; [Table 1]
  - LFU vs LRU tok/s: A100 3.33->3.64; A6000 2.34->4.32; L40 4.17->4.65; 3090 3.07->3.09; baseline LRU; ratio 1.09x; 1.85x; 1.12x; 1.01x (computed); [Table 2]
  - precision / recall of cache vs actually routed experts: LFU 29.9 / 59.8; LRU 29.1 / 58.2; baseline -; ratio +0.8 / +1.6 pts; [Table 2]
  - speculative next-layer gating precision=recall: 84.6% (one prompt); no speed measured; baseline -; ratio -; [Sec 5.4]
- Catalog / first-pass numbers:
  - LRU beats random by about 5-15% on Mixtral -> absent: No comparison with random eviction anywhere in the text (grep: 'random' only appears for the 12.5% chance-level expert reuse)
  - recent MoEs have weak temporal locality -> partly: paper says temporal locality exists but is not strong; expert imbalance (frequency) is much stronger - on Mixtral only; recent fine-grained MoEs are not tested
  - it is a cache-policy study -> yes: LRU vs LFU only
  - first-pass: LFU vs LRU A6000 2.34->4.32 -> yes: confirmed; NOTE Table 1 gives LRU A6000 at 4.23 tok/s for the same 4-offload setting, so the 2.34 in Table 2 is inconsistent with Table 1 (A6000 baseline probably unstable); the 84.6% LFU gain is probably noise-inflated
  - first-pass: precision 29.1% recall 58.2% -> yes: confirmed
- Transfers to our regime: no
- Why: Mixtral 8 experts top-2 with 2-bit experts on a GPU; cache of 4 of 8 experts per layer (50% share) vs our 2-8 footprints of 512; one prompt; tok/s noise. Qualitative message (frequency > recency, imbalance stronger than temporal locality) is the only portable part and is anecdotal.
- Quality loss: None for caching; the 2-bit expert quantization is the baseline's. Offloading more experts lowers MMLU (63.2->59.7) in Table 1 only because of the baseline's different configs.
- Ablation: LRU vs LFU only; no prefetch speed ablation (speculative loading implemented but not enabled).
- Stated limitations: Authors: only Mixtral 8x7B; one prompt, outputs vary run to run (stochastic), need varied cache sizes and datasets; overlap not implemented.
- Versus our findings: Partially: it says frequency beats recency (LFU>LRU), while our pubtech runs show a static pin below LRU at 10% share and warmed LRU as strong as pin. Their LFU is a dynamic (cumulative) counter over the generation, not a static pin from a 20-token window, so not the same test. Evidence is single-prompt Mixtral.

## P-0072  llama.cpp issue #20757: Feature Request: Two-tier GPU+RAM expert cache for MoE offload (pluggable eviction policy), by e1n00r

- Read: GitHub API JSON issue + 22 comments, created 2026-03-19, closed 2026-04-08 (state_reason completed)
- Setup: PoC 'tinyserve' in Python/HF transformers, GPT-OSS-120B (128 experts/layer, 36 layers, top-4, ~57GB expert weights) on RTX PRO 2000 8GB VRAM; host RAM size not stated. Proposed: Tier1 GPU slot buffer (SLRU + frequency-gated admission), Tier2 pinned RAM, Tier3 SSD mmap with POSIX_MADV_WILLNEED readahead and POSIX_MADV_DONTNEED page release. Informal.
- Numbers:
  - no-cache CPU offload, same hardware: ~0.5-1 tok/s; baseline -; ratio -; [issue body]
  - PoC cold start (0-80 tokens): 1.9-2.5 tok/s, hit 48-56%; baseline -; ratio -; [issue body table]
  - PoC warming (80-160): 2.6-5.1 tok/s, hit 53-78%; baseline -; ratio -; [issue body table]
  - PoC steady state (160+): 12-14 tok/s, hit ~98-100%; baseline 0.5-1 tok/s; ratio 12-28x (computed; the baseline range is wide); [issue body table]
  - SLRU+admission filter vs plain LRU: +8-15 pts steady-state hit rate on mixed prefill+decode; baseline LRU; ratio -; [issue body (unverified claim by author)]
  - comment 2: Vulkan RX 9070 XT 16GB, Gemma 4 26B-A4B: 27.9 -> 35.1 tok/s (+26%); hit 47% at 100 tok -> 67% at 500+; baseline MoE on CPU; ratio 1.26x; [comment by martinalderson]
  - comment: Qwen3-Coder-Next MXFP4 hot/cold static split, 5090+64GB: 72.1 -> 90.5-91.5 tok/s (+26%) at 68.6% coverage; gpt-oss-120B 36.2->47.6 (+31.6%); adaptive hot-list doubled coverage 28->45% but gave only +2.3% tok/s; baseline --n-cpu-moe 24; ratio 1.26x; [comments by JigSawPT]
  - comment: GLM-4.5-Air Q4_K_M, 32GB RAM + 16GB slab: 0.66 -> 0.60 tok/s; 24 GiB RAM tier collapsed to 0.12; baseline stock; ratio 0.91x; [comment by koren1712]
  - comment: Qwen3-Coder-30B-A3B fitted regime, 32GB RAM + 4080: stock mmap 8.02; streaming no slab 2.70 (141.5 GB streamed); 8 GiB slab 8.70 (8.18 GB); 16 GiB slab 7.16; baseline stock; ratio 1.08x for 8 GiB slab; [comment by koren1712]
  - comment: M1 Pro 16GB, Drama 3 GGUF 40x256 experts: QD1 read 327 MB/s vs 2-3 GB/s sequential; 16 read threads RTF 25.5->16.8; cache 32->64 hit 42.3->55.3%; baseline 1 thread; ratio 1.52x RTF; [comment by SurgeonTalus]
- Catalog / first-pass numbers:
  - CPU offload without a cache runs at about 0.5-1 tok/s -> yes: issue body (GPT-OSS-120B, 8GB GPU, --cpu-moe)
  - Proposes a page-cache-aware expert cache -> partly: proposes a 3-tier cache (VRAM slot buffer, pinned RAM, SSD mmap); the SSD tier does use madvise WILLNEED/DONTNEED (page-cache hints) but the core proposal is VRAM slots. First-pass 'not page-cache-aware' is therefore too strong.
  - first-pass: steady 12-14 tok/s at 98-100% hit -> yes: the 98-100% is measured on the author's own PoC trace, not independently
  - first-pass: closed completed 8 Apr 2026, no merged PR shown -> partly: the issue page itself shows no PR; comments mention forks/PRs #27861, #25294, discussions #28248 etc. (commenters' claims, not checked)
- Transfers to our regime: partly
- Why: Closest engineering analogue to ours but GPU slot buffer, VRAM+pinned RAM, all experts loaded in RAM for the 12-14 tok/s number; the 98-100% hit rate on 128 experts top-4 with N slots is a small-model, single-user trace. Community data in comments is directly relevant: with 32 GB RAM and a model larger than RAM, caching gave 0.66->0.60 tok/s (net loss) and 16 GiB beat 8 GiB only in the fitted regime; QD1 reads on an SSD run at 327 MB/s vs 2-3 GB/s sequential and parallel reads help 1.5x. These support our NVMe worry.
- Quality loss: None (lossless); one comment reports equal perplexity within error bars.
- Ablation: Informal: SLRU+admission vs LRU (+8-15 pts hit) claimed by author, no table; comment tables isolate slab size (8 vs 16 GiB) and read threads.
- Stated limitations: Author: Python developer, needs C++ implementer; double-buffering out of scope. Comment threads: residency is never free (prefill -9.9% with 4.4 GiB resident hot copies), adaptive hot-lists matter little on flat routers (gpt-oss +2.3%).
- Versus our findings: No. The comment-thread numbers (flat routers gain little from better hot sets; extra residency can hurt on 32GB RAM; QD1 reads slow) agree with our small-cache and NVMe-bound findings. Comments are untrusted, informal.

## P-0073  FlashMoE: Reducing SSD I/O Bottlenecks via ML-Based Cache Replacement for Mixture-of-Experts Inference on Edge Devices (Kim, Lee, Han, Yoo, Kim; KAIST/Chung-Ang/Yonsei)

- Read: arXiv:2601.17063v2 (5 Oct 2026); catalog URL is v1, not read
- Setup: OLMoE-1B-7B (64 experts/layer) and Qwen3-30B-A3B (128 experts, top-8). Ryzen 5 9600X, RTX 5070 Ti 16GB (PCIe 5), SK hynix P51 NVMe 7.4 GB/s, 4x16GB DDR5 = 64GB of which ~1GB left free (the rest locked) so page cache cannot help. Expert cache lives in VRAM (eq. 1), 12-24 of 64 (OLMoE) and 16-24 of 128 (Qwen3) experts per layer = 19-38% and 12.5-19%; cache 10.7-14.4 GB for Qwen3 (Fig 7). Experts stored as one .pt file each, loaded by torch.load (~3 ms per expert from SSD, FFN 158 us). Baselines: LRU, LFU, LIFO, ARC, LeCaR; systems Fiddler, DAOP, llama.cpp, HF. Policy = small per-layer FFN (~113KB) trained on routing traces from 512 TriviaQA samples (<=512 tokens) with Belady-optimal labels, ~2h pipeline. Batch 1 decode.
- Numbers:
  - hit rate vs LRU / LFU (OLMoE): FlashMoE +21% vs LRU, +51% vs LFU (relative vs absolute not specified); I/O reduction 22% / 35%; baseline LRU, LFU; ratio -; [Sec 4.3, Fig 6]
  - hit rate vs ARC / LeCaR: up to +28% / +21%; baseline ARC, LeCaR; ratio -; [Sec 4.3]
  - Belady vs LRU hit: 86% vs ~73% (1.9x more misses for LRU); baseline LRU; ratio 1.19x hit; [Sec 2.2]
  - decode speed vs best heuristic (LRU): +22% OLMoE, +7% Qwen3-30B-A3B; baseline LRU; ratio 1.22x / 1.07x; [Sec 4.4, Fig 7b,e]
  - load + prefill total latency vs llama.cpp / Fiddler & DAOP: 2.5x / 4.1x (it is load+prefill total, not prefill alone); initial load 4x / 6.8x faster; baseline llama.cpp / Fiddler; ratio as stated; [Sec 4.2, Fig 7a,d]
  - experts touched by prefill: 47% at 32 tokens, 58/64/67% at 64/128/256 tokens; baseline -; ratio -; [Sec 4.2]
  - expert loading share of decode time: over 70%; baseline -; ratio -; [Sec 3, Fig 5]
- Catalog / first-pass numbers:
  - SSD-backed expert cache with learned replacement on edge devices -> yes: GPU-VRAM cache over SSD, ML (not RAM) replacement
  - first-pass: hit +21% vs LRU (OLMoE) -> yes: also +51% vs LFU; abstract 'up to 51%' is vs LFU, easily misread as vs LRU
  - first-pass: decode +22% OLMoE, +7% Qwen3 vs LRU -> yes: confirmed
  - first-pass: prefill 2.5x vs llama.cpp -> partly: 2.5x is model load + prefill total latency; abstract also says up to 2.6x vs existing systems
  - first-pass: 16GB GPU + 64GB RAM -> partly: 64GB installed but only ~1GB left usable (rest locked) so it is effectively RAM-starved; cache in VRAM
- Transfers to our regime: partly
- Why: Right problem (SSD-backed, RAM-starved, batch 1) and honest page-cache starvation, but GPU 16GB VRAM cache of 13-38% of experts; our CPU-only cache is a RAM region of 2-8 footprints (<2% of 512 experts). Gains over LRU shrink with model size (22% -> 7%); policy needs a trained net per model and Belady-trace training data; runs on a PyTorch stack (torch.load) so absolute tok/s not comparable to llama.cpp. The Belady gap (86% vs 73%) bounds what any replacement policy can gain at their cache shares.
- Quality loss: None (lossless).
- Ablation: Cache-policy ablation (Fig 6, Fig 7b,e) isolates replacement from layout/prefetch: only replacement differs; no prefetch or layout study.
- Stated limitations: No dedicated limitations section found. Trained per model on TriviaQA; Qwen3-30B gain is only 7%; cache share and GPU assumption.
- Versus our findings: No. Consistent with our finding that nothing beats a warmed LRU by much (7% on the larger model; Belady only 13 points above LRU). Does not test static pinning.

## P-0075  SMIDT: High-Performance Inference Framework for MoE Models with Dynamic Top-K Routing (Jin, Fu, Tang, Bai, Wang, Zhu, Fang, Gong, Li; USTC), AAAI-26, vol 40 no 27, pp 22444-22453, 14 Mar 2026

- Read: AAAI OJS PDF (download/39403/43364) read as raw text (45 KB); doi 10.1609/aaai.v40i27.39403
- Setup: SGLang with pipeline parallelism; Qwen2-57B-A14B on 8 A40, Qwen3-235B-A22B on 32 A40, DeepSeek-V2 on A100s (16 A100 for disaggregation). Dynamic top-K = Top-P and Ada-K routing from the literature. Baseline = manually tuned parallelism strategies (PP/TP/EP). Datasets CNN-DailyMail, MultiNews (summarization), ShareGPT.
- Numbers:
  - prefill-only throughput: 1.20-3.13x; baseline manually tuned parallelism; ratio as stated; [Abstract, Fig 5a/6]
  - prefill+decode (summarization) throughput: 1.05-1.89x; baseline same; ratio as stated; [Abstract, Fig 5b]
  - disaggregated TTFT average / P99: 1.10-1.17x / 1.21-1.26x; baseline PP=8 baseline; ratio as stated; [Abstract, Fig 8]
  - memory waste in time-balanced PP stages: more than 30% of GPU memory in some stages; baseline -; ratio -; [Introduction]
- Catalog / first-pass numbers:
  - The search result for SMIDT is about dynamic Top-K routing in a serving framework -> yes: correct description
  - catalog row title 'Prefill-stage top-k reduction (paper title not retrieved)' -> wrong: SMIDT is a pipeline-parallel scheduling/partitioning paper; it does not propose reducing top-k in prefill. The ACL-indexed paper the catalog mentions is not identified (candidates from my search, not confirmed as the one meant: ExFold arXiv 2608.24938, MoBiLE 2510.12357)
  - first-pass: prefill-only 1.20-3.13x; disaggregated TTFT 1.10-1.17x -> yes: also omitted: 1.05-1.89x for prefill+decode and P99 TTFT 1.21-1.26x
- Transfers to our regime: no
- Why: Tens of GPUs in pipeline parallelism; speedups are from balancing stages for variable per-layer top-K and expert replication, not from reducing bytes. No accuracy study in this paper (relies on cited Ada-K / Top-P: 'fewer experts at 25% without accuracy loss' quoted from Ada-K).
- Quality loss: Not measured in this paper.
- Ablation: Memory/partition ablations (Fig 10, Table 3); none about top-k reduction itself.
- Stated limitations: No limitation section found in extracted text.
- Versus our findings: No; not evidence either way for prefill top-k cut.

## P-0076  Lynx: Enabling Efficient MoE Inference through Dynamic Batch-Aware Expert Selection (Gupta, Ju, Sinha, Gavrilovska, Iyer; Georgia Tech)

- Read: arXiv:2411.08982v3 (18 May 2026) PDF AND v1 HTML (the catalog URL). The two versions have different headline numbers.
- Setup: v1: Mixtral-8x7B (TP-2) and DBRX-Base (TP-4) on A100-SXM4-80GB x4 with NVLink; decode batch 32 for accuracy tables; latency across batch 8-64, seq 500-2000. v3: Qwen2-57B-A14B, Qwen3-30B-A3B, DeepSeek-Coder-V2, Mixtral-8x7B on H200/A100 with vLLM, chunked prefill, colocated and disaggregated. 'Experts retained' = size of the batch-level expert set per layer, not per-token top-k.
- Numbers:
  - v1 headline: reducing serving latency 1.55x; baseline -; ratio -; [v1 Sec 1]
  - v3 headline: up to 1.30x lower latency; median TPOT 1.09-1.30x across four models (Qwen2-57B 1.25-1.30x, Qwen3-30B 1.14-1.22x, Mixtral 1.09-1.18x); baseline vLLM; ratio as stated; [v3 Abstract, Table 2, Sec 5.2]
  - v1 DBRX batch 64, seq 500: 16->8 experts 1.7x; 16->4 1.9x; baseline all experts; ratio as stated; [v1 Sec 5.2]
  - v1 Mixtral reduce 8->2 at batch 8-16: speedup > 1.5x; half experts 1.3-1.5x; baseline all experts; ratio as stated; [v1 Sec 5.2]
  - v1 Mixtral GSM8K Lynx-Acc: 8: 59.14 (baseline) -> 6: 58.61, 4: 55.27, 2: 43.97; NAEE 4: 37.45, 2: 14.78; baseline baseline 59.14; ratio -3.87 pts at 4 experts; [v1 Table 2]
  - v1 Mixtral HumanEval: 37.80 -> 4 experts 34.76 (Acc) / 30.49 (Lat); NAEE 9.15; baseline 37.80; ratio -3.0 / -7.3; [v1 Table 2]
  - v1 DBRX GSM8K: 71.57 -> 4 experts 62.62 (Acc) / 28.66 (Lat); baseline 71.57; ratio -8.95 / -42.9; [v1 Table 1]
  - v3 batch effect: 1.20x at batch 8, 1.19x at 16, ~1.15x at 32/64; baseline -; ratio -; [v3 Sec 5.8]
- Catalog / first-pass numbers:
  - Up to 1.55x latency reduction -> partly: 1.55x is the v1 claim; the current v3 abstract says up to 1.30x
  - prefill is highly sensitive to expert dropping, decode is resilient -> yes: both versions (v1 Fig 8, v3 Fig 4): figure-only, HumanEval and GSM8K; v3 does not remap in prefill at all
  - first-pass: DBRX batch 64 1.7x (16->8), 1.9x (16->4) -> yes: v1 text; not in v3
  - first-pass: Mixtral 4 experts ~-4% GSM8K -> yes: 59.14 -> 55.27 (Lynx-Acc), -3.87 pts; Lynx-Lat 53.15
  - first-pass: 4x A100, batch 8-64 -> yes: v1; v3 uses other models/GPUs
- Transfers to our regime: partly
- Why: Prefill-sensitivity observation transfers (supports keeping full top-k in prefill). Speedups do not: they come from fewer DISTINCT experts per batch (batch >= 8), memory-bound HBM decode on GPUs; at batch 1 the union equals top-k, nothing to remap. Note: Lynx reduces the set of experts, not per-token top-k.
- Quality loss: v1: -3.9 pts GSM8K Mixtral at 4 of 8 experts (Lynx-Acc); NAEE -21.7; DBRX -9 pts at 4 of 16. v3: accuracy within 1% budget (tables).
- Ablation: Lynx-Lat vs Lynx-Acc policies; phase ablation (prefill vs decode reassignment) in Fig 8 / Fig 4 (values figure-only).
- Stated limitations: v3: benefit only in memory-bound decode batches; compute-bound batches bypass remap; smaller gains with fewer experts (Mixtral).
- Versus our findings: No. Supports our decision to treat prefill top-k reduction as risky. First-pass numbers (1.55x) are superseded by the v3 1.30x.

## P-0077  Alloc-MoE: Budget-Aware Expert Activation Allocation for Efficient Mixture-of-Experts Inference (Liu, Tian, Wang, Zhang, Qiao, Li; NUDT)

- Read: arXiv:2604.08133v1 (9 Apr 2026)
- Setup: DeepSeek-V2-Lite (26 MoE layers, 64 experts, top-6, 2.4B/15.7B), Qwen1.5-MoE-A2.7B (60 experts top-4), OLMoE-1B-7B (64 experts top-8). One H100 80GB, vLLM + lm-eval. Efficiency: dummy random-token prompts, batch 8, prompt length 32, decode 128, avg of 10 runs. Budgets e.g. DeepSeek B in {130,104,78,52} = avg top-K {5,4,3,2}. Baselines: Uniform, LExI, Dynamic-MoE, NAEE.
- Numbers:
  - half budget DeepSeek-V2-Lite prefill speedup: 1.15x; baseline original top-6; ratio 1.15x; [Abstract, Fig 5]
  - half budget decode speedup: 1.34x; baseline original top-6; ratio 1.34x; [Abstract, Fig 5]
  - plain reduction 6 -> 3 experts: 17% performance degradation; nearly 40% at 2 experts; baseline original; ratio -; [Intro, Fig 1a]
  - Alloc-MoE vs Uniform accuracy gain: avg +0.05% NLU, +0.70% Reasoning, +2.15% Math; baseline Uniform allocation; ratio -; [Sec 4.2, Fig 3]
  - Kbase ablation: Kbase=1 best; Kbase>=2 degrades; baseline -; ratio -; [Sec 4.3, Table 2]
- Catalog / first-pass numbers:
  - At half the expert budget on DeepSeek-V2-Lite: 1.15x prefill and 1.34x decode -> yes: matches abstract and Fig 5
  - first-pass: H100, batch 8, prompt 32 -> yes: confirmed (Appendix A.2)
- Transfers to our regime: no
- Why: GPU-resident model, batch 8, 32-token prompts, tiny 64-expert model; speedups are compute reductions, with weights already in HBM. In our regime the byte saving matters, not FLOPs. The allocation idea (per-layer sensitivity DP) is training-free and could lower average k, but quality at 512 experts top-10 is unmeasured.
- Quality loss: At half budget Alloc-MoE keeps 'close to original' (Fig 1b, figure-only); plain halving loses 17%.
- Ablation: Alloc-L vs Alloc-T (Table 4, 7, 8); Kbase (Table 2, 5, 6).
- Stated limitations: Authors: orthogonal to pruning/quantization, no hardware/placement awareness, targets pretrained models.
- Versus our findings: No. The 1.15x prefill / 1.34x decode at half budget is consistent with our Amdahl bound for a top-k cut (1.2-1.4x).

## P-0078  Distribution-Consistent Inference for Dynamic Sparse Mixture-of-Experts (Kim*, Soro*, Hwang; KAIST / DeepAuto.ai)

- Read: arXiv:2609.09241v1 (8 Sep 2026)
- Setup: Qwen3-30B-A3B (128 experts, top-8), kanana-2-30b-a3b-instruct, GLM-4.7-Flash; one RTX PRO 6000 Blackwell Max-Q 96GB, bf16, vLLM for efficiency (ShareGPT, NuminaMath-1.5, concurrent requests). LDA = per-layer per-dimension mean/std alignment of reduced-k outputs to default-k statistics, calibrated offline.
- Numbers:
  - MMLU Qwen3-30B-A3B k=8 baseline: 77.87%; baseline -; ratio -; [Table 2]
  - k=4 without / with LDA: 70.72% / 75.56%; baseline 77.87; ratio -7.15 / -2.31 pts; [Table 2]
  - k=2 without / with LDA: 27.97% / 67.40%; baseline 77.87; ratio -49.9 / -10.5 pts; [Table 2]
  - Table 10 RMS-scaling baseline k=4 / k=2: 75.33% / 65.17% (ExpertCount scaling 24.69 / 24.56); baseline -; ratio -; [Table 10 (app. D)]
  - TPOT ShareGPT k=8 -> 6 -> 4 -> 2: 33.22 -> 30.14 -> 26.00 -> 20.11 ms; throughput 910.7 -> 999.8 -> 1156.3 -> 1477.6 tok/s; baseline k=8; ratio 1.00, 1.10, 1.28, 1.65x TPOT; [Table 4]
- Catalog / first-pass numbers:
  - MMLU 77.87 (k=8) -> yes: Table 2
  - 70.72 (k=4) -> yes: without LDA
  - 27.97 (k=2) -> yes: without LDA
  - a correction method recovers 75.33 at k=4 -> wrong: 75.33 is the simple RMS-scaling baseline (Table 10); the paper's LDA gives 75.56 (Table 2). First-pass noticed the mismatch but did not explain it.
  - first-pass: TPOT 33.22 -> 26.00 -> yes: ShareGPT, saturated concurrent load (910 tok/s aggregate), not batch 1
  - first-pass: EMNLP Findings 2026 -> absent: venue not in the arXiv v1 text; unverified
- Transfers to our regime: partly
- Why: Quality part transfers qualitatively: plain top-k reduction at 128 experts costs 7 MMLU points at k=4 and collapses at k=2, but there is a cheap training-free correction. Speed does not transfer: GPU, loaded server (batch ~30), compute-bound; no byte-stream measurement. Correction requires calibration statistics and an affine op per layer (O(LD)), negligible on CPU.
- Quality loss: MMLU -7.15 (k=4), -49.9 (k=2) plain; -2.31 and -10.5 with LDA. Per-benchmark averages in Table 2 (GK/commonsense 59.52 -> 55.73 / 57.12 at k=4, second group 70.24 -> 60.06 / 67.72).
- Ablation: RMS scaling vs expert-count scaling vs LDA (Table 10); calibration dataset (Table 11) and sample size (Table 12).
- Stated limitations: Cannot recover lost capacity at extreme reduction; ignores covariance; needs calibration stage; (cut-off after four).
- Versus our findings: No. Gives our top-k-cut quality gate a published reference: plain k 8->4 on a 128-expert model costs ~7 MMLU points.

## P-0079  Training-Free Halving of Activated Experts in Fine-Grained Mixture-of-Experts Models (Chen, Yao)

- Read: arXiv:2609.04575v1 (4 Sep 2026)
- Setup: Qwen3.6-35B-A3B (40 layers, 256 experts, top-8, bf16 on a single 98GB accelerator, ~72GB resident) and Qwen3.5-397B-A17B (60 layers, 512 experts, top-10, ~752GB bf16 sharded). HF Transformers, no latency measured. MMLU 2000 questions 5-shot greedy (paired McNemar), GSM8K 500 0-shot, C-Eval 1300. Method: activate top-k1 experts, renormalize by the mass of top-k2 (k2 >= k, e.g. k2 = trained k).
- Numbers:
  - 397B MMLU native k=10: 89.45%; baseline -; ratio -; [Table 5]
  - 397B k1=5: k2=5 / 10 / 20 / 512: 87.35 (-2.10) / 88.90 (-0.55, p=0.24) / 87.75 (-1.70) / 70.90 (-18.55); baseline 89.45; ratio -; [Table 5]
  - 397B k1=8: k2=8 / 10 / 20: 89.60 (+0.15) / 89.00 (-0.45) / 88.45 (-1.00); baseline 89.45; ratio -; [Table 5]
  - 35B MMLU k1=4: k2=4: -4.65 (p=1.7e-9); k2=16: -0.35 (p=0.66); baseline 81.65; baseline 81.65; ratio -; [Intro, Table 5]
  - 397B GSM8K (n=500): native 96.60; k1=8,k2=8 -2.00; k1=8,k2=10 -0.40; k1=5,k2=5 -4.20; k1=5,k2=10 -0.80 (p=0.34); baseline 96.60; ratio -; [Table 6]
  - 35B GSM8K: native 95.00; k1=4,k2=4 -6.60; k1=4,k2=16 -0.60; baseline 95.00; ratio -; [Table 6]
  - 397B C-Eval baseline 88.31: k1=8,k2=8 -2.92; k1=8,k2=10 -1.00; k1=5,k2=10 -0.77 (p=0.20); baseline 88.31; ratio -; [Sec Downstream]
  - perplexity 397B k1=8: k2=8 +1.06/+1.54/+0.42%; k2=10 +0.46/+0.56/+0.34%; k2=512 +128/+146/+56%; baseline native 4.159/3.198/2.249; ratio -; [Table 4]
  - compute: halving k1 halves routed-expert compute (35B: 1.006B -> 0.503B activated routed params/token); memory unchanged; baseline -; ratio 0.5x; [Cost accounting]
- Catalog / first-pass numbers:
  - Halving works only with care -> yes: needs k2 >= trained k reference mass
  - perplexity-best operating point was worse on MMLU -> yes: perplexity minimum is at k2 = trained k; MMLU and PPL favour different k2 (for 35B the MMLU-best k2 is larger)
  - first-pass: 397B MMLU 89.45 -> 87.35 / 88.90 -> yes: Table 5
  - first-pass: GSM8K -4.20 / -0.80; 35B -4.65 / -0.35 -> yes: Table 6, Table 5
  - first-pass: 'hardware not named' -> yes: only 'single 98GB accelerator' and 'sharded across the machine'
  - first-pass: our test is top-4 not 5 -> yes: paper tests k1=5 of 10; top-4 of 10 not tested
- Transfers to our regime: partly
- Why: Quality transfers best of any paper in the batch: 397B-A17B is 512 experts/top-10, 60 layers (our target is 512/top-10, 48 layers, 125B), Training-free, no retraining. Speed unmeasured: they only count FLOPs/bytes; in our NVMe-streaming regime fewer activated experts reduces bytes per token directly (but only if the experts dropped are not cache hits). Caveat: top-4 of 10 (our K=4-of-10 analogue) is not covered; 10->5 loses 0.55 (MMLU) / 0.80 (GSM8K) with k2=10 but is not statistically excluded to be 0.
- Quality loss: 397B 10->5, k2=10: MMLU -0.55, GSM8K -0.80, C-Eval -0.77, PPL +0.4-0.6% at k1=8. Plain renormalization (k2=k1=5): -2.10, -4.20, n/a.
- Ablation: k2 sweep (Tables 4-6), expert identity vs weights, uniform-weight and sampling controls.
- Stated limitations: Both models same series; coarse-grained MoEs benefit little; English/Python corpora; long context untested; no long-form generation; non-significance is not proof of equivalence; geometric k2 grid.
- Versus our findings: Changes a premise: our pubtech top-k analysis (family s1 #3/#16, K=5-of-8 analogue, 'quality gate fails') assumed plain renormalized reduction. This paper shows the loss is largely a gain miscalibration fixed by k2=trained k, so our quality gate must test k2-decoupled renormalization before dismissing top-k cuts. Lynx/Alloc-MoE (GPU) agree on the speed bound, not on quality.

## P-0080  DavidAU: How To Set and Manage MOE Mix of Experts Model Activation of Experts (Hugging Face model page, created 2025-03-03, last modified 2025-07-27)

- Read: HF README.md raw (6 KB) and API JSON; catalog URL truncated (401); working URL is .../How-To-Set-and-Manage-MOE-Mix-of-Experts-Model-Activation-of-Experts
- Setup: Usage guide for merged MOE models (mergekit 2X8/4X8, Mixtral-style, llama arch). Says to start llama-server with --override-kv llama.expert_used_count=int:3, or set 'Experts' in LMStudio, text-generation-webui, KoboldCPP 1.8+, or num_experts_used in the API. No hardware, no benchmarks.
- Numbers:
  - no numbers: -; baseline -; ratio -; [-]
- Catalog / first-pass numbers:
  - --override-kv <arch>.expert_used_count=int:N already reduces active experts globally in llama.cpp -> yes: the page's example key is llama.expert_used_count (arch prefix of the model, e.g. qwen3moe.expert_used_count for others); global, not prefill-only
  - no quality numbers there -> yes: only anecdotal: claims more experts (2-4) give higher quality; 1 expert gives random, varied output
- Transfers to our regime: partly
- Why: Mechanism exists and can be used for the cheap-local test (set N at load; llama.cpp keeps renormalization with the lower k, i.e. the k2=k1 case of P-0079). Evidence on quality or speed: none. Applies to merged MOE models, not tested for fine-grained 512-expert models.
- Quality loss: No data (anecdote that more experts improve quality).
- Ablation: none
- Stated limitations: Marketing-style guide, anecdotal.
- Versus our findings: No.

## Family s1#10 (hot-set / pinned cache vs LRU; per-prompt learned hot set)

Further prior art found (read at abstract/section depth in raw text unless marked):
- SliceMoE (arXiv 2512.12990v5, KAIST, 8 Sep 2026): Predictive Cache Warmup reorders the cache at the prefill-to-decode switch by prefill access frequency; single-batch, Flash -> DRAM, DeepSeek-V2-Lite and Qwen1.5-MoE; warmup gives 2.31x lower decode energy and 1.96x speed vs an empty cache (cache 1.8-3.6 GB). Closest published match to 'use the first W tokens to decide what to keep'. Read in part (mechanism and results text).
- MoE-Infinity (P-0070): per-request activation matrix; ProMoE, DALI, FlashMoE (P-0073) for replacement/prefetch; vLLM RFC 38256 (CPU-pinned experts + GPU cache of hottest, LFRU) found by search, not read.
- llama.cpp thread comments (P-0072): imatrix-based static hot list (lee-b), static hot/cold split with adaptive per-session hot lists (+26% on a concentrated router, +2.3% on gpt-oss), 8 vs 16 GiB slab, QD1 SSD read results.
Not found in my search: a paper that pins a calibrated-from-first-tokens hot set at 2-8 footprints of cache on a CPU-only NVMe-streaming system.

What is new in OUR variant: (a) the regime: CPU-only, 8-32GB RAM, experts from NVMe, 512 experts top-10, cache of 2-8 footprints (<2% of experts) where all published caches are 12-50% of experts on GPUs; (b) testing a static pin built from a short window against warmed LRU at equal capacity. Closest papers: SliceMoE (prefill-hotness warmup), MoE-Infinity (per-request EAM), FlashMoE (SSD, learned replacement). Our own result (pin below LRU at 10% share) is consistent with FlashMoE's Belady gap (86% vs 73% hit) showing little headroom, with MoE-Infinity not being compared to LRU over explicit reads, and with the llama.cpp comments that flat routers give little from better hot lists.

## Family s1#11 (reduce top-k in prefill / globally)

Further prior art:
- ExFold (arXiv 2608.24938, Xiaohongshu/PKU): training-free token-level Top-K folding for prefill, batch-level pool folding for decode; vLLM, up to 1.41x TTFT and 2.45x TPOT, ~99% of quality (abstract read; GPU). Closest to 'prefill top-k reduction'.
- MoBiLE (arXiv 2510.12357, Tsinghua): half the experts for unimportant tokens, full for important, fallback+prefetch; consumer GPU + CPU DRAM, 1.60-1.72x, negligible accuracy loss (abstract read).
- Alloc-MoE (P-0077), Lynx (P-0076), SMIDT (P-0075, scheduling only), LDA (P-0078), P-0079; also Ada-K, NAEE, AdapMoE, LExI, Dynamic-MoE as cited in those papers (not read separately).
- HOBBIT (2411.01433) notes AdapMoE's skipping hurts at small top-k (search snippet, not read).

What is new in OUR variant: a lower top-k as a way to cut BYTES streamed from NVMe on CPU at batch 1 (not FLOPs), combined with the question of how many distinct experts a prefill window touches (window-union). Nobody in the set measures disk bytes; all speedups are GPU compute/HBM. The nearest quality evidence at our architecture is P-0079 (512 experts, top-10: 10->5 costs -0.55 MMLU with decoupled renormalization). What is NOT new: halving k training-free (P-0079, LDA, Alloc-MoE) and phase-specific treatment (Lynx: keep prefill full).

Correction to our own belief: the pubtech note that the top-k cut 'is a quality-for-speed knob only worth keeping if a perplexity gate passes' should add: (1) test k2-decoupled renormalization, (2) do not use perplexity alone (PPL and MMLU prefer different k2). Speed bound unchanged (1.2-1.4x).
