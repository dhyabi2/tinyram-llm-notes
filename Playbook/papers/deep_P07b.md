# Deep re-check P07b (raw full text, no digests): rows P-0153, P-0154, P-0156, P-0157, P-0158, P-0159, P-0161, P-0162, P-0163, P-0164, P-0165

Working folder (scratch): <scratch> All papers downloaded as PDF and converted with pdftotext; nothing executed. Counts: 10 rows read from raw text, 1 row UNREAD (P-0164, OpenReview blocks automated download; abstract only).

## P-0153

**Title / authors / venue:** Building py-kvcache: A Performance Characterization of External KV Caching for vLLM with NVMe SSDs (Joseph Kanichai, Tiziano De Matteis, Animesh Trivedi; Vrije Universiteit Amsterdam / IBM Research Zurich; cs.DC, 10 Sep 2026; work done Jan-Jul 2026; no venue)

**Version read:** arXiv 2609.11744v1 (PDF, full text via pdftotext, 108 kB)  (raw text read: yes)

**Setup (from the paper):** Models: Llama 3.2 3B Instruct (KV 114,688 B/token, 112 KiB) and Qwen3 4B Instruct 2507; batch: 50 concurrent requests in the disk-only test; docs 1k to 80k tokens; FP16 KV, one output token, 256-token KV blocks. Local node: Xeon Silver 4514Y, 256 GiB DDR5, 2x RTX 4000 Ada 20 GiB (PCIe 4.0), Kioxia CM7-R 1.92 TB (13.5 GB/s fio read), Samsung PM9A3 (6 GB/s). Snellius node: 2x EPYC 9334, 768 GiB, 4x H100 94 GiB (PCIe 5.0), Samsung PM1743 (13.5 GB/s). Baselines: LMCache disk backend, vLLM native KV Offload (DRAM + filesystem tier), GPU prefix cache only; vLLM v0.22, io_uring Python connector. Workloads: synthetic long-document, LongBench, SCBench, Alibaba Bailian traces.

**Numbers (with baseline, ratio computed from the paper's values, location):**
- Disk-only TTFT at 80k tokens, py-kvcache (with preload) vs LMCache: 2.0x faster (abstract); Fig 10a text: 2.0-2.5x across sizes; without preload about 1.5x | baseline: LMCache disk backend | ratio: 2.0x (as stated; figure-only values, not recomputed) | where: abstract p1; Sec 7.1 and Fig 10a
- Preload contribution, py-kvcache with vs without preload: 1.34x at 80k, 1.66x at 40k | baseline: same engine without preload | ratio: 1.34x / 1.66x | where: abstract; Sec 7.2 and Sec 7.5
- Tiered (GPU+CPU+disk) vs LMCache: 1.19x / 1.53x / 1.23x at 10k / 40k / 80k tokens | baseline: LMCache | ratio: as stated | where: Sec 7.3, Fig 10b
- Tiered vs native vLLM KV Offload: within 1.10x (40k) and 1.04x (80k); 1.12x faster at 10k | baseline: native vLLM Offload | ratio: as stated | where: Sec 7.3
- SCBench round-robin multi-turn on Snellius: 3.11x vs GPU prefix caching; 2.48x vs native CPU+disk offload; 1.26x vs LMCache | baseline: named | ratio: as stated | where: Sec 7.3 / Fig 11
- Break-even prefix, Llama 3.2 3B, Snellius: 8k prompt needs 77.8% prefix reuse to break even; at 80k only 7.8% | baseline: recompute on GPU | ratio: n/a | where: Sec 5.2, Fig 6
- Break-even storage bandwidth needed for a fully cached prompt: 23.2 GB/s at 1k tokens, 10.4 GB/s at 8k, 3.5 GB/s at 80k (KV 114,688 B/token) | baseline: GPU recompute time | ratio: n/a | where: Sec 5.2, Fig 7
- Bailian traces: Local RTX 4000 Ada: 1.12-1.79x vs GPU prefix caching, 1.10-1.25x vs LMCache. H100: TTFT identical to GPU prefix caching; avg request below 6,203-token SSD break-even for Qwen3 4B | baseline: GPU prefix caching only | ratio: 1.0x on H100 | where: Sec 7.4
- Raw bandwidths measured: fio 13.5 GB/s (CM7-R, PM1743), 6 GB/s (PM9A3); CPU-GPU copy 24-26 GB/s (local), 54-56 GB/s (Snellius) | baseline: n/a | ratio: n/a | where: Sec 4 / Sec 5

**Catalog / first-pass numbers checked:**
- [YES] Disk-loaded prefix KV does not always beat recompute; needs a load-vs-recompute decision -> Exactly the paper's thesis (abstract, Sec 5.2, Sec 7.4).
- [ABSENT] 'For our model, prefill is slow, so loading wins' -> Not in the paper; this is the catalog's own inference. The paper gives the criterion (required GB/s = KV bytes / prefill time). For our target the check is trivial: ~49 MB of KV (30B shape, 500 tokens) over a 19-22 s cold prefill needs about 2.5 MB/s, so loading wins by orders of magnitude (our arithmetic, not the paper's).
- [YES] First-pass: 80k disk load 2.0x vs LMCache, preloading 1.34x, 1.23x tiered vs LMCache, within ~4% of native -> All four confirmed in the text; first pass was abstract-only and is correct.

**Transfers to our regime (CPU-only, 8/16/32 GB, NVMe-streamed experts, batch 1, 125B 512-expert top-10):** partly. The qualitative lesson transfers (admission test: load only if KV bytes / bandwidth < recompute time; for a 69 GB streamed model recompute of even 500 tokens costs 19-22 s so loading always wins). The numbers do not: all GPU servers (H100 / RTX 4000 Ada), vLLM, request-rate workloads, 3-4B dense models, 13.5 GB/s Gen5 SSDs, KV staged through host RAM to a GPU. No CPU-only, no MoE, no batch-1 laptop, no 8-32 GB RAM case.

**Quality loss:** None: exact KV reuse, FP16, no lossy step.

**Ablation relevant to us:** Yes: with/without preload (1.34x/1.66x), disk-only vs tiered, CPU tier adds 1.30x/1.18x/1.08x at 10k/40k/80k (Sec 7.3). Isolates scheduling/overlap, not relevant to a single-user cache.

**Limitations stated by the authors:** Authors: two hardware classes, few models, FP16 KV only, one output token (no sustained-decode effect), single 256-token block size, no networked/RAID storage, CPU-staged I/O path only (GPUDirect could move the break-even), fast-moving vLLM/LMCache versions (Sec 10.1).

**Contradicts / agrees with our findings (Playbook/pubtech):** No. Consistent with Playbook/pubtech/D.md s2_3: our KV 98,304 B/token (30B shape) and 131,072 B/token (small MoE file) bracket the paper's 114,688 B/token for Llama 3.2 3B; load-vs-recompute arithmetic agrees.

## P-0154

**Title / authors / venue:** On 10X Better Scalability: KV Stores Scale Up KV Cache [system SGLang-LSM] (Weiping Yu, Ye Jiarui, He Mengke, Junfeng Liu, Siqiang Luo; Nanyang Technological University; cs.DB, v2 25 Nov 2025; no venue shown)

**Version read:** arXiv 2511.16138v2 (PDF, full text, 57 kB)  (raw text read: yes)

**Setup (from the paper):** Server: Xeon 4314 (16 cores), 64 GB host RAM, NVIDIA A30 24 GB, 8 TB SSD, PCIe 4.0. Models GLM-4-8B (40 KB KV/token), GLM-4-32B (60 KB), Llama-3-8B (paper says 120 KB/token). Synthetic workload: 10 stages x 1000 requests, expected hit rates [0.2,0.3,0.5,0.7,0.5,0.3,0.1,0.3,0.5,0.7]; prompts 4k/8k/16k; warmup of 100 M tokens of KV written through. Baselines: SGLang(file) and SGLang(memory). Design: LSM-tree prefix index with key-value separation (tensor log), adaptive compaction. SSD bandwidth is not stated.

**Numbers (with baseline, ratio computed from the paper's values, location):**
- Hit rate, 4k prompts: 45.4% vs 18.7% | baseline: SGLang(file) | ratio: +26.7 pts = 2.43x (+143%) | where: Sec 4.2, Fig 4 (a)
- TTFT, 16k prompts, GLM-4-8B: 1.78 s vs 2.35 s | baseline: SGLang(file) | ratio: 0.757 (-24.3%) | where: Sec 4.2
- Absolute TTFT saving growing with prompt length: 0.10 s (4k), 0.24 s (8k), 0.57 s (16k) | baseline: SGLang(file) | ratio: n/a | where: Sec 4.2
- TTFT GLM-32B 16k: 8.0 s vs 9.6 s | baseline: SGLang(file) | ratio: 0.833 (-16.7%) | where: Sec 4.3, Fig 5b
- TTFT Llama-3-8B: 0.36 vs 0.41 s (4k); 1.68 vs 1.93 s (16k) | baseline: SGLang(file) | ratio: -12% / -13% (paper: 13%-15%) | where: Sec 4.3, Fig 5
- File backend failure point: write anomalies / degraded reads at about 7 million files | baseline: n/a | ratio: n/a | where: Sec 4.2
- Dynamic compaction: 0.81 s vs 0.85 s mean TTFT (GLM-9B 8k), up to 14% in write-heavy phases | baseline: static compaction | ratio: -5% | where: Sec 4.3, Fig 5c

**Catalog / first-pass numbers checked:**
- [YES] up to 143% more hits -> 45.4% vs 18.7% at 4k prompts (+143%, comparing against the file backend, not against no cache).
- [YES] 24% lower TTFT -> 16k prompts, GLM-4-8B: 1.78 vs 2.35 s; other cases 13-24%.
- [PARTLY] per-token KV for Llama3-8B is about 120 KB -> The paper says 120 KB; exact value from the architecture (32 layers x 2 x 8 KV heads x 128 x 2 B, from memory of the Llama-3-8B config) is 131,072 B, so the paper rounds down by about 8%. Our own measured small-MoE GGUF cache (131,072 B/token) equals the exact figure.

**Transfers to our regime (CPU-only, 8/16/32 GB, NVMe-streamed experts, batch 1, 125B 512-expert top-10):** no. Value is an index/storage layout for millions of cached prefixes in a multi-tenant server (hit-rate under file-count limits). A single-user laptop holds a handful of sessions; the 7-million-file bottleneck, LSM compaction and A30 server setting do not exist there. Gain is over another disk cache, not over recompute.

**Quality loss:** None (exact KV).

**Ablation relevant to us:** Dynamic compaction on/off only (5% mean). No ablation of LSM vs layout vs cache size.

**Limitations stated by the authors:** Authors note write throughput is bounded by the inference engine's own latency; benefits shrink for models with large per-token KV (Llama-3) because the reuse-vs-recompute cost gap narrows (Sec 4.3). Synthetic workload only.

**Contradicts / agrees with our findings (Playbook/pubtech):** No. Supports pubtech D.md note that KV/token is 98-131 KB; the 'KV bytes vs recompute' effect (smaller gain for larger KV) is the same break-even logic as P-0153.

## P-0156

**Title / authors / venue:** ProMoE: Fast MoE-based LLM Serving using Proactive Caching (Xiaoniu Song, Zihang Zhong, Rong Chen, Haibo Chen; Shanghai Jiao Tong University / CAS key lab / Zhejiang University; cs.DC; v1 29 Oct 2024, v3 1 Sep 2025; no venue in the PDF)

**Version read:** arXiv 2410.22134v3 (PDF, full text, 81 kB)  (raw text read: yes)

**Setup (from the paper):** Models (Table 1): Deepseek-moe DS-1 16.4B/2.8B active, 28 layers, 64 experts, top-6; DeepSeek-V2-Lite DS-2 15.7B/2.7B, 27 layers, 64, top-6; Qwen1.5-moe QW-1 14.3B/2.7B, 24 layers, 60, top-4 (all FP16); Qwen2-moe QW-2 57.4B/14.2B, 28 layers, 64, top-8 (INT4); Mixtral-8x7B 46.7B/12.9B, 32 layers, 8, top-2 (INT4). Hardware: i9-14900K, 128 GB host DRAM, RTX 4090 24 GB, PCIe 4.0 (32 GB/s nominal, 23.9 GB/s measured); experts live in host DRAM and are copied to the GPU; attention/non-expert weights on GPU. Batch 1 default (1 to 4 swept), ShareGPT prompts. Frameworks: HF transformers and llama.cpp, both patched; 6,600 lines C++. Baselines: TO (transformers offload), LO (llama.cpp layer offload with CPU compute), UM (unified memory), static cache, LRU cache. Cache rate is a free parameter (50% of experts in the motivating examples). Predictor: per-layer MLP ~2M params, trained offline 1-2 h on one GPU from traces (90/10 split), runs on CPU in ~200 us; stride prefetching (prediction distance +1 layer, accuracy drops only 5%).

**Numbers (with baseline, ratio computed from the paper's values, location):**
- Abstract: average speedup prefill / decode vs existing offloading solutions: 2.20x (up to 3.21x) prefill; 2.07x (up to 5.02x) decode | baseline: TO/UM/LO/static/LRU mix in both frameworks | ratio: as stated; the 2.07x is the pooled average of both frameworks | where: abstract; Sec 1
- vs hand-built caches (static, LRU), both frameworks: 1.78x (up to 2.48x) prefill; 1.34x (up to 1.79x) decode | baseline: static and LRU | ratio: as stated | where: Sec 1
- transformers decode: ProMoE 1.47x (up to 1.77x) vs static, 1.31x (up to 1.46x) vs LRU; 2.47x (up to 5.02x) vs TO/UM | baseline: named | ratio: as stated | where: Sec 7.2, Fig 10
- llama.cpp decode (Fig 11): 1.49x (up to 1.79x) vs static; 1.09x (up to 1.17x) vs LRU; 1.66x (up to 2.07x) vs all baselines incl. LO | baseline: static / LRU / LO | ratio: 1.09x vs LRU | where: Sec 7.2, Fig 11
- llama.cpp, QW-2, cache-rate sweep decode: 1.14x vs LRU, 1.27x vs static (prefill 1.53x / 1.10x) | baseline: LRU, static | ratio: as stated | where: Sec 7.4, Fig 16
- llama.cpp batch sweep, QW-2 decode vs LRU: 1.16x at batch 1, 1.34x at batch 4 | baseline: LRU | ratio: as stated | where: Sec 7.5
- Predictor accuracy: learned 84.7% average (DS-1 and QW-2); token-ID based predictor 58.3%; skip-based 66.9% on QW-2 | baseline: token-based / skip-based | ratio: 1.45x over token-based | where: Sec 4.3-4.4, Fig 8
- Mixtral exception: LO baseline faster than ProMoE in decode (25% of experts active) | baseline: LO | ratio: <1.0x | where: Sec 7.2
- Precision sweep QW-2 llama.cpp prefill vs LRU: 2.06x at INT4 falling to 1.105x at FP16 | baseline: LRU | ratio: as stated | where: Sec 7.6

**Catalog / first-pass numbers checked:**
- [NO] 'Decode speedup averages 2.07x (up to 5.02x) on transformers' -> 2.07x / 5.02x is the abstract's pooled decode figure across transformers and llama.cpp vs offloading baselines. transformers alone: 2.47x avg (up to 5.02x) vs TO/UM, 1.47x vs static, 1.31x vs LRU.
- [PARTLY] 'and 1.66x on llama.cpp' -> 1.66x (up to 2.07x) is true but is an average against all baselines including LO (CPU-compute offload) and static; against LRU it is only 1.09x (max 1.17x) and 1.14x in the cache-rate sweep.
- [PARTLY] First-pass: 'catalog's 1.66x/2.07x labels are swapped relative to text vs abstract' -> Not swapped: 2.07x is the abstract's pooled average, and 2.07x also happens to be the 'up to' of the llama.cpp 1.66x. Neither is a transformers-only average.
- [YES] First-pass: predictor 84.7% (token-based 58.3%), lossless, llama.cpp 1.09x vs LRU / 1.49x vs static -> All confirmed in text.

**Transfers to our regime (CPU-only, 8/16/32 GB, NVMe-streamed experts, batch 1, 125B 512-expert top-10):** no. Breaks: (1) GPU: experts move host DRAM -> 24 GB GPU over PCIe at ~24 GB/s with compute on a 4090, ours is CPU compute with NVMe at ~3 GB/s; (2) cache is 10-70% of experts in GPU memory (50% in the examples) while ours is 2-8 footprints of a 69 GB model; (3) trained per-layer MLP predictors on layer inputs (needs the model, 1-2 h training); (4) models are 14-57B with 64 experts, top-4..8, not 512 experts top-10; (5) the 1.66x/2.07x are averages against weak baselines (LO, static, TO, UM). Vs the only realistic baseline (LRU) in llama.cpp the decode gain is 1.09x.

**Quality loss:** None: lossless, computations identical to the original (Sec 8).

**Ablation relevant to us:** Yes (Fig 12/13): LRU base then +prefetch, +chunked prefetch, +early preemption, +reordered inference. In decode the stack gives 1.35x over LRU in transformers; in prefill naive prefetch alone gives little or negative and reordered inference gives most. Not a cache-vs-layout-vs-prefetch split in our terms.

**Limitations stated by the authors:** Authors: Mixtral (25% expert activation) is better served by CPU compute (LO); low cache rates favour LO; improvement shrinks in llama.cpp because there is less overhead to hide; shrinks at FP16 (bigger experts, smaller cache rate).

**Contradicts / agrees with our findings (Playbook/pubtech):** No contradiction; mild support. Their token-ID-only predictor reaches 58.3% accuracy on 64 experts/top-6 (accuracy definition = share of predicted experts that are correct), comparable to our held-out id-based 0.31-0.43 precision on top-8-of-128. pubtech/D.md says 'ProMoE's 1.66x on llama.cpp ... rely on stronger predictors': more precisely 1.66x is an average dominated by LO/static baselines; vs LRU the full learned system gives 1.09x, close to our 1.00-1.075x.

## P-0157

**Title / authors / venue:** HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference (Peng Tang*, Jiacheng Liu*, Xiaofeng Hou, Yifei Pu, Jing Wang, Pheng-Ann Heng, Chao Li, Minyi Guo; SJTU and CUHK; cs.LG; 6 Nov 2024; conference template 'Conference'17', no venue)

**Version read:** arXiv 2411.01433v2 (PDF, full text, 71 kB)  (raw text read: yes)

**Setup (from the paper):** Models: Mixtral-8x7B (45B total, 14B active, 32 layers, 8 experts, top-2, 87 GB fp16) and Phi-MoE (42B, 6.6B active, 32 layers, 16 experts, top-2). Hardware: RTX 4090 24 GB + 256 GB RAM + 64 cores, PCIe 4.0; NVIDIA Jetson AGX Orin 32 GB unified memory with 12 CPU cores; Samsung 980 PRO NVMe (7 GB/s nominal, about 3 GB/s in practice). Orin runs int8 models (int2 replacement experts), 4090 runs fp16 (int4 replacements). Batch 1; input/output lengths [16,32],[16,128],[128,32],[128,128]; 60 Alpaca samples. Built on llama.cpp (~8k lines). Baselines: Transformers, DeepSpeed-Inference, llama.cpp, MoE-Offloading, MoE-Infinity, Fiddler. Method: token-level dynamic loading (cache-miss expert of low importance is replaced by a low-precision copy), layer-level adaptive prefetching from gate input similarity, multidimensional cache policy (LRU+LFU+LHU+precision).

**Numbers (with baseline, ratio computed from the paper's values, location):**
- Orin decode vs MoE-Infinity: 3.64x (Mixtral), 9.93x (Phi-MoE); prefill latency -60% / -72% | baseline: MoE-Infinity | ratio: as stated | where: Sec 5.2, Fig 14a,b
- Orin decode vs llama.cpp: 13.0x (Mixtral), 18.9x (Phi-MoE); prefill -79% / -80% | baseline: llama.cpp (mmap page faults on unified memory) | ratio: as stated | where: Sec 5.2
- RTX 4090 decode vs MoE-Offloading: 3.21x (Mixtral), 3.29x (Phi-MoE) | baseline: MoE-Offloading | ratio: as stated | where: Sec 5.2, Fig 14c,d
- RTX 4090 decode vs MoE-Infinity: 2.30x (Mixtral), 3.92x (Phi-MoE) | baseline: MoE-Infinity | ratio: as stated | where: Sec 5.2
- 4090 + CPU compute, decode vs llama.cpp / Fiddler: 1.31x / 1.42x vs llama.cpp; 0.99x / 1.46x vs Fiddler | baseline: named | ratio: as stated | where: Sec 5.5, Fig 15
- Ablation: dynamic (mixed-precision) loading alone: 1.19x to 1.57x speedup | baseline: float loading only | ratio: as stated | where: Sec 5.5.1, Fig 16
- Ablation: adaptive prefetching alone (no precision substitution), decode: 1.01x Mixtral; slightly below 1.0x Phi-MoE; with dynamic loading about 1.05x; prefill about -10% | baseline: no prefetch | ratio: ~1.0-1.05x | where: Sec 5.5.2, Fig 17
- Ablation: cache policy: 4.69-8.68% lower miss penalty vs LRU, 2.13-4.19% vs LFU | baseline: LRU / LFU | ratio: n/a | where: Sec 5.5.3, Fig 18
- Next-layer top-1 prediction: 96% average (Mixtral); about 90% for 2-3 layers ahead | baseline: n/a | ratio: n/a | where: Sec 3.2, Fig 7b
- Accuracy Table 3 (GSM8K / TruthfulQA truth,info): Mixtral fp16 0.52, (0.49,0.94) -> +int4 0.51, (0.49,0.93); int8 0.52 -> +int2 0.51 (0.52,0.90 from 0.50,0.91); Phi-MoE fp16 0.83 -> 0.82; int8 0.83 -> 0.83 | baseline: unmixed models | ratio: <=0.01 absolute on GSM8K | where: Sec 5.3, Table 3

**Catalog / first-pass numbers checked:**
- [YES] Up to 9.93x decode speedup -> 9.93x is Phi-MoE on Jetson Orin vs MoE-Infinity. Larger numbers exist in the paper: 13.0x and 18.9x vs llama.cpp on the same Orin (first pass missed these).
- [YES] uses prefetch plus low-precision substitution for misses (lossy) -> Both present; quality change measured <=1% (GSM8K, TruthfulQA).
- [YES] First-pass: 'accuracy drop <=1%' -> Table 3 confirms.
- [YES] First-pass: 'next-layer top-1 prediction ~96% on Mixtral' -> Fig 7b; note this is top-1 of top-2-of-8.
- [NO] Implicit: the 9.93x is mostly from prefetch -> Prefetch alone contributes 1.01x (Mixtral) to about 1.05x; the byte cut from int2/int4 substitution (1.19-1.57x alone) plus baseline weakness (MoE-Infinity and llama.cpp thrash on a 32 GB unified-memory board, 'severe page faults') produce the headline.

**Transfers to our regime (CPU-only, 8/16/32 GB, NVMe-streamed experts, batch 1, 125B 512-expert top-10):** partly. Closest regime of the rows read: llama.cpp base, batch 1, expert files on a ~3 GB/s NVMe, 32 GB board. Breaks: Orin has a GPU doing the compute (unified memory), model is top-2-of-8 (prediction/caching vastly easier than top-10-of-512), baselines are weak (llama.cpp with mmap thrash 13-19x slower), speedup comes from lossy precision substitution (needs low-precision copies of every expert stored too, +~25% storage for int2), the cache holds a large fraction of experts (Orin: 32 GB for an 87 GB int8-ish model).

**Quality loss:** Lossy: Table 3 GSM8K 0.52->0.51 (Mixtral fp16+int4), 0.83->0.82 (Phi-MoE); TruthfulQA truthful/informative within 0.02 (int8+int2 Mixtral: 0.50->0.52 truth, 0.91->0.90 info). Also Fig 3b: replacing <20% experts costs <=1%; skipping 10% costs >1% PPL.

**Ablation relevant to us:** Yes, all three parts separately (Sec 5.5). Key for us: prefetch alone 1.01x in decode, so the lossless prefetch part is about nothing, matching our 1.00-1.075x.

**Limitations stated by the authors:** Authors focus on GPU-centric mode; CPU-assist mode benefits smaller (Sec 5.5); prefetch decode gain small due to misprediction penalty (Sec 3.3 / 5.5.2); highest speedup on slow-transfer devices.

**Contradicts / agrees with our findings (Playbook/pubtech):** No; strong support. HOBBIT's own prefetch ablation (1.01x / ~1.05x) is the lossless-prefetch number and agrees with our 1.00-1.075x. pubtech/D.md says 'HOBBIT's 9.93x rely on stronger predictors / lossy substitution': the paper shows the predictor contributes ~1.0-1.05x, so it is substitution plus weak baselines.

## P-0158

**Title / authors / venue:** Budgeting Bytes: A Windowed Storage Roofline and Dual-Budget Architecture Ablations for Storage-Bound LLM Decoding (Hanhaodi Zhang, single author; cs.AR; 29 Jul 2026; 'draft for workshop submission')

**Version read:** arXiv 2609.04238v1 (PDF, full text, 32 kB)  (raw text read: yes)

**Setup (from the paper):** Part 1: sub-100M-parameter models (23 training runs, 3 scales) comparing address-determinism classes A0-A3. Part 2 (Sec 5.10): Qwen3-30B-A3B (48 layers, 128 experts, top-8; 4-bit 18 GB) on an 8 GB RK3588 ARM board with eMMC (277 MB/s sequential, 237 MB/s at 2.6 MB expert grain) and on an Apple M4 16 GB unified memory. Real prefetch advisor in llama.cpp: decode-time madvise(WILLNEED) over mmap'd expert ranges, driven by a learned per-layer predictor or captured routing. Single-run per configuration. A100 PCIe offload path used for the reconciliation (25.4 GB/s H2D).

**Numbers (with baseline, ratio computed from the paper's values, location):**
- RK3588 8 GB, Qwen3-30B 4-bit, static baseline decode: 0.19 tok/s at 40960 ctx; 0.51 tok/s at ctx 2048 (KV frees page cache) | baseline: n/a | ratio: 2.7x from context shrink alone | where: Sec 5.10
- Temporal-locality prefetch (prev-token experts), RK3588: 0.19 -> 0.12 tok/s at 46.7% top-8 hit rate | baseline: no prefetch | ratio: 0.63x | where: Sec 5.10
- Trace-driven ORACLE prefetch one token ahead, RK3588: 0.12 -> 0.13 tok/s (perfect prediction) | baseline: non-oracle prefetch run | ratio: 1.08x (and ~0.68x of the 0.19 baseline) | where: Sec 5.10
- Blind 2 MB readahead: 0.32 -> 0.14 tok/s | baseline: no readahead | ratio: 0.44x | where: Sec 5.10
- Per-token expert read: about 500-900 MB per token; eMMC at its 277 MB/s ceiling | baseline: n/a | ratio: n/a | where: Sec 5.10
- Sequential-expert-layout optimisation cap: at most 1.24x (237 MB/s scattered vs 247-294 MB/s sequential at 2.6 MB grain) | baseline: scattered reads | ratio: up to 1.24x | where: Sec 5.10
- Fit the fast tier: Q2_K (11 GB) on M4 16 GB, Metal-resident: 11.5 tok/s vs 0.51 (eMMC overflow) | baseline: eMMC overflow baseline | ratio: 22x (11.5/0.51); at 4-bit on M4 Metal 1.2 tok/s | where: Sec 5.10; abstract
- Pre-attention probe on Qwen3-30B-A3B: 91.2% average top-8 hit (14.6x the 6.2% random), per layer 0.84-0.97, trained on 3.6k docs | baseline: random 6.2% | ratio: 14.6x | where: Sec 5.8
- Previous-token expert reuse on the authors' own 24M fine-grained model (8 experts, top-2): 18.8% (28.7% on 4-expert top-1 model); 'at or below random' | baseline: random | ratio: ~random | where: Sec 5.8
- A100 PCIe offload, single stream: prefetch overlap tops out at 1.34x; grows with batch | baseline: no prefetch | ratio: 1.34x | where: Sec 5.10
- Strata planner projection, Qwen3-30B on RK3588: 0.2 -> 2.7 tok/s static offload to prefetch (projected) | baseline: n/a | ratio: 13x projected vs 0.13 measured with an oracle | where: Sec 5.8 Artifact

**Catalog / first-pass numbers checked:**
- [YES] madvise(WILLNEED) expert prefetch design for llama.cpp -> In Sec 5.8 'Artifact' and Sec 5.10: a decode-loop madvise(WILLNEED) advisor compiled into llama.cpp's common library, driven by a learned per-layer predictor (first pass could not verify; now verified).
- [PARTLY] previous-token expert prefetch hits only 18.8% on a top-2-of-8 model -> 18.8% is from the author's own trained 24M-parameter toy MoE (8 experts, top-2), not a published top-2-of-8 model like Mixtral. On the real Qwen3-30B-A3B the same previous-token policy gets 46.7% top-8 hit.
- [YES] First-pass: 91.2% pre-attention probe; prefetch fails even with oracle; 22x by fitting -> All in the abstract and Sec 5.10. 22x is Q2_K on an M4 16 GB with Metal, quality unmeasured (Limitations).
- [YES] First-pass: 'fitting the model gave 11.5 tok/s (22x)' -> 11.5/0.51 = 22.5x; note the baseline is the 0.51 tok/s ctx-2048 run.

**Transfers to our regime (CPU-only, 8/16/32 GB, NVMe-streamed experts, batch 1, 125B 512-expert top-10):** yes. Supports our regime directly: storage-bound, model larger than RAM (ϕ about 0.17 for the 8 GB board; ours 8/69 about 0.12), llama.cpp, mmap, madvise. Breaks: eMMC at 277 MB/s is about 11x slower than our 3 GB/s NVMe (the byte-volume wall is far lower), ARM board, 128 experts top-8 not 512/top-10, the 22x 'fit' lever uses a GPU-resident Q2_K that does not apply when the model is 69 GB.

**Quality loss:** Q2_K fit result: coherence checked, task quality not measured (authors, Limitations). Prefetch is lossless.

**Ablation relevant to us:** Yes, directly relevant: temporal prefetch vs oracle prefetch vs no prefetch vs blind readahead; layout headroom 1.24x. No cache-size sweep.

**Limitations stated by the authors:** Single-run per configuration; sub-100M ablations; roofline compute model single-anchor calibrated; intermediate band (ϕ large but incomplete) argued from end points, not swept; Q2_K quality unmeasured; one A100 path and one edge board.

**Contradicts / agrees with our findings (Playbook/pubtech):** Consistent. Previous-token top-8 hit 46.7% on Qwen3-30B matches our persistence 0.42-0.46 (D.md). Their oracle-prefetch null at saturated bandwidth matches our 'shared disk is the bottleneck' reading (oracle 1.25-1.6x vs 1.9x bound) in direction; caution: our sim is not fully bandwidth-saturated at 3.1 GB/s (disk 188 ms vs compute 219 ms per token) so our oracle still shows 1.25-1.6x, while theirs is fully saturated (0.13 tok/s, 277 MB/s).

## P-0159

**Title / authors / venue:** (a) Pre-Attention Expert Prediction and Prefetching for Mixture-of-Experts Large Language Models (Shien Zhu, Samuel Bohl, Robin Oester, Gustavo Alonso; ETH Zurich Systems Group; cs.CL; 10 Nov 2025; 'under review by a journal'). (b) Speculating Experts Accelerates Inference for Mixture-of-Experts (Vivan Madan*, Prajwal Singhania*, Abhinav Bhatele, Tom Goldstein, Ashwinee Panda; cs.LG; 9 Mar 2026)

**Version read:** arXiv 2511.10676v1 (full text, 57 kB) and arXiv 2603.19289v1 (full text, 43 kB)  (raw text read: yes)

**Setup (from the paper):** (a) Models: DeepSeek-V2-Lite (16.4B/2.4B active, 27 layers, 64 experts, top-6 + 2 shared), Qwen3-30B-A3B (30.5B/3.3B, 48 layers, 128 experts, top-8), Phi-mini-MoE (7.6B/2.4B, 32 layers, 16 experts, top-2). Per-layer 2-linear-layer predictors on the pre-attention (post-LayerNorm) activation of the same layer, trained on 10M MMLU-derived samples, 30 epochs, ranking-aware loss; predictors run on CPU in 0.075-0.129 ms while attention takes 0.73-1.13 ms. Hardware: TITAN RTX 24 GB, 128 GB RAM for accuracy; loading micro-benchmarks on V100-32GB / A100-40GB / A100-80GB (6 DeepSeek-V2-Lite experts = 99 MB). No end-to-end system. (b) Models Qwen3-30B-A3B, GLM-4.7-Flash, GPT-OSS-20B/120B, Qwen3-235B-A22B; A6000 48 GB with 128 GB DRAM over PCIe 4.0, A100 80 GB with 256 GB, GH200 96 GB with 480 GB over NVLink-C2C; batch 1, prompt 1k-64k, 32 output tokens; engine YALIS; expert weights pinned in CPU memory; GLM-4.7-Flash and GPT-OSS-120B speed runs use random expert weights.

**Numbers (with baseline, ratio computed from the paper's values, location):**
- (a) Table 4, first layer only, exact-match of all k experts, Arch 1 / Arch 2: DeepSeek-V2-Lite 93.03% / 92.31%; Qwen3-30B 94.69% / 91.82%; Phi-mini 97.62% / 96.63% | baseline: Fate 78.79% (DeepSeek-V2-Lite) | ratio: +14.2 pts vs Fate | where: Sec 5.2, Table 4 (paper says other layers similar or better)
- (a) over-provisioning (load 10 instead of 6, 12 instead of 8, 3 instead of 2): 98.65% / 98.81% / 99.05% hit; +67% I/O for DeepSeek | baseline: exact k | ratio: n/a | where: Table 4 and Sec 6.2
- (a) top-1 hit: 98.6-99.1% | baseline: n/a | ratio: n/a | where: Table 4, Sec 6.3
- (a) disk-to-GPU load of 6 experts (99 MB): V100 48.1 ms, A100-40 49.8 ms, A100-80 33.5 ms | baseline: memory-to-GPU 9.5 / 8.5 / 4.0 ms | ratio: 5.1x / 5.9x / 8.4x (paper text says '58.4x', inconsistent with its own table) | where: Table 5, Sec 6.1
- (a) expected residual loading time per token: (1-0.9303) x 9.5 = 0.66 ms (V100); paper range 0.27-0.64 ms | baseline: Fate: (1-0.7879) x 9.5 = 2.01 ms (range 0.85-2.01 ms) | ratio: saving 1.37 ms/token vs Fate on V100 | where: Sec 6.2 (analytical, not measured end-to-end)
- (b) Qwen3-30B-A3B TPOT breakdown on-demand: copy 84-88% of TPOT, compute 8-13% | baseline: n/a | ratio: max possible speedup bounded by compute share; 2x if equal | where: Sec 5.1.2, Eq 1
- (b) TPOT reduction with prefetch: Qwen3-30B-A3B 9-14% on A6000 (more at longer prompts); 5-8% elsewhere; abstract 'up to 14%' | baseline: on-demand loading from CPU memory | ratio: 1.10x-1.16x | where: Sec 5.1.3, Fig 8, Conclusion
- (b) task accuracy, speculated experts executed (Router-PF), average over HumanEval, MBPP+, GSM8k, AIME24, AIME25, StrategyQA: Qwen3-30B-A3B 0.817 -> 0.641 (Est-PF 0.750, Hybrid-PF 0.766); GPT-OSS-20B 0.799 -> 0.779; GPT-OSS-120B 0.849 -> 0.857 | baseline: unmodified model | ratio: -17.6 pts for Qwen3-30B Router-PF | where: Table 1
- (b) GSM8k Qwen3-30B-A3B: 0.950 -> 0.576 (Router-PF), 0.918 Est-PF, 0.946 Hybrid | baseline: unmodified | ratio: -37 pts | where: Table 1

**Catalog / first-pass numbers checked:**
- [PARTLY] Pre-attention routers reach 93-98% accuracy by predicting from hidden state -> Table 4 reports FIRST-LAYER accuracy; exact-match 93.03 / 94.69 / 97.62% (Arch 1). Arch 2 is 92.31 / 91.82 / 96.63%. Authors say other layers are similar or better, but only layer-1 numbers are tabulated.
- [PARTLY] (catalog) a much stronger predictor than the expert-id co-occurrence table used here (0.31-0.4x) -> Metrics differ: theirs is exact match of the whole expert set (which for top-8-of-128 is a hard criterion) from a trained per-layer net with MMLU-distribution training data on the same layer's pre-attention state, i.e. a 'no lead beyond attention time' predictor.
- [NO] First-pass: 'no end-to-end speedup, estimated 0.27-0.64 ms/token saved' -> 0.27-0.64 ms/token is the expected REMAINING loading time (misprediction rate x load time); the saving versus Fate is up to 1.37 ms/token (Sec 6.2). No end-to-end measurement exists in paper (a).
- [YES] First-pass: loading 6 experts 33.5-49.8 ms from disk vs 4.0-9.5 ms from memory -> Table 5 (V100 48.1, A100-40 49.8, A100-80 33.5 ms; memory 9.5, 8.5, 4.0).
- [YES] First-pass: (b) up to 14% TPOT reduction -> Qwen3-30B-A3B on A6000; 5-8% on other platforms. But the speculated experts are executed instead of the true ones (lossy for Qwen3, see Table 1).

**Transfers to our regime (CPU-only, 8/16/32 GB, NVMe-streamed experts, batch 1, 125B 512-expert top-10):** no. (a) Prediction window is attention time of one layer (0.73-1.13 ms), a one-layer lead; no system. Models have 16-128 experts, top-2..8 and bf16; disk numbers are from V100/A100 hosts. (b) CPU-to-GPU PCIe/NVLink offload where compute is only 8-13% of TPOT, so the Eq. 1 bound caps the win at that fraction; it executes speculated rather than true experts (quality loss on Qwen3). Neither has NVMe-bound streaming, a 2-8-footprint cache or 512 experts. The pre-attention predictor accuracy (91-98%) is nevertheless a data point that same-layer routing is predictable at one-layer lead, which is the quantity our oracle sim (1.25x top-4 / 1.59x top-8) bounds.

**Quality loss:** (a) none (prediction only). (b) Router-PF: Qwen3-30B-A3B average 0.817 -> 0.641, GSM8k 0.950 -> 0.576; GPT-OSS models within ~2 pts; estimator variants recover to 0.75-0.77.

**Ablation relevant to us:** (a) loss-function and architecture comparison (Table 3), not a prefetch-vs-cache split. (b) Router-PF vs Est-PF vs Hybrid-PF (Table 1) and context-length sweep of compute vs copy (Fig 7/8).

**Limitations stated by the authors:** (a) Authors list future work: end-to-end integration with caching, larger expert counts; accuracy given for layer 1. (b) bound min(copy, compute); gains shrink with fast links (5-8%); quality drops on Qwen3; no KV-cache offload support so contexts limited; random expert weights for two models.

**Contradicts / agrees with our findings (Playbook/pubtech):** No. (a) is consistent with D.md's 'Unmeasured: pre-attention predictors'; their disk-vs-memory table (5-8x) says nothing about NVMe on a Mac. (b)'s Eq. 1 (gain <= min(copy, compute)) is the same overlap bound as our ~1.9x/1.09x numbers; its measured 9-14% when compute is 8-13% of TPOT matches 'prefetch hides at most the compute share'.

## P-0161

**Title / authors / venue:** Delta Decompression for MoE-based LLMs Compression [D2-MoE] (Hao Gu*, Wei Li*, Lujun Li*, Qiyuan Zhu, Mark Lee, Shengjie Sun, Wei Xue, Yike Guo; HKUST / Birmingham / others; cs.LG; 24 Feb 2025; ICML 2025 per proceedings listing (PDF itself shows no venue))

**Version read:** arXiv 2502.17298v1 (PDF, full text, 64 kB)  (raw text read: yes)

**Setup (from the paper):** Models: Mixtral-8x7B, DeepSeekMoE-16B-Base, Phi-3.5-MoE, Qwen2-57B-A14B. Training-free: Fisher-weighted merge of experts into a shared base, truncation-aware SVD of per-expert deltas, semi-dynamic structured pruning of the base. Calibration: 512 random WikiText-2 samples (compression time e.g. 19 min for the scale matrix on Mixtral). NVIDIA A100s. Evaluation: perplexity on WikiText-2/PTB/C4 and 7 zero-shot tasks. Throughput test: Mixtral-8x7B, batch size 64, compression 60/70/80%. Baselines: NAEE, MoE-I2, LoSparse, MC-SMoE, SparseGPT, ASVD.

**Numbers (with baseline, ratio computed from the paper's values, location):**
- Mixtral-8x7B WikiText-2 PPL: 3.98 (orig) -> 4.65 / 5.28 / 6.46 at 20/40/60% compression | baseline: original | ratio: 1.17x / 1.33x / 1.62x PPL | where: Table 2
- Mixtral average zero-shot accuracy: 0.63 -> 0.60 / 0.57 / 0.52 | baseline: original | ratio: -0.03 / -0.06 / -0.11 | where: Table 2
- DeepSeekMoE-16B-Base: WikiText-2 6.38 -> 6.84 / 7.93 / 11.67; avg acc 0.56 -> 0.54 / 0.49 / 0.41 | baseline: original | ratio: 60%: PPL 1.83x, -0.15 acc | where: Table 2
- Phi-3.5-MoE and Qwen2-57B at 40%: PPL 3.48 -> 6.07, avg 0.62 -> 0.60; PPL 5.12 -> 8.19, avg 0.59 -> 0.58 | baseline: original | ratio: n/a | where: Table 2
- Mixtral throughput at batch 64 (60% / 70% / 80%): 277.72 / 300.33 / 313.29 tok/s; TFLOPs 481 / 386 / 290; size 18.68B / 14.01B / 9.33B; memory 34.8 / 26.1 / 17.3 GB | baseline: NAEE 271.89 / 272.66 / 278.53; MoE-I2 227.60 / 252.55 / 294.04; LoSparse 158.9 / 191.5 / 198.0 | ratio: 1.02x / 1.10x / 1.12x vs NAEE; no uncompressed row | where: Table 4, App A.3
- Ablation, merging the base: WikiText-2 PPL Fisher 5.28 vs frequency 6.42, mean 7.66, TIES 12.45, RegMean 187 | baseline: Table 5 (40%) | ratio: n/a | where: Table 5

**Catalog / first-pass numbers checked:**
- [YES] Shared base from Fisher-weighted merging plus SVD-compressed per-expert delta, pruning of the base -> Abstract and Sec 3.
- [YES] 40-60% compression on Mixtral, Phi-3.5, DeepSeek and (Qwen2) -> Abstract: 40-60%; Table 2 also has 20%.
- [YES] First-pass: Mixtral PPL 3.98 -> 4.65 / 5.28 / 6.46; avg acc 0.63 -> 0.60 / 0.57 / 0.52; throughput 277.7 vs NAEE 271.9 at batch 64 -> All confirmed in Tables 2 and 4. 'No uncompressed throughput given' also confirmed.
- [NO] Implicit: the speed gain is a decode speedup -> Throughput of 270-313 tok/s is at batch 64 on A100, compute-bound; the 'speed' reflects fewer FLOPs, not fewer streamed bytes.

**Transfers to our regime (CPU-only, 8/16/32 GB, NVMe-streamed experts, batch 1, 125B 512-expert top-10):** no. Gains are model-size cuts of 20-80% by parameter dropping (lossy), measured on a GPU at batch 64. The shared-base idea is the one we want, but (i) quality: Mixtral at 40% already loses 0.06 average accuracy and +33% PPL, at 60% 0.11 and +62%; (ii) it reduces stored parameters, so bytes per streamed expert fall by at most 1/(1-ratio) (1.25-2.5x for 20-60%), consistent with our 1.3-2.5x closed-form; (iii) fp16 baseline (not 4.3 bpw), so the real byte saving vs a 4-bit model is not shown; (iv) base weights stay resident, delta SVD factors add compute; (v) no offloading experiment.

**Quality loss:** Yes, see numbers: Mixtral avg acc -0.03/-0.06/-0.11 at 20/40/60%; DeepSeekMoE-16B 60% avg acc 0.41 vs 0.56.

**Ablation relevant to us:** Yes: merge method (Table 5), base/delta compressors (Table 6), compression-ratio split between base and delta (Table 7), calibration data (Tables 8, 9). Not isolating bytes vs compute.

**Limitations stated by the authors:** Authors: decomposition and pruning steps have some complexity (Limitations); no stated discussion of offloading.

**Contradicts / agrees with our findings (Playbook/pubtech):** No. Consistent with D.md s2_5 (published methods achieve 40-60% only with measurable loss). One caveat in D.md: it cites D2-MoE as 'consistent with needing fitted merging' for our OLMoE flat-spectrum probe; the paper provides no spectrum analysis of the delta, only the working-method results.

## P-0162

**Title / authors / venue:** MoBE: Mixture-of-Basis-Experts for Compressing MoE-based LLMs (Xiaodong Chen, Mingming Ha, Zhenzhong Lan, Jing Zhang, Jianguo Li; Inclusion AI / Ant Group, Renmin University, Westlake University; cs.LG; 7 Aug 2025)

**Version read:** arXiv 2508.05257v1 (PDF, full text, 51 kB)  (raw text read: yes)

**Setup (from the paper):** Models: Ling-Lite-Chat, DeepSeek-V2-Lite-Chat, Qwen3-30B-A3B-2507, Qwen3-235B-A22B-2507, DeepSeek-V3-0324 (671B), Kimi-K2-Instruct (1T; 384 experts, trained in two groups of 64 bases). Each up/gate expert matrix W = A B with B re-parameterised as a SiLU/tanh of a linear combination of m basis matrices shared across the layer (m = 4 for the small models, 32 / 64 / 128 for large); down projections untouched; fit by Adam (lr 0.07) on reconstruction error, data-free (no calibration), on H100s; no distillation. Variant MoBE-dagger reduces active experts k->k' to hold active parameters constant. Baselines: D2-MoE (not run on the 3 largest models, too costly), MoLAE. Benchmarks: 15 (ARC-C, IFEval, GPQA, BBH, MMLU, CEval, CMMLU, Math, GSM8k, AIME24/25, MBPP, HumanEval, MultiPL-E, LCB).

**Numbers (with baseline, ratio computed from the paper's values, location):**
- Qwen3-30B-A3B-2507, 24% params cut, Table 3 average: 78.6 -> 75.8 (MoBE), 75.7 (MoBE-dagger); D2-MoE 66.4, MoLAE 67.5 | baseline: original 78.6 | ratio: -2.8 pts (96.4% retained) | where: Table 3
- Qwen3-235B-A22B-2507, 24%: 81.5 -> 80.9 (MoBE), 79.7 (dagger); MoLAE 73.2 | baseline: original | ratio: -0.6 pts (99.3%) | where: Table 3
- DeepSeek-V3-0324, 30%: 79.3 -> 78.0 (MoBE), 77.9 (dagger); MoLAE 73.1 | baseline: original | ratio: -1.3 pts | where: Table 3
- Kimi-K2-Instruct, 24%: 82.4 -> 81.1 (MoBE), 80.8 (dagger); MoLAE 76.6 | baseline: original | ratio: -1.3 pts | where: Table 3
- Small models: Ling-Lite 16% cut: 59.7 -> 58.6; DeepSeek-V2-Lite 15%: 38.0 -> 36.9 | baseline: original | ratio: -1.1 pts each | where: Table 3
- Hard benchmarks, Qwen3-30B-A3B 24%: AIME24 59.4 -> 55.0; AIME25 51.3 -> 45.2; LCB 41.5 -> 35.6; MultiPL-E 70.6 -> 61.9 | baseline: original | ratio: -4.4 / -6.1 / -5.9 / -8.7 pts | where: Table 3
- Average over models: authors: about 1.4% accuracy lost converting to MoBE, 0.5% for dagger | baseline: original | ratio: n/a | where: Sec 4.3
- Effective rank of expert matrices (Kimi-K2, 7168x2048): mean effective rank > 1593 in most layers, above the SVD break-even threshold 1593 | baseline: SVD break-even | ratio: n/a | where: Appendix B, Fig 9-11
- Speed: none reported; custom mega-kernel needed | baseline: n/a | ratio: n/a | where: Limitations

**Catalog / first-pass numbers checked:**
- [YES] Parameters cut 24-30% on Qwen3-235B, DeepSeek-V3 and Kimi-K2 -> Table 3: 24% (Qwen3-235B), 30% (DS-V3), 24% (Kimi-K2).
- [PARTLY] 1-2% accuracy drop -> Absolute drops on the three large models are 0.6 / 1.3 / 1.3 points (abstract: '1%-2%'); but Qwen3-30B-A3B at 24% loses 2.8 points and individual hard benchmarks lose 4-9 points (AIME, LCB, MultiPL-E).
- [YES] First-pass: 'Qwen3-30B 24% -> 78.6 to 75.8' -> Confirmed.
- [YES] First-pass: no speed numbers; needs custom kernel; extra active params -> Authors' Limitations; MoBE-dagger cuts k to compensate.
- [YES] First-pass: only gate/up compressed; trained optimisation -> Sec 3; Adam fit of reconstruction error (data-free, not 'training' of the LLM).

**Transfers to our regime (CPU-only, 8/16/32 GB, NVMe-streamed experts, batch 1, 125B 512-expert top-10):** no. Reduces total params 24-30% (1.3-1.4x bytes) at bf16 on H100; no streaming; extra per-expert factors can raise active parameters; quality cost 0.6-2.8 average points, much worse on hard reasoning. The useful finding for us is negative: shared basis matrices lose information because expert matrices have effective rank above the SVD break-even, i.e. independent experts, which matches our flat-spectrum probe.

**Quality loss:** Yes (Table 3): -0.6 to -2.8 average points; -4 to -9 on AIME/LCB/MultiPL-E for Qwen3-30B-A3B.

**Ablation relevant to us:** Yes: activation function (SiLU/tanh best, ReLU worse, Fig 6), Z-score normalisation (Fig 7). No cache/bytes ablation.

**Limitations stated by the authors:** Authors: slight accuracy drop remains (suggest full-network KD); MoBE needs several fused-MoE kernel calls, so a dedicated mega-kernel is required to realise speed.

**Contradicts / agrees with our findings (Playbook/pubtech):** No. Supports pubtech/D.md: 24-30% cut (1.3-1.4x bytes) vs the 3.6x claimed by the family; effective-rank argument agrees with our OLMoE probe (16 of 64 components = 27.8% of centred energy vs 25% chance).

## P-0163

**Title / authors / venue:** MoE-SVD: Structured Mixture-of-Experts LLMs Compression via Singular Value Decomposition (Wei Li*, Lujun Li*, Hao Gu, You-Liang Huang, Mark Lee, Shengjie Sun, Wei Xue, Yike Guo; PMLR 267 (ICML 2025), pp 35209-35230)

**Version read:** ICML 2025 camera-ready PDF (proceedings.mlr.press/v267/li25az; no arXiv id found), full text, 87 kB. First pass read only the poster abstract.  (raw text read: yes)

**Setup (from the paper):** Models: Mixtral-8x7B and 8x22B, Phi-3.5-MoE, DeepSeekMoE-16B, Qwen2. Training-free: sensitivity-based selective decomposition of expert layers, then one shared V matrix across all experts and top-k selected U matrices; optional LoRA fine-tune (MoE-SVD-dagger). Calibration 512 samples. Runtime = throughput on a single NVIDIA H800 (batch size and sequence length 32 mentioned for the speed figures). Baselines: ASVD, SVD-LLM, MoE-I2, MC-SMoE, MoE-Compression, Wanda/SparseGPT 2:4, LoSparse.

**Numbers (with baseline, ratio computed from the paper's values, location):**
- Mixtral-8x7B runtime (tok/s) vs ratio: 87.7 dense; 104.7 (1.2x) at 20%; 109.8 (1.3x) at 40%; 123.9 (1.4x) at 50%; 156.1 (1.8x) at 60% | baseline: dense Mixtral on one H800 | ratio: 1.19x / 1.25x / 1.41x / 1.78x | where: Table 2; Table 11
- Mixtral average zero-shot accuracy: 0.63 -> 0.58 (20%) / 0.50 (40%) / 0.43 (50%) / 0.37 (60%); with LoRA 0.60 at 20%, 0.49 at 50% | baseline: original 0.63 | ratio: -0.05 / -0.13 / -0.20 / -0.26 | where: Table 2
- Mixtral WikiText-2 PPL: 3.98 -> 4.86 / 6.74 / 9.03 / 13.52 | baseline: original | ratio: 1.22x / 1.69x / 2.27x / 3.40x | where: Table 2
- Phi-3.5-MoE: runtime 98.2 -> 108.6 (20%) ... 148.7 (60%, 1.5x); avg acc 0.62 -> 0.60 (20%) / 0.56 (40%) / 0.53 / 0.49 (60%) | baseline: dense | ratio: 1.51x at 60% | where: Table 2
- Model size at 60% (Mixtral): 46.7B -> 17.6B; TFLOPs 5.27e14 -> 3.67e14 | baseline: dense | ratio: 0.38x size, 0.70x FLOPs | where: Table 11

**Catalog / first-pass numbers checked:**
- [YES] One V matrix shared across experts, U matrices trimmed -> Abstract/Sec 3.
- [PARTLY] 60% compression with faster inference -> Speedup at 60% is 1.78x on Mixtral and 1.51x on Phi-3.5 (abstract says 1.5x), but Mixtral at 60% has average accuracy 0.37 vs 0.63 and WikiText PPL 13.52 vs 3.98, so the abstract's 'minimal performance loss' does not hold at 60% in its own Table 2. The 1.2x speedup pairs with 20% compression (acc 0.58).
- [YES] First-pass: 60% compression, 1.5x faster, training-free (abstract) -> Matches abstract; first-pass caveat 'hardware and tables unverified' now resolved: single H800.

**Transfers to our regime (CPU-only, 8/16/32 GB, NVMe-streamed experts, batch 1, 125B 512-expert top-10):** no. GPU single-device throughput gain from fewer FLOPs/bytes of an in-memory model; the speedup (1.2-1.8x) comes with large quality loss at the ratios that give it. Streaming from NVMe would at best see the bytes ratio (60% cut = 2.65x fewer params by Table 11) but accuracy 0.37 is unusable. Shared V across experts is a different sharing axis (input-side) from the shared-basis-across-experts idea we tested, but gives no spectrum evidence either.

**Quality loss:** Yes (Table 2): Mixtral avg acc -0.05 at 20%, -0.13 at 40%, -0.26 at 60%; LoRA recovers only ~0.02 at 20%.

**Ablation relevant to us:** Yes (App B/C/D): selective decomposition, V sharing, top-k U, LoRA, significance test over 3 runs (Table 12). No bytes-vs-compute split.

**Limitations stated by the authors:** Authors: SVD methods add communication overhead under tensor parallelism (Limitations; App B.4).

**Contradicts / agrees with our findings (Playbook/pubtech):** Corrects pubtech/D.md only in that the row is no longer 'unmeasured beyond the spectrum probe': the published Table 2 shows large losses at 50-60% compression. No conflict with our results.

## P-0164

**Title / authors / venue:** DeltaMoE: Memory-Efficient Inference for Merged Mixture of Experts with Delta Compression (Boyko Borisov, Xiaozhe Yao, Nezihe Merve Gurel, Ana Klimovic; ICLR 2025 Workshop on Sparsity in LLMs (SLLM))

**Version read:** None read in full. Abstract only, from mlanthology.org/iclrw/2025/borisov2025iclrw-deltamoe and iclr.cc/virtual/2025/33539 (via WebSearch/curl). No arXiv version found (arXiv API title search returned 0 results).  (raw text read: NO)

**Setup (from the paper):** UNREAD. Abstract: training-free delta-compression pipeline for sparse MoEs built by merging fine-tuned models, using structured sparsity and quantization; claims up to 2.34x compression and 2.57x throughput, scalable with expert count. Models, hardware, baselines, batch size not obtained.

**Numbers (with baseline, ratio computed from the paper's values, location):**
- Abstract claims: up to 2.34x compression ratio, up to 2.57x throughput improvement | baseline: not stated in abstract | ratio: not computable | where: abstract only

**Catalog / first-pass numbers checked:**
- [YES] Training-free delta compression with structured sparsity and quantization -> Matches the abstract (not verified against the paper body).
- [YES] 2.34x compression, 2.57x throughput -> Matches the abstract; setting unverifiable.

**Transfers to our regime (CPU-only, 8/16/32 GB, NVMe-streamed experts, batch 1, 125B 512-expert top-10):** no. Cannot be checked from raw text. From the abstract alone: the MoE is a merge of fine-tuned models (deltas against a common base are small by construction), unlike a pretrained MoE whose experts are independent; so the transfer judgement 'no' is an inference from the abstract, not a reading.

**Quality loss:** Unknown (not in abstract).

**Ablation relevant to us:** Unknown.

**Limitations stated by the authors:** Unknown.

**Contradicts / agrees with our findings (Playbook/pubtech):** Unknown.

**UNREAD reason:** Full text only on OpenReview (openreview.net/pdf?id=FJ7Z8H6elV); the server returns an HTML 'ChallengeRequired' page (HTTP 403 verification challenge) to curl and the API, so the PDF could not be fetched; no arXiv, GitHub or proceedings copy found. Marked UNREAD; everything beyond the abstract is not filled from memory.

## P-0165

**Title / authors / venue:** Bandwidth-Efficient Adaptive Mixture-of-Experts via Low-Rank Compensation (Zhenyu Liu*, Yunzhen Liu*, Zehao Fan, Garrett Gagnon, Yayue Hou, Nan Wu, Yangwook Kang, Liu Liu; Rensselaer Polytechnic Institute, UMass Amherst, George Washington Univ., Samsung Semiconductor; arXiv Dec 2025 (v1 18 Dec 2025 per first pass); no venue in PDF)

**Version read:** arXiv 2512.17073 (PDF, full text, 48 kB)  (raw text read: yes)

**Setup (from the paper):** Models: Mixtral-8x7B (32 layers, 8 experts, top-2, 45.1B expert params), Mixtral-8x22B (56 layers, 8, top-2, 135.5B), DeepSeek-MoE-16B (28 layers, 64 experts, top-6 + 2 shared). Method: quantise all experts to INT2/INT3 (HQQ-style), precompute per-expert low-rank (SVD) compensators of the quantisation residual with kurtosis-guided rank allocation (average rank budget 32 Mixtral / 64 DeepSeek, quantised to INT3); at inference transfer low-bit experts plus compensators only for the top-n experts per token (n=1 for Mixtral, 3 for DeepSeek) and keep others low-bit. Evaluation: 7 zero-shot tasks + MMLU 5-shot + WikiText PPL; systems: H100 80 GB + DDR over PCIe (GPU-only offloading) and a simulated GPU+near-data-processing device (512 GB/s) using Ramulator and the MoNDE configuration. Input length 256, outputs 128/256/512; batch size not stated; whether the GPU-only throughput is measured or modelled is not stated. Profiling in Sec 2: Mixtral-Offloading on an H100 PCIe + EPYC 9334, FP16 experts, shows time dominated by host-device movement.

**Numbers (with baseline, ratio computed from the paper's values, location):**
- Mixtral-8x7B GPU-only offloading, 256/512 tokens: 2.37 tok/s (Mixtral-Offloading, FP16 experts) -> 12.27 (3-bit) / 18.11 (2-bit) | baseline: Mixtral-Offloading FP16 | ratio: 5.18x / 7.64x (paper: 5.17x / 7.64x) | where: Sec 4.3, Fig 7
- Mixtral-8x7B vs HOBBIT: 6.75 -> 25.43 (3-bit) / 36.97 (2-bit) | baseline: HOBBIT | ratio: 3.77x / 5.48x | where: Sec 4.3
- Mixtral-8x22B: 0.79 -> 4.1 / 6.05 tok/s (256/512) | baseline: Mixtral-Offloading | ratio: 5.18x / 7.63x | where: Sec 4.3
- DeepSeek-MoE-16B: 4.38x-5.93x vs Mixtral-Offloading, 2.95x-3.78x vs HOBBIT | baseline: named | ratio: as stated | where: Sec 4.3
- GPU+NDP (simulated) Mixtral-8x7B: 11.56 -> 54.96 (3-bit) / 77.33 (2-bit) tok/s | baseline: MoNDE | ratio: 4.75x / 6.69x | where: Sec 4.3
- Uniform GPTQ INT2 average accuracy: Mixtral-8x7B 70.03% -> 34.41%; Mixtral-8x22B 72.62% -> 37.10%; DeepSeek-MoE-16B 59.63% -> 38.50% | baseline: FP16 | ratio: -35.6 / -35.5 / -21.1 pts | where: Sec 4.2, Fig 6
- Method vs HQQ (same bit-width), Mixtral-8x7B: INT3 avg +0.45 pts, INT2 avg +8.24 pts (MMLU +1.11 / +10.50) | baseline: HQQ | ratio: n/a | where: Sec 4.2
- Method vs HQQ, DeepSeek-MoE-16B: INT3 avg +1.63, INT2 avg +2.30 (MMLU +1.83 / +1.73) | baseline: HQQ | ratio: n/a | where: Sec 4.2
- Restoring only top-1 expert at INT2, Mixtral-8x7B: MMLU 47.53%, WikiText PPL 6.32; top-2 only: 25.26%, 16.76 | baseline: n/a | ratio: n/a | where: Table 2 / Sec 4.4
- DeepSeek-MoE-16B INT2, restoring top-3 experts: MMLU 28.75% (random is 25%), WikiText PPL 10.99; top-4-6: 25.80%, 16.03 | baseline: n/a | ratio: near chance | where: Table 2 / Sec 4.4
- Compensator size: rank 16 = 0.32 MB (0.75% of one INT2 expert); rank 128 = 2.53 MB (6.03%); PPL falls 7.05 -> 3.77 from rank 16 to 128 (Mixtral INT2 ablation) | baseline: INT2 expert | ratio: n/a | where: Sec 4.4 (2)

**Catalog / first-pass numbers checked:**
- [YES] For offloaded MoE, transfers dominate runtime -> Fig 1(a): host-device movement dominates (FP16 Mixtral-Offloading on H100).
- [PARTLY] A low-rank compensator offsets low-bit expert quantization error, so a low-bit base is streamed and a small correction (cut off) -> Per-expert compensator for the top-n router-ranked experts only (n=1 Mixtral, 3 DeepSeek); the other experts stay plain INT2/INT3.
- [YES] First-pass: 2.37 -> 12.27 / 18.11; vs HOBBIT 6.75 -> 25.43 / 36.97; GPTQ INT2 70.03% -> 34.41%; rank-16 = 0.32 MB (0.75%) -> All confirmed.
- [YES] First-pass: 'batch size unspecified' -> Still not stated; only input/output lengths.

**Transfers to our regime (CPU-only, 8/16/32 GB, NVMe-streamed experts, batch 1, 125B 512-expert top-10):** partly. Relevant to our 'low-bit base + small residual for the few most important experts' (family 9), not to shared-basis (family 5): compensators are per-expert and the base is plain low-bit. The speedups (5-8x) are dominated by FP16 -> 2-3-bit byte reduction against an FP16 PCIe-offload baseline and a HOBBIT baseline; vs a 4-bit streamed baseline the byte gain would be ~1.3-2x. GPU/H100, 8-64 experts, top-2..6; performance partly simulated. Critically, the fine-grained model (DeepSeek-MoE-16B, 64 experts, top-6; closest to our 512/top-10) is NOT rescued: INT2 + top-3 compensation gives MMLU 28.75% (chance 25%) and PPL 10.99; the benefit relies on Mixtral's skewed router (top-1 dominates).

**Quality loss:** INT2 uniform: -21 to -36 pts average accuracy. With method: Mixtral-8x7B INT2 MMLU 47.53% (top-1 restored; FP16 reference MMLU not read in text, figure only); DeepSeek-MoE-16B INT2 MMLU 28.75%; INT3 within ~0.5-1.6 pts of HQQ INT3.

**Ablation relevant to us:** Yes (Fig 8, Table 2): number/position of restored experts, rank budget, kurtosis-guided vs uniform rank (PPL 7.05 vs 7.55 at rank 16). Isolates the compensation part, relevant to family 9.

**Limitations stated by the authors:** Authors: DeepSeek needs higher-rank compensators or more restored experts (smaller gains); compensator transfer assumed negligible; future work on rank allocation and adaptive top-n.

**Contradicts / agrees with our findings (Playbook/pubtech):** No. Consistent with D.md s2_5 closed form (byte cut with overlap 1.0x at 16/32 GB) and relevant to F.md family 25 (2-bit base + refinement): evidence that a 2-bit base needs a skewed router to be recoverable. The 'shared basis' reading in the P07 first pass catalog row is right: this paper is per-expert.

# Per-family sections

## Family screen 2 #3 (persist KV of a finished prompt): rows P-0153, P-0154

Prior art found (read at the depth stated):
- Prompt Cache: Modular Attention Reuse for Low-Latency Inference (Gim, Chen, Lee, Sarda, Khandelwal, Zhong; arXiv 2311.04934v2, 25 Apr 2024). Raw text downloaded; read abstract/intro/headline only: TTFT 8x on GPU and up to 60x on CPU (abstract), 20-70x CPU in intro; exact reuse of modules, no model change. Same lever, CPU result is the closest regime to ours; weakens novelty of any KV-reuse claim.
- DUAL-BLADE: Dual-Path NVMe-Direct KV-Cache Offloading for Edge LLM Inference (Jeong et al., arXiv 2604.26557, ICDCS 2026). Raw text downloaded (69 kB) and skimmed (OPT-6.7B, edge GPU, NVMe-direct vs page cache; prefill -33.1%, decode -8.2..-42.4% over baselines); not read at table depth. Shows page-cache thrash is a known issue for NVMe-resident state under memory pressure.
- Search-snippet only (not opened this pass): LMCache (MLSys 2026 talk), Mooncake, llm-d, CacheBlend (2405.16444), A3 (2511.17560), llama.cpp --prompt-cache / slot save-restore (P-0148/0151, first pass), hearth-llm (P-0149).
- Memory (labelled from memory): vLLM automatic prefix caching, SGLang RadixAttention/HiCache, AttentionStore/CachedAttention (ATC 2024).

What is new in OUR variant: nothing in the mechanism (exact KV persistence is shipped in stock llama.cpp). The only specific element is the regime arithmetic: cold prefill for a streamed 69 GB model costs 19-22 s of expert I/O/compute against 0.4 s KV load, so the break-even test of py-kvcache (Sec 5.2) is always passed for any reuse of a few hundred tokens. Closest papers: py-kvcache (break-even criterion), Prompt Cache (CPU TTFT 60x). Neither covers MoE streamed from disk, and none changes decode tok/s.

Verdict: novelty = known; published evidence supports TTFT only; our pubtech/D.md s2_3 numbers (decode exactly 1.0x, request 1.24x) stand. Corrections to D.md: none needed. P-0154's '120 KB per token for Llama-3-8B' is a rounding of 131,072 B.

## Family screen 2 #4 (cross-layer expert prefetch): rows P-0156, P-0157, P-0158, P-0159

Prior art found and read in raw text (beyond the four rows):
- Fate: Fast Edge Inference of MoE via Cross-Layer Gate (Fang, Hong, Huang, Lyu, Chen, Yu, Yu; arXiv 2502.12224v2, 7 May 2025; read intro/setup): RTX 3090 24 GB + i9-9900X + 128 GB host, PCIe 3.0 x16 (16 GB/s) and a 4 GB/s case; DeepseekMoE 16.4B and Qwen1.5-MoE; uses the current layer's gate input to predict next-layer experts (prefetch accuracy 78.79% without training, up to 97.15% with their shallow-favouring scheme/cache; decode up to 4.1x vs load-on-demand and 2.2x vs activation-path methods). Same lever as the family, GPU/DRAM; lossy only via optional INT4/INT2 experts.
- Who Should Own the Expert Cache? Kernel-Managed Tiering for Trillion-Parameter MoE Inference (Si, Lin, Li, Zhang; arXiv 2608.12103v2, 30 Aug 2026; read abstract, Sec 2-4, prefetch results): GH200 node, 1.45 TB pool, 17.55 MB experts, 27.0 GB/s NVMe; replay of router traces. Router lookahead delivered as kernel readahead at 64.7% measured recall changes median time by 0.3%; perfect one-layer advice gains 5.0% and nothing through blocking reads; oracle-pinned arena is 1.09-1.11x faster than page cache; Qwen3-30B-A3B also replayed on an M-series Mac. Strong evidence that one-layer-lead advice is worth <=5% when I/O is already the bulk of the step.
- FlashMoE: Reducing SSD I/O Bottlenecks via ML-Based Cache Replacement for MoE Inference on Edge Devices (Kim, Lee, Han, Yoo, Kim; arXiv 2601.17063v2, 5 Oct 2026; read abstract/eval): Ryzen 5 9600X + RTX 5070 Ti 16 GB + 64 GB DDR5 with ~1 GB left free by locking, SK hynix P51 7.4 GB/s NVMe; OLMoE-1B-7B and Qwen3-30B-A3B; ML (113 KB FFN) eviction policy approximating Belady using recency/frequency, hit rate +21% over LRU and +51% over LFU on OLMoE, up to 2.6x over Fiddler/DAOP/llama.cpp/Transformers (prefill 2.5x vs llama.cpp). Cache policy, not prefetch; shows a cheap learned eviction beats LRU by 21 hit-rate points in a tiny-cache SSD regime. Does not break our conclusion about prefetch; is a candidate for our cache family.
- Edge0, The Other Half of the Memory Wall: Serving 35B MoEs from SSD with Trained Routing Prediction (Lin, Wang, Cai, Liu, Zeng; AutoArk; arXiv 2609.18063v2, 17 Sep 2026; read Sec 1-5 key parts). Qwen3.6-35B-A3B (40 layers x 256 experts + shared, int4 19.5 GB, 1.77 MB per expert), Mac mini M4 Pro 24 GB: 20.4 tok/s at 2.9 GiB active with routing width K=4 (reduced from 8), 19.9 tok/s with the prerouter disabled; resident mlx-lm at 3.9 tok/s because the 19.5 GB model barely fits. On a 16 GB MacBook M2 with an 18.4 GiB checkpoint streamed from SSD: prerouter prefetch lifts decode 4.8 -> 8.6 (K=2), 3.5 -> 6.4 (K=4), 1.8 -> 3.3 tok/s (K=8) (+80/+82/+84%, Table 3), blocked-load time 244.0 -> 101.9 ms per step at K=4; disk read is at most 12% of the step, GPU 35-41%. Quality: mean per-benchmark gap to fp16 3.9 points (35B) and 2.8 (8B) over five benchmarks, bought back by an unmerged recovery LoRA. The predictor is trained, one token ahead, and its output IS the routing (so a miss changes the computation instead of stalling).
- DALI, SpecMD, Cache-conditional experts, MoE-Infinity, Flash-MoE blog (first-pass list; not re-read here).
- Search-snippet only: FreeToken (arXiv 2608.16157, blog), MoEpic, EStream (2609.06551); search found no paper on CPU-only llama.cpp expert streaming through the OS page cache.

What is new in OUR variant: the combination (lossless, id-based/history predictor, one-layer lead, 2-8 footprint cache, top-10 of 512 on a 3 GB/s NVMe) is not published in that exact form; the closest regime papers are Budgeting Bytes (P-0158, bandwidth-saturated eMMC, null for oracle), 2608.12103 (<=5%), HOBBIT (prefetch 1.01-1.05x) and Edge0 (long lead, trained, lossy, large gain). The new thing Edge0 shows is that a one-TOKEN lead (not one-layer) predictor that is allowed to alter routing converts prediction into +80% on a Mac whose disk is not saturated. We did not model that.

Verdict: novelty = known. Published evidence for a lossless one-layer-lead prefetch gain in a bandwidth-bound regime: no. Our measurement (1.00-1.075x) stands and is corroborated by HOBBIT's ablation (1.01-1.05x), 2608.12103 (0.3% / 5.0%), Budgeting Bytes (oracle null). Result that qualifies our belief: Edge0 Table 3 (+80% on 16 GB M2 streaming from SSD) shows the gain is large when (a) the lead is a full token step and (b) disk is under 12% busy; our sim's oracle row with a one-layer lead and a shared FIFO disk (1.25-1.6x) does not cover a token-lead predictor, whose bound is the 1.9x perfect-overlap limit already in D.md. It requires training heads and tolerates a 2.8-3.9 point quality gap, so it is not a free lossless lever.

## Family screen 2 #5 (shared low-rank basis + low-bit residual): rows P-0161, P-0162, P-0163, P-0164, P-0165

Prior art found and read in raw text beyond the rows:
- CARDAN, A Multi-Engine Dataflow for MoE Decoding on AWS Trainium (Ma, Fan, Liu, Li; arXiv 2609.21137; read abstract and the representation section). Independent evidence on a real 128-expert model: on Qwen3-30B-A3B a pure shared-basis low-rank representation raises perplexity from 9.82 to 34.91 (Sec 1); reconstruction NMSE at about 4 bits/weight is 13.46% for low-rank only vs 5.72% for VQ+low-rank vs 1.62% for VQ only (d=2); low-rank at 8 bits/weight (6.98%) is still worse than VQ+LR at 3 bits (6.73%) (Fig 5a). Speedups 1.15-1.31x batch 1 on Trainium3 over BF16 megakernels. This is the nearest published combination of our families 2 and 5 and it says the shared basis alone is not enough.
- MC-SMoE (Li et al., ICLR 2024, arXiv 2310.01334) and Sub-MoE (arXiv 2506.23266): raw text downloaded; first-pass numbers (up to 80% memory, 20% FLOPs; 96%/86% retained at 25%/50%) not re-verified at table depth this pass.
- Search-snippet only: BitsMoE (2606.00079), RFID-MoE (2602.09316), LorExperts/BTExperts (2608.07814), MoLAE, plus MoE-SVD's own Table 2 (read) which compares ASVD/SVD-LLM/MoE-I2/MC-SMoE on Mixtral at 20-60%: MC-SMoE at 20% collapses (PPL 1341), MoE-I2 PPL 18.8, ASVD 9.44 vs MoE-SVD 4.86.
- Memory (from memory): Delta-CoMe, BitDelta (fine-tune deltas), QMoE.

What is new in OUR variant: none in the shared-structure idea. Specific to us: keeping the basis RESIDENT and streaming only a 2-3-bit residual per expert from NVMe, for a 512-expert model on an 8-32 GB machine. No paper evaluates this with streaming. The best evidence about whether the basis can be small enough comes from three independent places: (1) MoBE Appendix B (effective rank above the SVD break-even for Qwen3-235B, DeepSeek-V3, Kimi-K2), (2) CARDAN (Qwen3-30B PPL 9.82 -> 34.91 with low-rank only), (3) our own OLMoE probe (16 of 64 components = 27.8% of centred energy vs 25% chance). Closest per-expert alternatives that do work: P-0165 (per-expert low-rank compensator for the top-n experts) and CARDAN (VQ + LR).

Verdict: novelty = known / variant; published evidence for a speed-up in our regime: no. The byte cut that the published methods reach (24-60% parameters, i.e. 1.3-2.5x) at 0.6-11 points of quality is what our closed form gives (1.5-1.75x serial, 1.09x overlap at 8 GB). New information from the full reads: MoE-SVD's headline ('60% compression, 1.5x faster, minimal loss') is contradicted by its own Table 2 (Mixtral acc 0.63 -> 0.37 and PPL 3.98 -> 13.52 at 60%); D2-MoE's quality at 60% is -0.11 average accuracy on Mixtral; P-0165 shows a 2-bit base with compensation fails on a fine-grained MoE (DeepSeek-MoE-16B INT2 + top-3 compensators: MMLU 28.75%, near chance). DeltaMoE (P-0164) is UNREAD.

# Corrections to the first pass (P07.md/P07.json)

1. P-0156: first pass said the catalog's 1.66x/2.07x are 'swapped'. Actually 2.07x (up to 5.02x) is the abstract's pooled decode average over both frameworks; transformers-only is 2.47x vs TO/UM, 1.47x vs static, 1.31x vs LRU; llama.cpp is 1.66x (up to 2.07x) over a mix incl. LO, 1.49x vs static, 1.09x vs LRU. Added: predictor accuracy 84.7% is an average over DS-1 and QW-2; cache sweep numbers.
2. P-0157: 9.93x is Phi-MoE vs MoE-Infinity on Orin; the paper also reports 13.0x / 18.9x vs llama.cpp on Orin (missed). Prefetch alone is 1.01x (Mixtral) / <1.0 (Phi-MoE), about 1.05x with substitution; dynamic precision loading alone is 1.19-1.57x.
3. P-0158: the 'unverified' items are verified: madvise(WILLNEED) advisor is in the paper; 18.8% previous-token hit is on the author's own trained 24M MoE (8 experts, top-2), not a published top-2-of-8 model; on Qwen3-30B previous-token top-8 hit is 46.7%. The 22x is Q2_K on an M4 16 GB vs the 0.51 tok/s eMMC run; quality unmeasured.
4. P-0159: Table 4 accuracies are FIRST-LAYER only; '0.27-0.64 ms/token' is the expected residual loading time, not savings (savings vs Fate up to 1.37 ms/token); paper's '58.4x' disk-vs-memory is inconsistent with its Table 5 (5.1-8.4x). Speculating Experts: speculated experts are executed (lossy): Qwen3-30B average 0.817 -> 0.641, GSM8k 0.950 -> 0.576; 14% is Qwen3-30B on A6000, 5-8% elsewhere.
5. P-0163: first pass read the abstract only. Full text: Mixtral 60% -> 1.78x runtime but average accuracy 0.37 (orig 0.63), PPL 13.52; the abstract's 1.5x is the Phi-3.5 60% figure; 'minimal performance loss' is contradicted by Table 2 at 50-60%.
6. P-0162: abstract's '1-2% drop' holds for the three huge models (0.6 / 1.3 / 1.3) but Qwen3-30B-A3B loses 2.8 points and AIME/LCB/MultiPL-E lose 4-9 points.
7. P-0164: first pass marked 'accurate'; in fact UNREAD (abstract only); the transfer judgement is from the abstract.
8. P-0165: numbers confirmed; added that DeepSeek-MoE-16B (the fine-grained model) is not rescued at INT2 (MMLU 28.75% with top-3 compensators), performance may be partly simulated (Ramulator, MoNDE config), GPU-only measured/modelled status and batch size not stated.
9. P-0153 / P-0154: numbers confirmed; Llama-3-8B KV is 131,072 B/token by arithmetic (paper says 120 KB).

# Results that change what we should believe about our own measurements

- Corroborated (raises confidence): previous-token persistence 0.42-0.46 (D.md) vs Budgeting Bytes 46.7% on Qwen3-30B; lossless prefetch 1.00-1.075x (D.md) vs HOBBIT's own ablation 1.01x/1.05x and 2608.12103's 0.3%/5.0%; flat expert spectrum (our OLMoE probe) vs MoBE Appendix B and CARDAN (Qwen3-30B low-rank-only PPL 9.82 -> 34.91).
- Qualified: our 'prefetch refuted' verdict (D.md s2_4) is specific to lossless id-based predictors with a ONE-LAYER lead on a mostly saturated disk. Edge0 (arXiv 2609.18063) streams from an SSD on a 16 GB Mac with a trained one-token-ahead predictor whose output is the routing and gets +80% decode (4.8 -> 8.6 tok/s at K=2), with disk <=12% busy and a 2.8-3.9 point benchmark gap. Our sim has no token-lead lossy predictor and its disk is ~86% of the compute time per token; the 1.9x overlap bound in D.md applies, but we have not tested a token-lead predictor or what share of that gain survives top-10-of-512.
- D.md s2_4 'Published vs measured' attributes ProMoE 1.66x and HOBBIT 9.93x to 'stronger predictors / lossy substitution'. More precisely: ProMoE vs LRU in llama.cpp decode is 1.09x; HOBBIT's prefetch is worth ~1.0-1.05x, and 9.93x is mainly a byte cut plus a thrashing baseline.
- D.md s2_5 lists MoE-SVD as 'unmeasured'; its Table 2 shows large quality losses, supporting the refuted verdict for aggressive shared-structure compression.
- P-0165 (family 9 relevance): a 2-bit base with per-expert compensators only works for a skewed router (Mixtral top-1); on a 64-expert top-6 model it stays near chance, which is a caution for our '2-bit base + residual for the top experts' variant on 512 experts.
