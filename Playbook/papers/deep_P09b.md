# Deep re-check P09b (raw full text): P-0199, P-0200, P-0201, P-0202, P-0203, P-0205, P-0206, P-0207, P-0208, P-0209

Texts read from raw sources under scratchpad deep_P09b (pdftotext of the arXiv PDFs, Discourse JSON, blog and docs HTML, Google Patents HTML). All 10 rows read from raw text, 0 unread. P-0204 (idea row) and P-0197/P-0198 are not in this batch.

## P-0199: FloE: On-the-Fly MoE Inference on Memory-constrained GPU (Zhou*, Li*, Zhang, Wang, Wang, Xie, Chen, Shou; Zhejiang Univ. / Hangzhou High-Tech Zone Inst. of Blockchain and Data Security / UW). ICML 2025 (PMLR 267), arXiv 12 May 2025

- ID / version read: arXiv 2505.05950v2 (PDF, pdftotext; first pass opened the v1 HTML; the numbers I checked are the same)
- Raw text read: True
- Setup: Mixtral-8x7B only (47B total, 8 experts, top-2; every test case, footnote 3). Decode single batch, ShareGPT prompts, input/output lengths (32|64|128) x (128|256|512), mean of 5 runs. Hardware: one GeForce RTX 3090 24 GB, 64-core 2.3 GHz CPU, 256 GB DRAM, PCIe 4.0 (single-expert tests also on H100, A100, A6000). Baselines: DeepSpeed-MII (ZeRO-Infinity, FP16 offload), Mixtral-Offloading (dvmazur; expert cache + quantization), Fiddler (CPU-GPU co-execution), Mixtral-GPU (HQQ INT2 fully GPU resident, the upper reference). Method: gate/down projections sparsified by magnitude of the up-projection output (80-90% sparsity), up projection stored INT2 (HQQ), two predictors (learned inter-expert next-layer predictor; parameter-free intra-expert predictor that reuses the next layer's up-projection on the current hidden state), Triton sparse GEMV kernel (CATS-derived), compact column/row layout and AVX-512 multithreaded pinned-memory transfer. Accuracy: 7 lm-eval tasks (ARC-e, ARC-c, BoolQ, SciQ, OBQA, Winogrande, 5-shot MMLU) and WikiText-2.

- Numbers (what | value | baseline | ratio | where):
  - Decode speed vs DeepSpeed-MII (avg over 9 length settings, 12 GB VRAM budget) | 48.7x | DeepSpeed FP16 offload, bars 0.02 of Mixtral-GPU | 0.91/0.02 = ~45 from rounded bar labels; text says 48.7x | Sec 4.1 text, Fig 6, p7-8
  - Decode speed vs Mixtral-Offloading | 2.60x | Mixtral-Offloading avg bar 0.35 of Mixtral-GPU | 0.91/0.35 = 2.6 | Sec 4.1, Fig 6
  - Decode speed vs Fiddler | 3.14x | Fiddler avg bar 0.29 | 0.91/0.29 = 3.14 | Sec 4.1, Fig 6
  - FloE relative to GPU-resident INT2 Mixtral | 91% avg (95% max) | Mixtral-GPU = 1.0 | 0.91 | Sec 4.1, Fig 6 (absolute tok/s is not stated in the text, only relative bar labels and an axis to 7 tok/s)
  - Sparse GEMV kernel, one expert, RTX 3090, 50% / 70% / 90% sparsity | 0.379 / 0.316 / 0.283 ms | dense 0.542 ms | 1.43x / 1.72x / 1.92x (table values 1.43/1.72/1.92) | Table 1, p8
  - Same kernel on H100 at 90% | 0.103 ms | 0.169 ms | 1.64x | Table 1
  - Compact async transfer vs PyTorch native | up to 88% of PCIe peak | PyTorch native | 12.6x faster | Fig 7, Sec 4.1
  - Accuracy, 7-task average, Mixtral-8x7B full | 72.0 | - | - | Fig 10, p9
  - Accuracy FloE-80% / FloE-90% (sparsity + INT2 up) | 67.6 / 64.4 | 72.0 | -4.4 / -7.6 absolute points (-6.1% / -10.6% relative) | Fig 10
  - Accuracy HQQ INT3 / HQQ INT2 / CATS-80% / CHESS-80% / FloE-Wup-80% | 66.5 / 31.1 / 65.7 / 66.5 / 69.3 | 72.0 | - | Fig 10
  - Predictor quality | next-layer cosine similarity >0.95 (except layer 0); inter-expert precision avg 0.88; intra-expert recall avg 0.95 | - | - | Fig 4, Sec 3.3
  - Expert size / transfer | Mixtral expert >300 MB FP16, ~15 ms over PCIe 4.0 vs ~5 ms compute on 3090; 9.3x per-expert compression; 11 GB VRAM minimum | - | 3x I/O-bound | Abstract, Sec 3.1
- Catalog / first-pass numbers checked:
  - [yes] catalog line: intra-expert sparsity predictor reusing the up-projection plus inter-expert next-layer predictor, in an offloading setting: matches Sec 3.3 (line truncated in the catalog, no numbers in it)
  - [yes] first pass: 2.60x vs Mixtral-Offloading: 2.60x, at 12 GB VRAM, avg of 9 length settings
  - [yes] first pass: 48.7x vs DeepSpeed-MII: correct but baseline is DeepSpeed ZeRO-Infinity FP16 offload (bar 0.02), not a tuned system; the ratio is dominated by the weak baseline
  - [yes] first pass: 91% of GPU-resident: 91% of an INT2 GPU-resident Mixtral (a lossy reference), 95% max
  - [yes but units unstated] first pass: acc loss 4.4-7.6%: 4.4 and 7.6 are ABSOLUTE points on the 7-task average (72.0 -> 67.6 / 64.4); relative loss is 6.1% / 10.6%. 80% / 90% sparsity + INT2 up projection.
  - [yes] first pass: setting Mixtral-8x7B, RTX 3090: also 256 GB DRAM, PCIe 4.0, 12-24 GB VRAM sweep (Fig 8)
- Transfers to our regime: **partly**. The mechanism that transfers is the idea itself: sparsify gate/down by |up-projection output| and predict the mask early by multiplying the previous layer's hidden state with the next layer's up matrix (hidden states of consecutive layers have cosine >0.95; intra recall 0.95). What breaks: (1) all speedups are PCIe DRAM-to-VRAM transfer savings on a 24 GB GPU with 256 GB DRAM; ours is CPU-only with weights on NVMe, so the saved bytes would be NVMe bytes, but the baseline is not DeepSpeed FP16 offload, it is already a 4.3-bit quantized layout with an LRU expert cache. (2) Up projection must be kept resident (INT2) or read in full, so at best 2/3 of the bytes are sparsified: gate-first/up-first caps byte saving at 1.5x for 50% (matches our pubtech/E.md gate-first cap). INT2 up projection is a quality loss we cannot take at 4.3 bits without our own test. (3) Mixtral has 8 experts of ~300 MB (FP16) with top-2; ours are 512 small experts, top-10 (a few MB each at 4.3 bits): sparsifying 80-90% of a few-MB expert leaves reads of ~0.3-0.5 MB, far below efficient NVMe read sizes (our E.md reads: 4 MiB 2.4 GB/s, 1 MiB 1.56 GB/s), so the compact layout trick matters but the read-size penalty is not measured by the paper. (4) The sparse magnitude threshold is calibrated on Mixtral, not on a 512-expert top-10 model, whose neurons may be less sparse. (5) Cache: FloE's best result needs cached layers in VRAM (Fig 8); our cache is 12-46% of the experts in RAM.
- Quality loss: Yes, lossy: sparsity 80%/90% plus INT2 up projection gives -4.4 / -7.6 points average over 7 tasks (Fig 10); per task FloE-90% MMLU 53.1 vs 69.5 dense (-16.4 points; Fig 10 MMLU column). FloE-Wup-80% (without INT2) -2.7 points (69.3), FloE-Wup-90% -6.4 (65.6). For comparison (Fig 10): HQQ INT3 66.5, CATS-80% 65.7, CATS-90% 55.9, HQQ INT2 31.1. Authors note the MMLU drop from FloE-Wup to FloE is the largest.
- Ablation: Yes, partly: sparsity strategy (gate vs up vs down magnitude, Fig 3a/9a, up-projection chosen), quantization sensitivity per matrix (Fig 3b), sparse kernel alone (Table 1, up to 2x for the kernel), compact transfer alone (Fig 7, 12.6x), VRAM sweep 12-24 GB (Fig 8). There is no ablation separating predictor, sparsity and caching contributions to the end-to-end 2.6x, and no cache-size-as-share-of-experts isolation.
- Limitations (authors): Authors do not include a limitations section. Stated scope: single-batch latency-sensitive decoding; all tests on Mixtral-8x7B (other models only for sensitivity in the appendix); when VRAM is sufficient raising up-projection bits recovers MMLU. Not stated: prefill/TTFT, batch >1, NVMe.
- Versus our own findings: No contradiction. It agrees with our E.md closed form: with up kept dense the byte gain is capped (gate-first cap 1.5x at d=0.5), and sparse kernel speedup of the expert matmul is 1.43x at 50% (our compute-ceiling lift 1.17x at d=0.5 because dense share is fixed). It does not support the screen's 2.6x claim; the 2.6x in this paper is a different quantity (vs Mixtral-Offloading, includes INT2 and prediction).

## P-0200: Turbo Sparse: Achieving LLM SOTA Performance with Minimal Activated Parameters (Song, Xie, Zhang, Wen, Ma, Mi, Chen; IPADS SJTU, Tsinghua, Shanghai AI Lab). arXiv preprint, v2 11 Jun 2024

- ID / version read: arXiv 2406.05955v2 (PDF); first pass read v1 HTML
- Raw text read: True
- Setup: Mistral-7B and Mixtral-47B (8 experts, top-2) converted to dReLU (ReLU on both gate and up) by continued pretraining on ~150B tokens (diverse web/code/math mixture, Table 4) on 64 A800-80G, then SFT (predictors trained per expert during SFT). Small-scale study: 300M models, 5B tokens fineweb (dReLU vs SwiGLU). Speed: PowerInfer (sparse-aware) running TurboSparse models vs llama.cpp running the ORIGINAL dense-activation models. Hardware: PC-Laptop i9-14900HX 32 GB host RAM + RTX 4090 16 GB, PCIe 4.0; PC-2080Ti i7-12700K 64 GB + RTX 2080Ti 11 GB, PCIe 3.0; OnePlus 12 (Snapdragon 8 Gen 3, 24 GB DRAM, UFS 4.0) with PowerInfer-2. INT4 for Mixtral, FP16 for Mistral. Decode, batch 1.

- Numbers (what | value | baseline | ratio | where):
  - Natural sparsity Llama-2-7B / ReLULlama-7B / ShiftedReLULlama-7B | 40% / 67% / 71% | - | - | Table 1, p4
  - 300M model WikiText-2 ppl, top-k magnitude sparsity 0/50/80/85/90%: dReLU | 28.45 / 28.45 / 28.45 / 28.65 / 29.19 | SwiGLU 28.48 / 28.62 / 36.28 / 48.55 / 112.36 | dReLU 90% = +2.6% ppl; SwiGLU 80% = +27% | Table 3, p6
  - Neuron sparsity after dReLU: TurboSparse-Mistral-7B / TurboSparse-Mixtral-47B (per expert) | 90% / 85% zero-valued activations | - | with 75% expert-level sparsity: 3% of MoE-layer params active | Sec 6.2, Fig 6, p8
  - Active params Mixtral 13B -> TurboSparse-Mixtral | 4.3B | 13B | 3.0x fewer | Table 6, p8
  - Quality TurboSparse-Mixtral-47B vs Mixtral-47B (OpenLLM avg) | 71.76 | 69.34 | +2.4 (MMLU 70.48 vs 70.53, GSM8K 68.50 vs 58.91, TruthfulQA 56.64 vs 48.59, Hellaswag 85.22 vs 86.62, WinoGrande 82.24 vs 83.35) | Table 6
  - CPU decode, PC-2080Ti Mixtral-47B INT4 | 11.98 tok/s PowerInfer | 4.26 llama.cpp | 2.81x | Table 7, p9
  - CPU decode, PC-Laptop Mixtral-47B INT4 | 16.1 | 6.91 | 2.33x (paper: 2.32) | Table 7
  - CPU decode, Mistral-7B FP16 (2080Ti / Laptop) | 9.94 / 8.71 | 4.78 / 4.13 | 2.08x / 2.11x | Table 7
  - Hybrid CPU/GPU decode Mixtral-47B INT4 (2080Ti / Laptop) | 22.24 / 33.12 | 6.63 / 13.1 | 3.35x / 2.53x | Table 8
  - OnePlus 12 Mixtral-47B INT4 (PowerInfer-2 vs llama.cpp) | 11.1 | 0.5 | 22.2x | Table 9, p10
- Catalog / first-pass numbers checked:
  - [yes] catalog: SwiGLU about 40%: Llama-2-7B natural sparsity 40% (Table 1); it is the fraction of exact/near-zero activations, not what magnitude pruning can reach
  - [yes] catalog: ReGLU about 70%: 67% (ReLULlama), 71% (shifted ReLU)
  - [wrong/misleading] catalog: dReLU 75% to 97%: 75% is the expert-level (2 of 8) sparsity of Mixtral; 97% is the combined fraction (routing x neuron). Within-expert neuron sparsity is 85% (Mixtral), 90% (Mistral); dReLU 90% on the 300M model.
  - [yes] catalog: retraining needed: ~150B tokens on 64 A800 (Sec 5, Table 5)
  - [yes] first pass: CPU 2.08-2.81x vs llama.cpp: Table 7 gives 2.08, 2.81, 2.11, 2.32 (the text says 2.08-2.28, a typo). Baseline is llama.cpp on the original un-sparsified model, so the ratio mixes sparsity, predictor and PowerInfer engine.
  - [yes] first pass: phone 11.1 vs 0.5 tok/s: Table 9, 22.2x; 24 GB phone, UFS 4.0
  - [partly] first pass: needs ~150B-token retraining; existing SwiGLU checkpoints stay ~40%: 40% is the natural zero fraction. The paper's own Table 3 shows a SwiGLU model loses almost nothing at 50% top-k magnitude sparsity (28.48 -> 28.62 ppl) and degrades at 80% (36.28); Fig 5a shows Mixtral/DeepSeek-MoE/Qwen1.5-MoE lose ~1-2% average performance at 50% sparsity without any retraining.
- Transfers to our regime: **no**. (1) Requires the ReLU-fied checkpoint: ~150B training tokens, and every released 125B-class MoE we could run is SwiGLU, so we cannot get the 85-97% sparsity. (2) The speed comparison is against dense-activation llama.cpp on the same machine class and includes PowerInfer's predictors (per-expert neuron predictors trained in SFT), which cost memory (PowerInfer-2 measures 2.6 GB of predictors for the 47B). (3) Hardware is GPU+CPU PCs with 32-64 GB RAM holding a 47B INT4 model (~24 GB) - the model fits in RAM (Table 7 'CPU only' means RAM-resident), so there is no NVMe streaming in this paper; only the phone row streams (via PowerInfer-2). (4) Mixtral has 8 big experts; at 512 experts top-10 the sparse-FFN share is different. What does transfer: the observation (Fig 5a) that expert FFNs stay as sparse as dense FFNs (~50% prunable for ~1-2%).
- Quality loss: Training-based: TurboSparse-Mixtral average 71.76 vs 69.34 original (Table 6), i.e. no loss (gain), but only after 150B tokens of continued pretraining plus SFT; contamination/mixture effects not isolated. Training-free sparsity on the original SwiGLU MoE (Fig 5a): about 1-2% average drop at 50% sparsity, steeper beyond 50% (no table, chart only).
- Ablation: Partial: dReLU vs SwiGLU at small scale (Tables 2-3), no ablation of predictor vs sparsity vs engine in the speed tables; no ablation of cache size.
- Limitations (authors): Preprint under review, no limitations section. Sparsity measured as exact zeros on fineweb; predictor accuracy not reported in the paper; speed baselines run different checkpoints.
- Versus our own findings: Consistent with E.md (TurboSparse needs ReLU-fication). One refinement: E.md says SwiGLU experts are 'only about 40% sparse until ReLU-fied'; the paper's Table 3 and Fig 5a show that magnitude-threshold sparsity of 50% is nearly free even on SwiGLU, which is the TEAL/CATS regime, not the 40% number.

## P-0201: PowerInfer-2: Fast Large Language Model Inference on a Smartphone (Xue*, Song*, Mi, Zheng, Xia, Chen; IPADS SJTU). arXiv v3 12 Dec 2024

- ID / version read: arXiv 2406.06282v3 (PDF; pdftotext works, the first pass said the PDF was unreadable)
- Raw text read: True
- Setup: Smartphones OnePlus 12 (24 GB DRAM, 19 GB usable, UFS 4.0, Snapdragon 8 Gen 3) and OnePlus Ace 2 (16 GB / 11 GB usable, UFS 3.1). Models: Mistral(SiLU)-7B, sparse Qwen2-7B, Bamboo-7B, sparse Llama-13B, TurboSparse-Mixtral-47B (INT4). Baselines: llama.cpp (mmap offload), LLMFlash re-implemented in llama.cpp (predictor, row-column bundling, neuron cache), QNN, MLC-LLM. FFN weights 50% in DRAM (75% offloaded for Mixtral on Ace 2). Decode prompts up to 64 tokens, 1024 output tokens, mean of 10 runs, batch 1 (BoN in 7.4). Method: neuron clusters as I/O/compute unit, gate-up-down neuron bundles (7.5 KB for INT4, 24 KB for FP16) read as one random read, segmented neuron cache (hot/cold), I/O-compute pipelining at cluster granularity, NPU for prefill/dense clusters, mlock via rooted Android.

- Numbers (what | value | baseline | ratio | where):
  - Decode speed, Mixtral-47B on OnePlus 12, all 19 GB usable | 11.68 tok/s | LLMFlash (3.12x faster) / llama.cpp (21.2x faster) | 3.12x / 21.2x (text) | Sec 7.2.3, Fig 10, p10
  - Decode speed, 50% of FFN weights in DRAM, average over models, OnePlus 12 | 24.6x avg (up to 27.8x) vs llama.cpp; 3.84x avg (up to 4.63x) vs LLMFlash | llama.cpp / LLMFlash | as stated; Fig 7 labels show 11.1 vs 0.4 (27.8x) | Sec 7.2.1, Fig 7, p9
  - Decode speed, Mixtral-47B at 50% FFN in DRAM | 9.96 tok/s | - | - | Sec 7.2.1 text
  - Decode speed, only 7 GB available (400 MB neuron cache = 1.8% of FFN) | 2.13 tok/s | LLMFlash | 1.84x faster than LLMFlash | Sec 7.2.3, p10
  - Memory split at 7 GB | 1 GB non-FFN, 2.6 GB predictor weights, 2.7 GB quantization scales, 0.3 GB runtime, 0.4 GB cache | - | - | Sec 7.2.3
  - Cache miss rate, Mixtral-47B, 50% FFN in DRAM | 3.5% avg, P99 18.9%; P99 token latency 40.9% above mean (Bamboo: ~16.5% for slowest 10%) | - | - | Sec 7.2.4
  - SiLU-based Mistral-7B, 50% FFN offloaded: PowerInfer-2 / LLMFlash | 5.3 / 2.18 tok/s | LLMFlash | 2.4x | Table 6; Bamboo-7B (ReLU) 11.1 vs 2.4 = 4.6x
  - Time split, Bamboo-7B: LLMFlash vs PowerInfer-2 I/O share | 76.7% vs 13.7% | - | - | Table 4
  - Ablation Bamboo-7B, 50% offload: base / +bundle / +neuron cache / +pipeline / +XPU | 0.4 / 1.1 / 4.18 / 9.60 / 11.07 tok/s | - | step ratios computed from the numbers: bundle 2.75x, cache 3.8x at 95% hit rate, pipeline 2.3x, XPU 1.15x; the text states 2.82x for the cache and 1.29x for the pipeline, which do not match its own numbers | Fig 14, Sec 7.5
  - Accuracy (OpenCompass avg of ARC-C, ARC-E, MMLU, GSM8K): Bamboo-7B llama.cpp / QNN / PowerInfer-2 | 70.12 / 63.26 / 68.35 | llama.cpp | -1.77 points vs llama.cpp | Table 7, p12
  - Accuracy Qwen2-7B: llama.cpp / QNN / PowerInfer-2 | 79.25 / 56.93 / 78.38 | llama.cpp | -0.87 points | Table 7
  - Flash I/O of UFS 4.0 | sequential 4 GB/s at 512 KB blocks (450 MB/s at 4 KB); random 4 KB in 128 MB range ~1 GB/s; 12x gap in random IOPS vs NVMe cited | - | - | Sec 2, Table 1
- Catalog / first-pass numbers checked:
  - [yes] catalog: 11.68 tok/s TurboSparse-Mixtral-47B on a phone: 19 GB usable memory, UFS 4.0
  - [yes, with condition] catalog: up to 27.8x over baselines: 27.8x is the best single model at 50% FFN in DRAM vs llama.cpp (avg 24.6x); vs the better baseline LLMFlash the gain is 3.84x avg / 4.63x max; at 19 GB 3.12x vs LLMFlash
  - [yes] catalog: needs a sparse-trained model: Mistral SiLU gets only 5.3 tok/s (2.4x vs LLMFlash) vs 11.1 on ReLU Bamboo
  - [yes] first pass: 2.13 tok/s with 7 GB: with 1.8% of FFN cached
  - [yes] first pass: SiLU Mistral only 5.3 tok/s: but that is a 7B model with half of FFN weights resident, not comparable to the 47B figure
  - [yes] first pass: caveat 'phone flash': UFS 4.0 single queue, 4 GB/s sequential; also model quality is not measured for the 47B in this paper (Table 7 only 7B models)
- Transfers to our regime: **partly**. Transfers: the design principle that fits our regime is neuron-bundle reads (gate-up-down co-located, one random read per neuron bundle), cluster-level I/O/compute pipelining, and a segmented hot/cold cache; the ablation (Fig 14) shows bundle + cache + pipeline give 0.4 -> 9.6 tok/s on the phone. The 95% cache hit rate and 3.5% miss rate depend on the cache holding most of the activated neurons, because ReLU-fied TurboSparse activates only ~3B of 47B and 19 GB of 24 GB INT4 weights stay resident (about 75-80% of the model), whereas ours has 8-32 GB against a 69 GB model (12-46%) and SwiGLU. At 7 GB (1.8% cache of FFN) the system gets 2.13 tok/s, which is the closest to our 8 GB case, but it is a 47B with 3B active at 4 bits and 0.4-6 GB-class reads per token, vs our 125B with 6B active. Breaks: needs the sparse-trained model; NPU + CPU heterogeneity; UFS not NVMe; 2.6 GB of predictors; Android root.
- Quality loss: PowerInfer-2 itself adds no extra loss beyond the sparse-model predictors, but the paper only measures accuracy on Bamboo-7B and Qwen2-7B (Table 7: -1.77 and -0.87 average points vs llama.cpp using group quantization) and not on TurboSparse-Mixtral-47B; 'negligible accuracy degradation' for the 47B is not shown here (TurboSparse paper reports no loss with retraining).
- Ablation: Yes (Fig 14, Bamboo-7B, 50% offload): bundling, neuron cache, pipeline, XPU each isolated; cache size sweep 7-19 GB (Fig 10) isolates memory. Layout (bundle) vs cache vs pipeline are separated, which is the useful part for us.
- Limitations (authors): Not in a limitations section; the text notes the 47B at 9.96 tok/s still has room with a larger neuron cache, and UFS single command queue, 40% bandwidth loss under contention. Android root needed for mlock.
- Versus our own findings: No contradiction; supports that cache size is the lever (decode speed scales linearly with memory up to 19 GB, Sec 7.2.3). Their finding that reads of a bundled 7.5 KB neuron unit work well on UFS (the cluster is small) is a data point our small-read penalty model (E.md: 35 KB block = 0.37 of large-read rate) did not include: they report bundling raising 0.4 -> 1.1 tok/s.

## P-0202: DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction (Cai, Qin, He, Cui, Li, Huang; HKUST-Guangzhou, Univ. Maryland). arXiv v1 25 Aug 2025

- ID / version read: arXiv 2508.18376v1 (PDF)
- Raw text read: True
- Setup: Training-free post-training system on SGLang. Models: Mixtral-8x7B (also Instruct and a fine-tuned 8->32 partitioned variant), OLMoE-Instruct (64 experts, top-8), DeepSeek-V2-Lite-Chat (64 routed top-6 + 2 shared). Hardware: server with 8x NVIDIA H20; Mixtral TP=8, OLMoE 1 GPU, DeepSeek EP=8. Speed test: 2000 random prompts, input 500 / output 100 tokens, batch size not stated (continuous batching in SGLang). Quality: lm-eval-harness on ARC-C, BoolQ, GSM8K(5-shot), HellaSwag, MMLU, OBQA, PIQA, RTE, WinoGrande. Method: split each expert into 'major' and 'minor' sub-experts by calibration importance of neurons (profiling on MMLU), then dual-threshold dropping on normalised gate score (below T_minor drop the whole token-expert pair, between thresholds compute major sub-expert only); load-aware thresholds per device.

- Numbers (what | value | baseline | ratio | where):
  - Accuracy drop at ~25% computation drop rate, 2T-Drop with reconstruction (avg of tasks) | Mixtral 71.04, OLMoE ~ -0.28, DeepSeek -0.18 | no-drop 71.12 (Mixtral fine-tuned 8/32) | -0.08 / -0.28 / -0.18 absolute points | Table 2, Sec 5.3.1, p10
  - 1T-Drop / 2T partition-only vs 2T-reconstruct on Mixtral at 24% drop | 70.88 / 70.90 / 71.04 | 71.12 | -0.24 / -0.22 / -0.08 | Table 2
  - MoE-module speedup at 22-27% drop rate (single/TP/EP deployments) | 1.17x to 1.23x | no-drop SGLang | as stated | Sec 5.3.2, Fig 10
  - End-to-end speedup at the same drop rates | 1.07x to 1.12x | no-drop | as stated | Sec 5.3.2, Fig 10
  - DeepSeek-V2-Lite EP=8 with load-aware thresholding | 1.41x MoE module, 1.13x end-to-end, 0.5% average accuracy loss | no-drop | as stated | Sec 5.3.3, Fig 11
  - Comparison on Mixtral-8x7B-Instruct, GSM8K change: 2T-Drop partition / reconstruct | +0.5% / +1.2% with 1.08x speedup | EES 1.05x -2.4%; EEP r=6 1.20x (-24% memory) -8.0%; EEP r=4 1.28x -14.9% | - | Table 3, p12
  - Expert partition fine-tuning | +0.59% downstream accuracy after fine-tuning 8->32 experts | unpartitioned fine-tune | - | Sec 5.2.1, Table 1
- Catalog / first-pass numbers checked:
  - [yes] catalog: neuron-level sparsity inside experts combined with expert partition; dropping computation in MoE: correct in concept; no numbers in the catalog line
  - [yes] first pass: 1.17-1.23x MoE module, 1.07-1.12x e2e at 22-27% drop: but the abstract headline numbers are 0.08-0.28% accuracy loss at ~25% drop and 1.41x / 1.13x with load-aware thresholding
  - [yes] first pass: Mixtral, OLMoE, DeepSeek-V2-Lite on 8x H20: not all on 8 GPUs: OLMoE single H20, Mixtral TP=8, DeepSeek EP=8
- Transfers to our regime: **no**. The gain is GPU computation and load balance in batched expert-parallel serving (grouped GEMM over many tokens per expert); the drop rate is measured over (token, expert) pairs and is translated into FLOP savings, not weight-read savings. With batch 1 and NVMe streaming, each token touches its own 10 experts and a dropped pair would save its bytes, so a gate-score threshold plus a major/minor split WOULD cut I/O in our regime, but (a) the paper does not test that, (b) the drop thresholds were calibrated on top-2/top-6/top-8 models (64 experts at most), and a 512-expert top-10 router has flatter gate scores, (c) our own k-reduction measurements (B.md, decode k=4 ~1.5x with unmeasured quality) cover the dynamic-drop idea, and the paper's contribution beyond that is the static major/minor neuron reordering (which would give the in-expert half-read). Quality results are on GPU-served instruct models with GSM8K the most sensitive task.
- Quality loss: Yes (lossy, training-free): -0.08 (Mixtral fine-tuned), -0.28 (OLMoE), -0.18 (DeepSeek-V2-Lite) average points at ~22-27% token-expert drop, 9 tasks; 0.5% with load-aware thresholding at 1.41x; GSM8K the most sensitive task (they report GSM8K separately in Fig 11). 1T-Drop alone (no reconstruction) is worse on average.
- Ablation: Yes: 1T-Drop vs 2T (partition only) vs 2T (reconstruct) in Table 2 isolates the neuron-level reconstruction contribution; neuron importance metric ablation (accumulated gate / |gate| / gate-up / |gate-up|, DeepSeek avg 67.17/67.29/66.79/67.65, Sec 5.3.4); S-ETP communication only simulated/8xH20.
- Limitations (authors): Authors state: calibration-sample dependence of neuron importance profiling, drop rate non-linear in threshold and varies by layer (per-layer thresholds left to future work), designed for distributed server-side inference rather than edge, MoE variants with more recent architectures remain future work.
- Versus our own findings: No contradiction. Their EES row (skip second expert of top-2 when score below beta x first) is the same family as our decode-k-reduction rows; Table 3 gives a data point: EES -2.4% GSM8K at 1.05x, compared with our unmeasured-quality 1.4-1.7x k=4 byte gain.

## P-0203: Enabling Dynamic Sparsity in Quantized LLM Inference (SpQt) (Wang*, Shu*, Lin; Univ. of Virginia, Zoom). arXiv v1 6 Nov 2025, 'preliminary work, under review'

- ID / version read: arXiv 2511.04477v1 (PDF)
- Raw text read: True
- Setup: Llama-2 7B/13B/70B and Llama-3 8B (dense, no MoE), Q4_K via a custom 'zigzag' column-grouped quantization layout, Metal GPU kernels on llama.cpp b5711. Hardware: Apple A19 Pro (5 GPU cores, 75.8 GB/s), M2 (8, 102.4), M2 Pro (19, 204.8), M2 Max (30, 819.2 GB/s). Decode of ~10-token prompt; PPL on WikiText test. Sparsity = TEAL-style magnitude thresholds on the input hidden states of ALL matmuls (q,k,v,o and FFN), either TEAL's non-uniform greedy levels or unified; dense kernels in prefill.

- Numbers (what | value | baseline | ratio | where):
  - End-to-end decode, A19 Pro, Llama-2-7B, unified 50% / TEAL 50% | 22.86 / 22.79 (labelled ms/token but values behave as tok/s) | Q4_K dense 14.78 | 1.55x / 1.54x | Table 2, p9
  - Same, M2 Max Llama-2-7B | 68.87 (unified 50%) | 57.96 | 1.19x for this model (my reading of the garbled table columns; other models reach 1.45-1.48x) | Table 2 vs Sec 6.2 text
  - Summary in caption | 1.55x A19 Pro, 1.46x M2, 1.32x M2 Pro, 1.45x M2 Max | dense Q4_K | - | Table 2 caption
  - Summary in text | 1.47x M2 Max, 1.33x M2 Pro, 1.39x M2 | dense Q4_K at 50% | - | Sec 6.2 text; inconsistent with caption
  - Sparse GEMV micro-benchmark at 50% | 1.51x - 1.78x | dense Q4_K | as stated | Fig 9 caption / abstract (1.8x GFLOPS)
  - Perplexity Llama-2-7B: F16 / Q4_K / Unified 25,40,50,65% / TEAL 50% | 5.13 / 5.28 / 5.27, 5.40, 5.65, 7.39 / 5.59 | Q4_K 5.28 | +0.37 at 50% unified (+7%); +2.11 at 65% | Table 3
  - Perplexity Llama-3-8B Q4_K -> unified 50% | 5.93 -> 6.67 | - | +0.74 (+12%) | Table 3
  - Perplexity Llama-2-70B Q4_K -> unified 50% | 3.35 -> 3.77 | - | +0.42 | Table 3
  - Naive sparse in original layout | slowdown (A19 Pro baseline 0.42x/0.38x speeds shown only for F16) | Q4_K | - | Fig 7
- Catalog / first-pass numbers checked:
  - [wrong (fused)] catalog title: CATS / Enabling Dynamic Sparsity in Quantized LLM Inference: the URL is SpQt only, which builds on TEAL (Liu et al., ICLR 2025). CATS is a different paper (Lee et al., arXiv 2404.08763, COLM 2024), which I read separately.
  - [yes] catalog: threshold-based sparse decode on llama.cpp with custom quant layout: also needs custom Metal kernels and a re-quantization tool
  - [partly] catalog: skipping down-proj columns for zero activations: sparsity is on the INPUT of every GEMV (q,k,v,o,gate,up,down), not only down-proj
  - [yes] first pass: e2e up to 1.55x (A19 Pro): Llama-2-7B on A19 Pro; other platforms 1.2-1.46x
  - [yes (max of 1.51-1.78)] first pass: sparse GEMV 1.78x at 50%: -
  - [yes] first pass: ppl 5.28 to 5.65 at 50%: unified 50%; TEAL 50% 5.59
  - [yes, confirmed] first pass: inconsistent numbers: Table 2 caption (1.46/1.32/1.45) vs text (1.39/1.33/1.47) disagree; caption says ms/token but numbers are throughput
- Transfers to our regime: **partly**. Transfers as a CPU/NVMe-agnostic idea only: the 50% input-activation sparsity costs +0.4 ppl on quantized dense Llama, so reading half of each weight column would be nearly free quality-wise on dense models. Breaks: (1) Apple Metal GPU kernels, no CPU path (the paper says nothing about CPU or SSD). (2) Dense models only; no MoE experiment; MoE experts are small and the input hidden state to each expert is shared by 10 experts. (3) The skipped weights come from RAM-resident quantized weights; for NVMe-streamed experts, skipping columns requires column-contiguous quantization groups (their zigzag layout: groups along columns) which then fits exactly the bundle layout needed for partial reads; that part is a useful design observation. (4) Gain is 1.15-1.55x end to end at 50% because attention/embedding/etc. are not sparsified; prefill dense.
- Quality loss: Perplexity only (no task benchmarks): Q4_K -> unified 50% sparsity +0.37 (Llama-2-7B), +0.74 (Llama-3-8B), +0.42 (Llama-2-70B); 65% is clearly worse (7.39 on 7B).
- Ablation: Yes: naive sparse in default layout (B1, slowdown), own layout without load balancing (B2), full; dense zigzag kernel vs default; sparsity 25/40/50/65%. No cache or I/O ablation (everything is RAM-resident).
- Limitations (authors): Authors: preliminary work; dense GEMM/tiling for prefill not optimised, Apple GPUs only in the current implementation, Llama-3-70B excluded for a quantization problem, beam search only expected to work. No MoE.
- Versus our own findings: No contradiction. Supports 40-50% input sparsity being nearly free in quality for dense models; for the MoE the E.md compute-ceiling argument stands since this paper never tests experts.

## P-0205: Forum thread 'Prefill-only requests that consume input tokens without decoding' (OpenAI Developer Community; posted by user dedbot 2026-02-06; one reply by user _j 2026-09-22; closed 2026-09-25)

- ID / version read: Discourse JSON API of the thread (2 posts, full text read) plus OpenAI's public Prompt caching guide page (platform.openai.com/docs/guides/prompt-caching, section 'Prewarm the cache')
- Raw text read: True
- Setup: Not a paper. Feature request: API to create/reuse a prompt cache or prefill state without generating output, for speculative UI branching; motivation: 0.5-1 s time-to-first-token feels sluggish. The reply (a community member with trust level 3, not a moderator/admin, not marked as OpenAI staff) says the feature now exists in the Responses API for gpt-5.6 and later: prompt_cache_options.prewarm = true prepares the cache with no output; the first working breakpoint must be beyond 1024 tokens; also a comparison_response_id diagnostic field. The OpenAI guide (fetched HTML) confirms: 'For GPT-5.6 and later ... Set prompt_cache_options.prewarm to true in a Responses API request to prepare the prompt cache without generating output', with an example of prewarming shared instructions at startup; tokens written during prewarm are billed at the cache-write rate; minimum cacheable prompt 1,024 tokens.

- Numbers (what | value | baseline | ratio | where):
  - Time-to-first-token complaint in the request | ~0.5-1 s | - | - | Post 1
  - Minimum cacheable prefix / first breakpoint | 1,024 tokens | - | - | Reply and guide
- Catalog / first-pass numbers checked:
  - [wrong as of today (Oct 2026)] catalog: 'idea of a precompute-prefix operation is requested as a missing feature; no shipped speculative typing-time prefill found': a prewarm operation shipped for GPT-5.6+ (documented in the official guide, not only the forum reply). The shipped feature warms a KNOWN exact prefix (>=1024 tokens); it is not a streaming prefill of a prefix that grows while the user types.
  - [yes] first pass: summary 'wrong', reply of Sep 2026 says prewarm now exists: confirmed and additionally confirmed in OpenAI's docs page
  - [yes] first pass: hosted prefix warm, not typing-time: the guide's own example is startup-time prewarm of shared instructions
- Transfers to our regime: **no**. Hosted API feature on GPU servers; no local KV state, no NVMe, no MoE expert I/O. It only shows that a 'prefill without decode' call is considered a worthwhile product feature. Relevant to us only as prior art for the user-visible idea (warm before the user submits), not for its mechanism. Note it cannot capture our contribution: ours would prefill the actually-typed partial text in small chunks and pay expert-read cost; the API warms a fixed, complete prefix.
- Quality loss: None (exact prefix cache).
- Ablation: none
- Limitations (authors): No benchmarks in the thread or the guide; price: cache-write tokens billed; 1,024-token minimum.
- Versus our own findings: No.

## P-0206: US 11,556,707 B2, 'Fulfillment of actionable requests ahead of a user selecting a particular autocomplete suggestion for completing a current user input' (inventor Keun Soo Yim; assignee Google LLC; filed 2020-06-18 as US16/905,338; granted 2023-01-17; continuations US11960837B2, US12223266B2, US20250181830A1; EP4121843B1 etc.)

- ID / version read: Patent: the USPTO PDF has no text layer (image-only; pdftotext returned nothing), so the text was read from Google Patents' OCR/HTML transcription of the same patent (abstract, 21 claims, description). Not verified against page images.
- Raw text read: True
- Setup: Not an experiment. Claim 1: receive user input of natural-language characters at a client device; render a selectable GUI element containing an autofill suggestion; send the suggestion to a server; before the user selects the suggestion, receive from the server command data generated from the suggestion, which applications can use to fulfil the action; after selection, apply the command data to start the action. Dependent claims: several suggestions with priorities from historical interaction data (3-4), receive command data while the user is still typing (5), automated-assistant applications (6). Description: assistant commands such as 'Turn on the security lights', fulfilment data (links, IoT commands) pre-fetched for suggestions; terms 'language model', 'LLM', 'prefill', 'KV', 'cache' do not occur in the text (0 hits for cache).

- Numbers (what | value | baseline | ratio | where):
  - Filing / priority / grant | filed 2020-06-18, granted 2023-01-17, expires 2040-12-25 (adjusted, active) | - | - | Google Patents metadata
  - Claims | 21 (independent claim 1 plus others), 9 cited references | - | - | claims section
- Catalog / first-pass numbers checked:
  - [partly] catalog: processing partial user input before the user finishes is patented for assistant commands: what is claimed is fulfilment of the ACTION behind a displayed autofill SUGGESTION before the suggestion is selected (server-generated command data), not processing of the partial text itself and not generation by a language model
  - [yes] catalog: the same pattern, not for LLM KV: no LLM, KV cache or prefill is mentioned
  - [corrected] first pass: claims not read: claims read now; claim 1 requires a rendered selectable GUI element for an autofill suggestion and server-provided command data
- Transfers to our regime: **no**. Different object (command fulfilment for assistant suggestions, not prefill of the typed text), different system (client/server assistants). It does show the general pattern 'start work on candidates while the user types' is old and patented in a narrow form; claim 1's limitations (autofill suggestion GUI element, command data) are unlikely to read on prefilling the actual typed prefix of an LLM prompt, but that is a legal opinion we cannot make; this is a technical reading only.
- Quality loss: None (not a lossy method).
- Ablation: none
- Limitations (authors): Patent has no measurements.
- Versus our own findings: No.

## P-0207: A Photonic-CXL Memory Appliance for Scalable KV Cache Management in LLM Inference (Ding, Nishant, Ambati, Kamati, Diep; Marvell Technology). IEEE-style paper, PDF created 2026-07-03

- ID / version read: arXiv 2607.27187 (PDF, 745 text lines; no version label in the PDF)
- Raw text read: True
- Setup: Part 1, characterisation with vLLM + LMCache: 'repeated request' benchmark (100% hit rate, identical requests), batch sizes {1,2,8,16,32}, prompt lengths 1K-100K (Maverick to 1M, Scout to 4M); models LLaMA-8B (1xA100/H100/H200), LLaMA-70B (2xH200), LLaMA-405B (8xH200, TP=8, BF16), LLaMA Maverick and Scout (MoE, 8xH200). A100 server: PCIe 4.0 x16 (~32 GB/s), 512 GB DRAM, 7 TB local disk; H100/H200 PCIe 5.0 (~64 GB/s), up to 2 TB DRAM. Metric TTFT; speedup = T_compute / T_retrieve. Part 2: Marvell Photonic Fabric memory appliance, 32 TB shared across 16 hosts, 128 GB/s per host; latency emulated on Siemens Veloce Strato-M (350 ns vs 800-900 ns reported for an electrical CXL pool). Part 3: LLMServingSim simulation of LLaMA-405B on 8xH200, multi-turn chat: 10 turns/conversation, 5,120 input + 500 output tokens per turn, 512-token shared system prompt, 50-300 conversations.

- Numbers (what | value | baseline | ratio | where):
  - Host-memory KV retrieval speedup vs recompute, LLaMA-8B 1xA100 / H100 / H200 | 3.3x-27.5x / 1.9x-11.8x / 1.5x-17.2x | GPU recompute | - | Table I
  - Host-memory, LLaMA-70B 2xH200 / 405B 8xH200 / Maverick / Scout | 2.7x-51.4x / 2.7x-100.0x / 2.7x-23.1x / 2.6x-54.6x | GPU recompute | - | Table I
  - Disk (SSD) retrieval speedup, LLaMA-8B A100 / H200 / 405B 8xH200 | 0.53x-1.72x / 0.38x-1.27x / 2.1x-9.7x | GPU recompute | <1 means retrieval slower than recompute | Table I
  - Disk crossover, LLaMA-8B on A100 | speedup only above ~50K tokens; 0.38x-0.53x below 20K tokens (text attributes both to the A100, table gives 0.53 for A100 and 0.38 for H200) | recompute | - | Sec II-A3 text
  - Effective bandwidth, 405B host-memory path | 165 GB/s (8 GPUs aggregated) | - | - | Sec II-A3
  - Electrical CXL pool vs photonic latency (64-byte access) | 350 ns | 800-900 ns (Beluga CXL 2.0 switched pool) | -56% (>50% reduction) | Sec IV
  - Serving simulation TTFT, 32 TB appliance, 50-300 conversations | flat 2,690 ms, prefix hit rate 82% | 2 TB baseline: 8,156 ms at 100 conv (3x); at 300 conv hit rate 6.35% | 6.6x at 300 conversations (so baseline about 17.8 s) | Fig 8, Sec VI-B
- Catalog / first-pass numbers checked:
  - [yes] catalog: disk retrieval beats recompute only above about 50K tokens (LLaMA-8B, A100): for 8B on A100; but 405B on 8xH200 beats recompute by 2.1x-9.7x across the tested range, so the crossover depends on recompute speed (it is a GPU-prefill result)
  - [partly] catalog: MoE gains less: host-memory speedup of Maverick 2.7x-23.1x and Scout 2.6x-54.6x is below the 405B dense 100x but not below the 8B dense 17.2-27.5x; the authors' own limitations say disk experiments were only done on 8B and 405B, MoE disk was NOT tested
  - [unsupported] catalog: a 500-token KV reload would not pay off versus recompute: the smallest context tested is 1K tokens; for disk at <20K tokens on a GPU server reload is slower than recompute, but that says nothing about a CPU-only laptop where a 500-token prefill takes ~19-21 s (our D.md/E.md numbers)
  - [yes] first pass: host memory up to 100x, 6.6x TTFT multi-turn (simulation): 100x is 405B on 8xH200 at the longest tested context; 6.6x is simulated, not measured, LLaMA-405B
  - [corrected] first pass: 50K crossover claim not in abstract, could not verify: it IS in the body (Sec II-A3, Table I) - first pass read only the abstract
  - [partly] first pass catalog 'wrong' for this row: the numbers are right; the catalog's inference to a 500-token reload is the part that is unsupported
- Transfers to our regime: **no**. Server GPUs (A100/H100/H200), PCIe 4/5 at 32-64 GB/s, multi-user datacentre KV pooling with TBs of CXL memory; the 'disk' path is a 7 TB SSD behind PCIe, loaded to GPU HBM. The crossover (disk > recompute only for long contexts) is determined by GPU prefill speed, which is orders of magnitude faster than our CPU prefill (a 500-token prompt = 6e12 FLOP at ~311 GFLOP/s = ~19 s, plus expert I/O), so our break-even is at a few tokens, not 50K. Their formula S = T_compute / T_memory (T_compute ~ 2sP/MFU + attention) makes the dependency explicit; with compute that slow, KV reload wins at any length (D.md P-0153 arithmetic: loading 0.016-0.05 s vs 19-22 s).
- Quality loss: None (exact KV reuse).
- Ablation: Characterisation varies model, platform, tier (host vs disk), context and batch (Fig 1), which isolates retrieval-vs-recompute; there is no ablation of the appliance beyond the emulation latency.
- Limitations (authors): Authors: MoE variants and DeepSeek/Qwen only future work (disk tests only 8B and 405B); appliance performance projections come from emulation fed to a simulator, hardware validation pending; single-node.
- Versus our own findings: Not a contradiction but it corrects how E.md reads this row: E.md says 'disk reload does not pay below about 50K tokens, which we did not test'. The 50K figure is specific to an 8B on A100/H200; on 405B reload wins at all tested lengths, and on our CPU prefill it wins by orders of magnitude (D.md).

## P-0208: Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation (Liu, Chen, Zhang; Univ. of Chicago/Together AI/CMU). arXiv v2, 19 May 2025 (ICML 2025 paper)

- ID / version read: arXiv 2502.02789v2 (PDF)
- Raw text read: True
- Setup: Llama-3.1-8B-Instruct (BF16) as the speculator picking important prompt chunks via attention scores (look-ahead steps), main model Llama-3.1-70B-Instruct (BF16) or Llama-3.1-405B-Instruct-FP8 receiving only the kept tokens (with original position ids) in vLLM, tensor parallelism 8 on 8xH100/H200 node (efficiency on H200). Quality on LongBench, RULER, short tasks vs LLMLingua, RAG-Llama baselines; token keep rates 10-90%, fixed per task. Efficiency: (1) vLLM server with constant-QPS client on LongBench queries; (2) TTFT vs batch x sequence-length products. Theoretical FLOP model in the appendix (Fig 9: 7.72x predicted vs 7.66x measured at 32K tokens, batch 16).

- Numbers (what | value | baseline | ratio | where):
  - TTFT speedup, 405B, 10% keep rate | 7.66x (analysis: 7.72x) | TTFT with full prefill of the same model on the same node | 7.66x | Fig 3, Sec 4.7.2, App E (sequence 32K, batch 16 in the analysis)
  - Maximum sustainable QPS, 405B, multi-doc QA, 10% keep | up to 7x | no speculative prefill | 7x with >95% accuracy kept | Sec 4.7.1, Fig 1, abstract
  - Speculation overhead | higher at lower keep rates; lower at higher keep rates (annotated in Fig 9) | - | - | App E
  - Quality LongBench 70B at 10% keep | preserves most of quality for single/multi-doc QA, few-shot; summarisation degrades; code completion sometimes improves | full-context 70B | - | Table 1, Sec 4.4
  - Prior method comparison | SpecPrefill up to 87% TTFT reduction vs ~50% for compared methods | - | - | Sec 2
- Catalog / first-pass numbers checked:
  - [yes] catalog: a small model picks important tokens to cut TTFT; not typing-time prefill: accurate
  - [yes] first pass: up to 7.66x TTFT (405B, 10% keep), Llama-3.1-70B/405B, 8xH100/H200: also 7x max QPS
  - [yes] first pass: multi-turn needs recompute: Sec 5: token dropping without explicit context recomputation can fail in multi-turn conversations
- Transfers to our regime: **no**. Lossy prompt-token dropping for long contexts (32K, batch 16) on a GPU node; the 7.66x is a FLOP ratio of the dropped prompt (10% kept), where speculator FLOPs (8B) are small against 405B. On our setup the main model is a 125B MoE with expert I/O dominating prefill (500-token prefill is I/O-bound at 21.4 s), the speculator would itself need a prefill of the whole prompt, and a prompt of ~500 tokens is below the 'long-context' regime in which it pays. It is orthogonal to typing-time prefill (it reduces the work after submit; typing-time prefill hides work before submit) and could in principle be combined, at the cost of quality.
- Quality loss: Yes, lossy and task dependent: LongBench/RULER at 10% keep retains most single/multi-doc QA and few-shot quality, degrades summarisation and some RULER tasks (multi-hop tracing at ~400 kept tokens); no single delta number (Table 1, Table 2). Not usable for exact KV.
- Ablation: Yes: keep-rate sweep 10-90%, with/without look-ahead (Fig 4), versus LLMLingua/RAG baselines, versus MInference TTFT (Fig 5), speculator-size overhead; the cache/I-O dimension is not present.
- Limitations (authors): Authors: no explicit logits for all input tokens (generative tasks only); multi-turn conversation can fail without recomputing dropped context; fixed keep rate, no adaptive selection yet.
- Versus our own findings: No.

## P-0209: LMCache's New Architecture Boosts MoE Inference Performance by 10x (Weishu Deng and the LMCache Team; LMCache Blog, dated 2026-04-03)

- ID / version read: Blog HTML fetched raw with curl and tags stripped (full post read, 9,075 characters of text); no paper, no code run
- Raw text read: True
- Setup: Qwen3-235B-A22B-Instruct-2507-FP8 (235B total, 22B active, 128 experts top-8 per its model card - from memory, not in the blog), 8x NVIDIA H100 80GB, vLLM 0.18.1 with LMCache 0.4.3-dev, data parallel size 8 with expert parallel, gpu-memory-utilization 0.8, max-num-batched-tokens 1024. Baseline: vLLM native in-process KV offload to host, 50 GB per DP rank (400 GB total, process-local, no cross-rank sharing). LMCache MP mode: one shared 400 GB host KV pool (LRU) as a service shared by all ranks. Workload: 'lmcache bench engine --workload multi-round-chat --mrc-qps 2.0 --mrc-duration 120' (2 QPS for 120 s). Single run, no variance, no stated prompt lengths.

- Numbers (what | value | baseline | ratio | where):
  - Mean TTFT | 0.29 s | 3.98 s (in-process offload) | 13.7x (blog says approximately 13x; headline '10x') | Metric table
  - p99 TTFT | 1.30 s | 13.55 s | 10.4x | Metric table
  - Mean decode speed | 37.47 tok/s | 9.81 tok/s | 3.82x | Metric table
  - p99 decode speed | 45.14 tok/s | 34.27 tok/s | 1.32x | Metric table
- Catalog / first-pass numbers checked:
  - [approx.] catalog: 13.6x mean TTFT (0.29 s vs 3.98 s): 3.98/0.29 = 13.7x from the rounded values; the blog itself writes 'approximately 13x' and titles it 10x
  - [yes] catalog: from a shared host-memory KV pool: prefix caching helps TTFT on multi-turn, in a server setting: the gain is specifically from cross-DP-rank sharing: the baseline cannot reuse a prefix cached by another rank
  - [yes] first pass: baseline vLLM in-process offload, no variance, dev build: single run, LMCache 0.4.3-dev, 2 QPS x 120 s
  - [yes] first pass: setting 8x H100, multi-round chat: also decode speed improved 3.8x because the baseline was apparently overloaded
- Transfers to our regime: **no**. Server multi-tenant prefix reuse across 8 data-parallel ranks, H100s with host RAM pools of 400 GB, comparing shared against unshared per-rank caches; the improvement removes redundant prefill that a shared cache avoids. In a single-user CPU-only laptop there is one process, so there is no cross-rank loss to recover; the part that would transfer is generic 'persist KV across turns' (our D.md session KV rows), where the saving is the whole prefill (19-22 s at 500 tokens). The blog gives no prompt lengths or hit rates, so it cannot be used to size that.
- Quality loss: None (exact KV reuse).
- Ablation: none (one A/B, shared vs process-local pool; no cache-size or hit-rate breakdown)
- Limitations (authors): Vendor blog, development build, single configuration, no variance, no hit rate, no prompt-length distribution; roadmap mentions P2P and PD disaggregation not yet available.
- Versus our own findings: No.

## Family 12 [screen 2 #12]: sparsity inside each expert (rows P-0199 to P-0203; sister rows P-0197 idea, P-0198 TEAL not in this batch)

### Further prior art found and what was read
- CATS, Contextually-Aware Thresholding for Sparsity (Lee, Lee, Zhang, Tiwari, Mirhoseini; COLM 2024, arXiv 2404.08763v4), read from raw text: threshold on the gate SiLU output only, Mistral-7B and Llama2-7B/13B, 50% activation sparsity within ~99% of base downstream performance, ~15% wall-clock latency gain with a custom kernel (abstract). Dense models, no MoE, no offloading. FloE reuses its Triton kernel and compares against it.
- TEAL, Training-Free Activation Sparsity in LLMs (Liu et al., Together AI/MIT; ICLR 2025, arXiv 2408.14690v3), read: magnitude thresholds on the hidden state input of every matrix, 40-50% model-wide sparsity, 1.53x / 1.8x decode speedup at 40% / 50% (abstract); Llama-2/3, Mistral, 7B-70B, A6000/A100; limitation text says it does not scale to higher batch and that low-batch MoE is a setting where it should be easier, but MoE is not evaluated; cites CPU-offloading extensions (Song 2023, Alizadeh 2024 'LLM in a flash').
- SmallThinker (arXiv 2507.20984), read the sparsity part: a family natively trained with ReGLU experts; median intra-expert neuron sparsity above 0.6 in deeper layers, early layers 0.9-1.0 (Sec 5.3, Fig 6), with fused sparse FFN kernels and SSD offload of experts; this is the only trained-from-scratch model we found that has the property TurboSparse needs 150B tokens to create, but it is a 4B/21B model family, not 125B-class.
- LLM in a flash (arXiv 2312.11514, Alizadeh et al.): row-column bundling and windowing for dense ReLU models from flash; found, not read in depth (used as the LLMFlash baseline in PowerInfer-2). 
- Others found via search by title, NOT read at depth here (abstract level only, so no numbers taken): HOBBIT mixed-precision expert offloading (2411.01433), FlashMoE SSD offloading with ML cache replacement (2601.17063), SeqMoE (2609.12978), Mixtral-Offloading (2312.17238), Deja Vu, PowerInfer-1 (2312.12456), EES/EEP (Efficient Expert Skipping/Pruning, compared in DualSparse Table 3).
- Not found: any paper that combines magnitude-sparse partial reads of each routed expert (neuron/column level) with expert streaming from NVMe on CPU for a post-hoc SwiGLU MoE.

### What is new in OUR variant
Our family 12 idea (zone-map block sparsity: read only the live blocks of each expert from NVMe, decided after the gate or from the cached prior token) is, as far as these papers show, closest to FloE (partial gate/down column reads chosen from |up output|, predicted one layer ahead by reusing the next layer's up matrix) and to PowerInfer-2 (neuron-bundle reads from flash). Differences: (a) FloE is GPU/PCIe from DRAM, 8 big experts; PowerInfer-2 needs a ReLU-fied model; ours is training-free on SwiGLU 512-expert top-10 with NVMe reads. (b) The new element would be block-granular (zone map) selection to keep read sizes large (FloE's compact layout and PowerInfer-2's bundles address the same problem at neuron/column granularity). Nothing in the read papers measures the dead fraction of a SwiGLU intermediate at a fixed output error for a MoE with 512 small experts; CATS/TEAL/FloE measure it for Mistral/Llama/Mixtral (50% nearly free, 80-90% costly: FloE-90% -7.6 points). Our own closed-form (E.md) predicted the same ceiling: gate-first caps bytes at 1.5x at d=0.5, and these papers' sparse-kernel speedups (1.43x at 50% in FloE Table 1, 1.54-1.55x e2e in SpQt, 1.53x/1.8x in TEAL) are consistent with that ceiling.

### Beliefs to update
1. 'SwiGLU is only 40% sparse' (TurboSparse Table 1) is about natural zeros; thresholded magnitude sparsity of 50% is nearly lossless on SwiGLU dense models (TEAL, CATS, SpQt: +0.37 ppl, TurboSparse Table 3: 28.48 -> 28.62) and costs ~1-2% on MoE (TurboSparse Fig 5a), but 80-90% costs 7+ points (FloE).
2. Every positive speed number in this family comes with: a trained/ReLU-fied model, or GPU kernels, or a RAM-resident model (FloE, TurboSparse CPU rows, SpQt) except PowerInfer-2's phone-flash result, where gains came from bundling + cache + pipeline (Fig 14), not from the sparsity alone.
3. PowerInfer-2's 11.68 tok/s is at 19 GB usable memory of a ~24 GB INT4 model; the closest to our 8 GB case is its 7 GB row, 2.13 tok/s with 1.8% of FFN cached.

## Family 13 [screen 2 #13]: typing-time chunked prefill (rows P-0205 to P-0209; sister row P-0204 idea not in this batch)

### Further prior art found and what was read
- Anthropic Claude Cookbook 'Speculative prompt caching' (page dated 2025-05-15, platform.claude.com/cookbook/misc-speculative-prompt-caching), fetched raw HTML and read: warms the cache with a 1-token (max_tokens=1) request while the user is still typing a question, then sends the real query on the warm cache, demo on two SQLite source files; the pattern is the typing-time warm of a KNOWN large context, exactly as the OpenAI prewarm. This is the closest published prior art to typing-time prefill at API level; it does not prefill the partial typed text.
- OpenAI Prompt caching guide, 'Prewarm the cache' for GPT-5.6 and later (read raw HTML): prompt_cache_options.prewarm; 1,024-token minimum.
- Cake, 'Compute or Load KV Cache? Why not both?' (Jin, Liu, Zhang, Mao; arXiv 2410.03065v2), read abstract/introduction: computes KV and loads it from storage in parallel from both ends; on average 2.6x TTFT reduction versus compute-only; deals with the same compute-vs-load trade-off as P-0207, on GPU servers.
- Speculative prefill papers (P-0208) and LMCache (P-0209) are the server side. HiCache/Dynamo/vLLM+LMCache tiers are mentioned in the photonic paper intro (not read).
- Not found: any paper or product that incrementally prefills the typed prefix of a prompt chunk by chunk during typing (as opposed to warming a fixed prefix), nor one for a streamed-expert MoE where small chunks multiply expert reads.

### What is new in OUR variant
Ours differs from every row read in two ways: (1) the prefix being prefilled is the partial text the user is typing (growing, small chunks), not a known fixed context; (2) for a streamed-expert MoE each small chunk touches an expert set and re-reads them (chunk I/O ratio 8.3x at c=16, 5.3x at c=32 in E.md), a cost the API prewarm, the cookbook and the patent do not face because their prefill cost is GPU compute. The closest published ideas are the Anthropic cookbook / OpenAI prewarm (warm before submit, fixed prefix) and SeqMoE (P-0060 in the B.md batch: chunked prefill repeats expert transfers). Our measured result (E.md: refuted for normal typing, only 1-2 tok/s typists benefit) stands unchanged by these papers.

### Beliefs to update
1. The catalog statement that no shipped speculative prefill exists is out of date: prefill-without-decode shipped in OpenAI's Responses API (GPT-5.6+) and a typing-time warm recipe is in Anthropic's public cookbook since May 2025. Novelty would have to rest on prefill of the partial typed text and the MoE-streaming angle.
2. The 50K-token crossover in P-0207 is a GPU result (8B on A100/H200). It should not be used to argue a 500-token KV reload does not pay; with CPU prefill of ~19-21 s per 500 tokens, reload wins.
3. The 13.6x LMCache TTFT is the removal of redundant prefill across 8 data-parallel ranks, so it is an upper bound on what multi-turn prefix reuse recovers when nothing is shared, not a number for one user on one machine.
