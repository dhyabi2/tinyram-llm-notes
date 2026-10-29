# Deep re-check P07a from raw text (BATCH = P07a)

Raw text read for 12 of 12 rows (0 unread). Sources: arXiv PDFs via pdftotext (T-MAC 2407.00088v2, AQLM 2401.06118v4, KBVQ-MoE 2602.11184v2, CARDAN 2609.21137v4, Tutti 2605.03375v1), GitHub API for llama.cpp PRs #4773, #4856, #6341 and the commit hashes in the catalog URLs, GitHub discussion HTML, the Microsoft blog and Hugging Face docs HTML, and PyPI JSON API for hearth-llm and diskllm (the pypi.org project pages themselves return a JS challenge). Not read: diskllm repo docs/benchmarks.md and patch, stillwarm-bench, hearth repo scripts, the semanticdiff and gitlink catalog URLs themselves (same commits verified by hash).

Machine-readable twin: deep_P07a.json (same entries, fields per BRIEF2).

## P-0140: T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge. Jianyu Wei, Shijie Cao, Ting Cao, Lingxiao Ma, Lei Wang, Yanyong Zhang, Mao Yang (USTC/UCAS/Microsoft Research). EuroSys '25 (Rotterdam, Mar 30-Apr 3 2025), doi 10.1145/3689031.3696099

- **Version read:** arXiv 2407.00088v2 (25 Mar 2025), full PDF via pdftotext (15 pp)  (raw text: True)
- **Setup:** Llama-2-7B/13B at 1/2/3/4 bit (4-bit = GPTQ Q4_0-style gguf; 2/3-bit from BitDistiller QAT; 1-bit from OneBit) and BitNet-b1.58-3B. Devices (Table 2): Apple M2-Ultra 16 cores 819.2 GB/s; Raspberry Pi 5 4xA76 17.1 GB/s; Jetson AGX Orin 12xA78AE 204.8 GB/s; Surface Book 3 i5-1035G7 4 cores 58.2 GB/s (the only x86 box). Later Table 6/7: Surface Laptop 7 (Snapdragon X Elite, 4 of 12 cores), OnePlus 12, Jetson Orin NX. Baseline llama.cpp b2794 (May 2024) dequantise kernels (and BLAS for GEMM). Batch 1 for GEMV/decode; mpGEMM test uses seq 256. E2E: 64 tokens x 20 iterations. Method: weights split into 1-bit planes, activation groups of g=4 precomputed into 16-entry tables held in one register (ARM TBL / AVX2 PSHUF), int8 table quantisation, optional fast 8-bit aggregation (FA).
- **Headline numbers (what | value | baseline | ratio | where):**
  - Kernel speedup, max / mean over llama.cpp | 6.6x max, 3.6x average | llama.cpp b2794 | as stated | Sec 1 (p.2); only stated in the intro, not in a table
  - Single-thread mpGEMV max speedup at 1/2/3/4 bit | 11.2x / 5.8x / 4.7x / 3.1x | llama.cpp dequant kernels | as stated | Sec 5.2 text, Fig 6a (p.8)
  - Multi-thread mpGEMV 2-bit speedup on 4 devices | 4.0x, 4.0x, 5.3x, 2.5x (device order as in text) | llama.cpp | as stated | Sec 5.2; paper says multi-thread is 'primarily constrained by memory bandwidth'
  - Single-thread E2E Raspberry Pi 5, M1/M2/M3 = Llama-2-7B-4bit / 7B-2bit / BitNet-3B | 2.8x / 6.7x / 5.8x | llama.cpp | as stated | Sec 5.3, Fig 8a. NB the intro (p.2) says 2.8x for the 2-bit model, which contradicts Fig 8's model order (2.8x belongs to 4-bit, 2-bit is 6.7x)
  - Multi-thread E2E M2-Ultra | 1.1x / 2.3x / 1.7x (4-bit/2-bit/BitNet); peak 71 tok/s BitNet-3B | llama.cpp | as stated | Sec 5.3, Fig 8b
  - Llama-2-7B-4bit, M2-Ultra, 1 thread: tok/s | T-MAC 7.34; T-MAC+FA 8.97 | llama.cpp 5.65; un-quantised 3.79 | 1.30x; 1.59x (computed) | Table 4 (p.11)
  - Same row, quality (WikiText2 PPL / lambada PPL / WinoGrande) | T-MAC 5.96/12.95/70.8 (identical to llama.cpp); FA 6.38/13.99/67.8 | llama.cpp 5.96/12.95/70.8; FP 5.80/12.65/71.0 | FA costs +0.42 PPL, +1.04 lambada PPL, -3.0 pt WinoGrande | Table 4; the prose says 'accuracy drops by 0.3%' but the table shows 3.0 points
  - Llama-2-7B-4bit/2bit tok/s on Surface Laptop 7 (4 cores) | T-MAC 21.63 / 31.83 | llama.cpp CPU 10.64 / 9.39; NPU 10.40 | 2.03x / 3.39x (computed) | Table 7 (p.12)
  - Same, OnePlus 12 | 10.19 / 16.62 | 8.24 / 6.95 | 1.24x / 2.39x | Table 7
  - Same, Jetson Orin NX | 7.53 / 11.41 | 3.97 / 3.20 (GPU 14.76 / 7.94) | 1.90x / 3.57x | Table 7
  - Jetson AGX Orin Llama-2-7B-2bit | T-MAC CPU 15.62 tok/s, 10.4 W | llama.cpp CPU 7.08, llama.cpp GPU 20.03 | 2.21x vs CPU; 0.78x of GPU | Table 5
  - Energy, M2-Ultra multi-thread | -20.6% / -61.2% / -51.3% energy; power -10.3% / -10.3% / -17.3% | llama.cpp | as stated | Sec 5.4, Fig 9 (abstract says 70%, intro 60-70%)
  - Optimisation breakdown: TM-base (no memory optimisation) | at most 17% slower than llama.cpp; tiling up to 1.45x, permutation 1.39x | llama.cpp | - | Sec 5.5, Fig 10
- **Catalog / first-pass numbers checked:**
  - Kernel speedup up to 6.6x -> **yes**: 6.6x max (Sec 1)
  - average 3.6x over llama.cpp dequant kernels 'at 1-2 bit' -> **no**: 3.6x average is stated for all kernels/devices vs llama.cpp (Sec 1); the paper does not restrict it to 1-2 bit. Per-bit maxima are 11.2/5.8/4.7/3.1x
  - end-to-end 2.8x for Llama-2 -> **partly**: 2.8x is in the intro for the 2-bit model, but Fig 8's order gives 2.8x to the 4-bit model on RPi5 single thread and 6.7x to 2-bit; internally inconsistent. Multi-thread M2-Ultra e2e is only 1.1-2.3x
  - first-pass: Table 4 5.65 -> 7.34 tok/s -> **yes**: 5.65 -> 7.34 (1.30x), same PPL as llama.cpp
  - first-pass: Table 7 Surface Laptop 7 21.63 vs 10.64 at 4 bit -> **yes**: 21.63 vs 10.64 = 2.03x; the 2-bit row is 31.83 vs 9.39 = 3.39x
  - first-pass: M2-Ultra multithread e2e 1.1-2.3x -> **yes**: 1.1x, 2.3x, 1.7x
  - first-pass: 'Baseline is llama.cpp dequantise+FMA kernels' -> **yes**: llama.cpp b2794, May 2024; T-MAC's own text says llama.cpp gains nothing below 4 bit and 3-bit is 15% slower than 4-bit
- **Transfers to our regime:** no. Assumptions that break: (1) baseline is a 2024 dequantise+FMA kernel at 1-3 bit, while our baseline is a tuned int8 dot on Q4-class weights; the only 4-bit comparison on a like-for-like weight file (Table 4, same Q4_0 model) gives 1.30x single-thread and the paper says multi-thread is bandwidth-limited. (2) T-MAC tables are bit-serial (activation groups of 4 against 1-bit weight planes, pshufb/tbl in-register), not 8-dim VQ codebook gathers, so it does not test our gather LUT (our own gather kernel 0.06-0.17x of int8 dot is not contradicted). (3) Only one x86 device (4-core i5-1035G7, 58 GB/s), no AVX2 number is broken out. (4) Weights are in RAM; no NVMe streaming; in our regime at 8 GB the expert stream is the bottleneck, so a compute-kernel gain only matters at 16/32 GB hit rates. (5) 2/3-bit models need QAT (BitDistiller/OneBit); 4-bit needs nothing.
- **Quality loss:** None for the table-lookup itself at 4-bit (identical PPL 5.96 to llama.cpp Q4_0, Table 4); Table 3 NMSE 3.35e-3 vs 3.33e-3 llama.cpp for 4096x4096. Fast aggregation: +0.42 wiki PPL, +1.04 lambada PPL, WinoGrande 70.8 -> 67.8. Low-bit models' own quality (2-bit via BitDistiller) is not evaluated by this paper.
- **Ablation:** Yes (Sec 5.5, Fig 10): TM-base up to 17% slower than llama.cpp; table quantisation brings parity; tiling up to 1.45x; permutation 1.39x; tuning small. Isolates kernel design only; no cache/layout/prefetch notion.
- **Limitations stated:** Authors say multi-thread mpGEMV is constrained by memory bandwidth and e2e gains shrink due to non-GEMV operators; FA hurts quality; M2-Ultra GEMM loses to AMX BLAS (T-MAC max 2.0x at 1-bit). No dedicated limitations section.
- **Contradicts our findings?** No. It supports our 'bytes half only' reading: gains vanish when memory-bound (multi-thread e2e 1.1-2.3x on M2-Ultra; 4-bit Table 4 only 1.3x). Our gather-LUT result (0.06-0.17x of int8 dot) is for a different table design and is neither confirmed nor refuted. Paper-internal inconsistency (2.8x) noted above.

## P-0141: Advances to low-bit quantization enable LLMs on edge devices. Shijie Cao, Lingxiao Ma, Ting Cao. Microsoft Research Blog, published 5 Feb 2025

- **Version read:** https://www.microsoft.com/en-us/research/blog/advances-to-low-bit-quantization-enable-llms-on-edge-devices/ (HTML, tags stripped); catalog URL was truncated, this is the full address  (raw text: True)
- **Setup:** Blog summarising Ladder, T-MAC and LUT Tensor Core. T-MAC numbers: Surface Laptop 7 (Snapdragon X Elite); BitNet-b1.58 3B, 2-bit and 4-bit 7B Llama; Raspberry Pi 5 for BitNet 3B. No thread count, no baseline numbers, no table; refers to the T-MAC paper.
- **Headline numbers (what | value | baseline | ratio | where):**
  - Surface Laptop 7 BitNet-b1.58 3B | 48 tok/s | llama.cpp (not given) | '4-5x' (text) | blog, T-MAC section
  - Surface Laptop 7 2-bit 7B Llama | 30 tok/s | not given | '4-5x' vs llama.cpp (text) | same
  - Surface Laptop 7 4-bit 7B Llama | 20 tok/s | not given; paper Table 7 llama.cpp CPU = 10.64 | paper ratio 2.03x for 4-bit, so '4-5x' is not the 4-bit ratio (computed) | same; compare T-MAC paper Table 7 (21.63 vs 10.64)
  - vs NPU | 'doubling the speed of a dedicated NPU' | NPU 10.40 tok/s (paper Table 7) | 31.83/10.40 = 3.06x (2-bit), 2.08x (4-bit) per paper | blog; paper Table 7
  - Raspberry Pi 5 BitNet 3B | 11 tok/s, using 1/4 to 1/6 of the CPU cores for the same rate as llama.cpp | - | - | blog
  - LUT Tensor Core (hardware, different work) | 6.93x speed, 38.3% area, 20.9x density, 11.2x energy | traditional Tensor Core | - | blog
- **Catalog / first-pass numbers checked:**
  - Llama-2-7B W2 9.3 tok/s on 1 thread -> **no**: absent from the blog. 9.39 appears only in the T-MAC paper Table 7 as the llama.cpp CPU baseline for 2-bit on Surface Laptop 7, i.e. it is the baseline, not T-MAC
  - 28.4 tok/s on 4 threads (W2) -> **no**: absent from blog and from the T-MAC paper text; blog gives 30 tok/s, paper Table 7 gives 31.83 (4 cores)
  - W4 18.7 tok/s on 4 threads -> **no**: absent; blog gives 20 tok/s, paper Table 7 gives 21.63
  - LUT kernels exist but use bit-serial tables, not VQ codebooks -> **yes**: consistent with the T-MAC paper (g=4 bit-pattern tables)
  - first-pass: 48 / 30 / 20 tok/s, 4-5x llama.cpp, ~2x NPU -> **yes**: exactly these blog statements
  - first-pass: 'thread counts not stated' -> **yes**: blog gives no thread count; paper Table 6 says 4 cores for Surface Laptop 7
- **Transfers to our regime:** no. Marketing summary of the T-MAC paper; ARM Snapdragon NEON TBL, 1-4 bit scalar weights vs llama.cpp dequant baseline, weights in RAM, no x86 AVX2 data, no MoE, no NVMe. Same reasons as P-0140. The '4-5x' headline is a 2-bit figure; the paper's own 4-bit ratio on this device is 2.03x.
- **Quality loss:** Not discussed in the blog (links BitDistiller, BitNet for the models).
- **Ablation:** none
- **Limitations stated:** None stated.
- **Contradicts our findings?** No. Note the catalog's tok/s values are wrong; do not cite 9.3/28.4/18.7.

## P-0142: Extreme Compression of Large Language Models via Additive Quantization (AQLM). Vage Egiazarian*, Andrei Panferov*, Denis Kuznedelev, Elias Frantar, Artem Babenko, Dan Alistarh. ICML 2024

- **Version read:** arXiv 2401.06118v4 (11 Sep 2024), full PDF via pdftotext (first pass could not read the PDF)  (raw text: True)
- **Setup:** Llama-2 7B/13B/70B, Mistral-7B, Mixtral-8x7B (46.7B total, 8 experts, top-2). Quantisation on A100/H100 (1-8 GPUs), calibration RedPajama 8M tokens. Speed (App. C): GPU RTX 3090 24 GB, CPU Intel Core i9-13900K, '8 cores'; Table 5 = FP16/FP32 gate_proj matrix-vector in PyTorch; Table 14 = 128 generated tokens, batch size 1, compiled graph. CPU kernel only for codebook sets of 8-bit codebooks (2x8, 4x8, 8x8, group size 8); the paper text does not mention Numba. Baseline for CPU = FP32 PyTorch.
- **Headline numbers (what | value | baseline | ratio | where):**
  - CPU gate_proj GEMV speedup, 2x8 | x2.75 (7B), x3.54 (13B), x3.69 (70B) | FP32: 1.83 / 3.12 / 11.31 ms | as stated | Table 5 (p.9)
  - CPU 4x8 / 8x8 | x2.55/3.02/4.07 ; x2.29/2.68/4.03 | FP32 | as stated | Table 5
  - GPU gate_proj, 2x8 and 1x16(Table-1 models) | x1.57/1.82/3.05 ; x1.31/1.20/1.20 | FP16 129/190/578 us | as stated | Table 5
  - End-to-end CPU tok/s, 2x8 | 6.961 (7B), 4.180 (13B), 0.966 (70B) | FP32 3.106 / 1.596 / 0.297 | 2.24x / 2.62x / 3.25x (computed) | Table 14, App. I
  - End-to-end CPU tok/s, 4x8 and 8x8 (7B) | 6.837 and 5.319 | FP32 3.106 | 2.20x and 1.71x | Table 14
  - End-to-end GPU 3090 tok/s, 2x8 | 114.1 / 68.1 / 14.3 | FP16 54.2 / 29.5 / 5.8 | 2.1x / 2.3x / 2.5x | Table 14
  - Effective weight bandwidth, my derivation for 7B gate_proj (4096x11008 = 45.1M weights, shape from memory of Llama-2) | FP32 180 MB in 1.83 ms = 98 GB/s; AQLM 2x8 ~11.3 MB in 0.665 ms = 17 GB/s | - | kernel moves 6x fewer bytes/s than the baseline, i.e. it is compute/lookup-bound, not bandwidth-bound | derived from Table 5
  - Quality, 2.02-bit 1x16 (not the CPU-kernel config), Llama-2-7B Wiki2 PPL / avg 5-task acc | 6.59 / 57.28 | FP16 5.12 / 62.35 | +1.47 PPL, -5.07 pt | Table 1
  - Quality, 70B 2.07-bit | 3.94 / 68.75 | FP16 3.12 / 70.17 | +0.82 PPL, -1.42 pt | Table 1
  - Quality of the CPU-kernel config 2x8 (2.0 bit), 7B: PPL without / with end-to-end fine-tune | 7.61 / 6.57 (acc 54.32 / 56.19) | FP16 5.12 (62.24) | +2.49 / +1.45 PPL | Table 12 (App. F)
  - 2x8 13B / 70B (fine-tuned) | 5.63 / 4.21 | FP16 4.57 / 3.12 | +1.06 / +1.09 | Table 12
  - Mixtral-8x7B 1.98 bit (1x16-type) | Wiki2 4.61, avg acc 67.68 | FP16 3.46, 72.33 | +1.15 PPL, -4.65 pt | Table 3
  - Quantisation cost | 7B ~1 day on 1 A100; 70B 10-14 days on 1 GPU or 3-4 days on 8; fine-tune 7B 3-6 h on 4 A100, 70B 1-2 days on 8 A100 | - | - | App. D
- **Catalog / first-pass numbers checked:**
  - 2x8 codebooks on an 8-core i9: matvec 2.75x (7B), 3.54x (13B), 3.69x (70B) faster than FP32 -> **yes**: Table 5 exactly; the baseline is a PyTorch FP32 single-layer matvec
  - up to about 4x -> **yes**: 4.07x (4x8 config, 70B)
  - against FP32, not an int8/Q4 kernel -> **yes**: baseline is Original (float32); Table 14 e2e also vs float32
  - first-pass: Table 1 2-bit Wiki2 PPL 6.59 (7B), 3.94 (70B) -> **partly**: values are right but belong to the 1x16-bit-codebook models (2.02/2.07 bit) that have only a GPU kernel; the CPU-kernel 2x8 config has PPL 7.61 (no tuning) / 6.57 (7B), 4.21 (70B, fine-tuned)
  - first-pass: GPU 2x8 x1.57/1.82/3.05 vs FP16 -> **yes**: Table 5
  - first-pass: 'paper text does not say Numba' -> **yes**: grep finds no Numba in the paper
  - first-pass: 'PDF unreadable' -> **no**: PDF is readable with pdftotext -layout
- **Transfers to our regime:** no. (1) Baseline FP32: 16x more bytes than the 2-bit codes yet only 2.75-4x faster, so against a Q4 int8 kernel that already runs at memory speed the ratio is <=1 (the AQLM kernel is not bandwidth-bound; derived 17 GB/s of codes). (2) Needs expensive post-training with fine-tuning (days of A100) that we cannot do for a 125B MoE. (3) Codebooks are per-matrix learned, not shared across 512 experts; Mixtral was only done at 1x16 with GPU kernels. (4) At 2 bit quality loss is material (7B -5 pt avg acc). (5) Weights in RAM; no NVMe. The byte reduction (2.0 vs 4.3 bpw) is the part that could transfer and equals what IQ2 already ships.
- **Quality loss:** See numbers: 7B 2x8 +1.45 PPL after fine-tune (+2.49 before); 2-bit Mixtral -4.65 pt avg acc, Wiki2 3.46 -> 4.61. MMLU/GSM8k relative degradation 'higher than on standard evals' (App. K).
- **Ablation:** Yes: fine-tuning scope (Table 7), codebook count/group (Table 9: 2x8gs8 7.61, 4x8gs16 8.14, 8x8gs32 7.38, 15x8gs64 7.85 untuned), calibration size (Table 8). No cache/layout notion.
- **Limitations stated:** Authors: more expensive to quantise than RTN/GPTQ; smaller 8-bit codebooks (needed for fast CPU/GPU kernels) raise quantisation error; better fine-tuning left for future work.
- **Contradicts our findings?** No. Consistent with our reading that published LUT-kernel gains are vs FP32/FP16. Additionally shows the AQLM CPU kernel is lookup/compute-bound (6x lower byte rate than baseline), the same direction as our gather-LUT result. Note the AQLM kernel itself was not run by us.

## P-0143: Hugging Face Transformers documentation, 'AQLM' quantization page (v4.57.2)

- **Version read:** https://huggingface.co/docs/transformers/v4.57.2/quantization/aqlm HTML, tags stripped  (raw text: True)
- **Setup:** Docs table of AQLM inference kernels: Triton (KxN), CUDA 1x16, CUDA 2x8, Numba Kx8. States install `pip install aqlm[gpu,cpu]` (Python 3.10+). No hardware, model, baseline or batch size given.
- **Headline numbers (what | value | baseline | ratio | where):**
  - Triton KxN kernel speedup | up to ~0.7x | not stated | - | docs table
  - CUDA 1x16 (accuracy 'Best') | up to ~1.3x | not stated | - | docs table; matches paper Table 5 x1.31 vs FP16
  - CUDA 2x8 (accuracy 'OK') | up to ~3.0x | not stated | - | docs table; matches Table 5 x3.05 vs FP16
  - Numba Kx8 (accuracy 'Good', fast CPU inference) | up to ~4.0x | not stated | - | docs table; matches Table 5 x4.07 vs FP32 (4x8, 70B)
- **Catalog / first-pass numbers checked:**
  - CPU path is a Numba kernel, up to about 4.0x -> **yes**: Numba, 'Kx8', up to ~4.0x
  - 2x8 CUDA kernel about 3.0x -> **yes**: ~3.0x
  - Not in llama.cpp -> **yes (by omission)**: the page documents the transformers/aqlm path only
  - first-pass: Triton ~0.7x, CUDA 1x16 ~1.3x -> **yes**: as listed; 0.7x has no counterpart in the paper (unsourced)
  - first-pass: 'speedups relative to the unquantised model' -> **absent**: the docs do not state the baseline; the numbers equal the paper's Table 5 maxima vs FP32 (CPU) and FP16 (GPU), so that reading is an inference
- **Transfers to our regime:** no. Same source numbers as P-0142 restated without setup; Python/Numba kernel, FP32 baseline, no llama.cpp.
- **Quality loss:** Column 'Accuracy' only (Best/OK/Good); no numbers.
- **Ablation:** none
- **Limitations stated:** None stated.
- **Contradicts our findings?** No.

## P-0144: llama.cpp PR #4773 'SOTA 2-bit quants' (IQ2_XXS), Kawrakow (ikawrakow); opened 4 Jan 2024, merged 8 Jan 2024 as commit dd5ae06405c5565b99889bdb3f168f4351252cfb (the commit in the catalog URL); follow-up PR #4856 'part 2' (IQ2_XS) merged 11 Jan 2024

- **Version read:** GitHub API: pulls/4773 body, all 41 issue comments, commit dd5ae06 message; pulls/4856 body. The catalog semanticdiff URL itself was not opened (first pass got HTTP 429); the same commit hash was read through the GitHub API  (raw text: True)
- **Setup:** LLaMA-v2 7B/13B/70B, Mistral-7B, Mixtral-8x7B. Hardware for speed: RTX 4080 (CUDA), 30-core M2 Max (Metal), Ryzen 5975WX CPU; metric TG-128 for a 7B LLaMA. Format IQ2_XXS: 2.0625 bpw = per 8 weights an 8-bit index into a fixed 256-point E8-lattice grid + 7 sign bits, 4-bit scale per 32, fp16 super-block scale per 256 (QuIP#-derived, no incoherence processing); quantised with an importance matrix (code comment: without it 'total garbage'). IQ2_XS (PR 4856): 512 grid points, blocks of 16, 2.31 bpw. PPL at context 4096.
- **Headline numbers (what | value | baseline | ratio | where):**
  - TG-128 7B CPU Ryzen 5975WX | 24.3 tok/s | Q4_0 15.4 tok/s | 1.58x (computed) | PR 4773 body TL;DR
  - TG-128 7B Metal M2 Max | 54 tok/s | Q4_0 63.5 | 0.85x | same
  - TG-128 7B CUDA RTX 4080 | 155 tok/s | Q4_0 130 | 1.19x | same (and commit log 155.1)
  - ARM NEON dot product at the time of the PR | 'strangely slow (112 ms/token)' = 8.9 tok/s | - | - | commit message dd5ae06
  - PPL IQ2_XXS (file GiB): LLaMA-v2-7B / 13B / 70B / Mistral-7B | 7.067 (1.728) / 5.728 (3.295) / 4.079 (17.03) / 6.446 (1.855) | no FP16 or Q2_K column in the PR | - | PR 4773 table
  - PPL IQ2_XS (2.31 bpw) | 6.308 (7B, 1.895 GiB), 5.288 (13B), 3.783 (70B, 18.94), Mistral 5.822 | - | - | PR 4856 table
  - Mixtral-8x7B (a MoE) IQ2_XXS / IQ2_XS | PPL 4.948 (11.44 GiB) / 4.514 (12.73 GiB) | - | - | PR 4856 table
  - Mixtral on 16 GB GPU | 2.10 bpw model fully fits at ctx 4096, ~29 tok/s | - | - | user comment he29-net, 8 Jan 2024
- **Catalog / first-pass numbers checked:**
  - Codebook-indexed 2-bit quants already ship -> **yes**: IQ2_XXS merged 8 Jan 2024
  - CPU TG-128 21.4 tok/s (AVX2) vs 19.5 (NEON) on a 7B-class model -> **no**: not in the PR body, the commit message, any of the 41 comments or PR 4856. PR reports 24.3 tok/s on Ryzen 5975WX (AVX2 path) vs Q4_0 15.4; NEON was 8.9 tok/s at merge
  - 'Grid lookup then int8 dot, not LUT-gather accumulate' -> **yes**: codebook grid indices are expanded to int8 magnitudes/signs, then dotted with Q8_K activations; confirmed by commit log ('AVX2 dot products ... Q8_K quants in -127..127')
  - first-pass: 2.0625 bpw, PPL 7.067 / 4.079, 24.3 vs 15.4, M2 Max 54 vs 63.5 -> **yes**: as above
  - first-pass: catalog URL not opened -> **yes**: same commit hash verified through API
- **Transfers to our regime:** partly. Directly shipping and runs in llama.cpp on CPU, with MoE (Mixtral) working, so the byte lever (2.06-2.31 bpw vs 4.3 bpw) is real and available now, which is why the 'bytes' half of an 8-dim codebook idea is known. The CPU speedup over Q4_0 (1.58x, one 24-core Ryzen) is a memory-bandwidth effect of fewer bytes and says nothing about LUT-accumulate compute. Breaks: model-level quality loss is not quantified against FP16 in the PR; needs an importance matrix; weights are in RAM (no NVMe streaming); quality at 2 bit is poor for small models (7B PPL 7.07 at ctx 4096; no baseline in the PR).
- **Quality loss:** PPL table above; no FP16/Q4 reference in the PR. Commenter Ttl (9 Jan 2024): 2.20 bpw model's top token missing from FP16's top-10 in 2.8% of positions vs 0.85% for Q2_K, 0.10% for Q4_K_S.
- **Ablation:** Partial: IQ2_XXS vs IQ2_XS bit budget (grid size 256 vs 512, block 32 vs 16) in PR 4856: PPL 7.067 -> 6.308 for +0.25 bpw.
- **Limitations stated:** Author: quantiser not included in the PR; not all backends; CUDA MMQ not implemented; performance 'reasonable but not great'; imatrix mandatory.
- **Contradicts our findings?** No. Confirms pubtech D line 84 (shipping codebook-grid quants with the byte lever and int8 dot). Our D.md already notes the catalog's 21.4/19.5 cannot be sourced; now confirmed absent from every PR artifact.

## P-0145: KBVQ-MoE: KLT-Guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models. Zukang Xu, Zhixiong Zhao, Xing Hu, Zhixuan Chen, Dawei Yang (Houmo AI; Nanyang Technological University). 'Published as a conference paper at ICLR 2026'

- **Version read:** arXiv 2602.11184v2 (24 Feb 2026; v1 submitted 30 Jan 2026), full PDF via pdftotext. First pass read v1 via digest and could not confirm the venue; v2 header confirms ICLR 2026  (raw text: True)
- **Setup:** Qwen1.5-MoE-A2.7B (14.3B total, 60 experts, from memory), Qwen3-30B-A3B (128 experts, top-8), Mixtral-8x7B, DeepSeek-V2-Lite. Method: KLT-guided SVD pulls shared low-rank part across experts (rank = 1/128 of full; about 0.08 extra bits per weight, 2.08 effective bits for the 2-bit setting), VQ with vector length 4 (not 8) on the residual per expert (k-means++, 100 iters), channel-wise affine bias correction. Calibration 256 x 4096 tokens RedPajama. Hardware: RTX A6000 (Sec 5), appendix A.10 says 'A100'. Baselines RTN, GPTQ, MoEQuant, plain VQ. Eval: 7 zero-shot tasks, LM-Eval-Harness. Speed test (Table 6): 1k input tokens, BF16 vs quantised, GPU, no kernel description.
- **Headline numbers (what | value | baseline | ratio | where):**
  - Qwen1.5-MoE-A2.7B 3-bit avg acc / W2 PPL | 67.99 / 7.74 | FP16 68.07 / 7.22; GPTQ 66.36 / 7.58 | -0.08 pt vs FP16; +1.63 vs GPTQ | Table 1 / Table 10
  - Mixtral-8x7B 3-bit | 78.35 / 4.07 | FP16 78.57 / 3.88; GPTQ 77.43 / 4.17 | -0.22 pt | Table 1 / 12
  - Qwen3-30B-A3B 3-bit | 69.09 / 9.26 | FP16 70.24 / 8.70; GPTQ 63.51 / 11.32 | -1.15 pt | Table 1 / 11
  - DeepSeek-V2-Lite 3-bit | 68.73 / 6.95 | FP16 70.68 / 5.92; GPTQ 68.89 / 6.98 | -1.95 pt; GPTQ is 0.16 pt better on avg acc | Table 1 / 13
  - 2-bit (2.08 effective) avg acc: Qwen1.5 / Qwen3 / Mixtral / DS-V2-Lite | 62.78 / 63.37 / 75.69 / 63.10 | FP16 68.07 / 70.24 / 78.57 / 70.68 | -5.29 / -6.87 / -2.88 / -7.58 pt | Table 1
  - Qwen3-30B-A3B 2-bit PPL | 11.87 | FP16 8.70; GPTQ 438.42 | +3.17 | Table 1
  - Decoder speed, 1k input tokens | Qwen1.5-MoE 35.24; Qwen3-30B 17.37 tok/s | BF16 22.31; 10.85 | 1.58x; 1.60x (computed) | Table 6 (p.10)
  - Ablation Qwen3 3-bit W2 PPL: none / IDRE only / BCOS only / both | 18.72 / 11.67 / 14.32 / 9.26 | FP16 8.70 | - | Table 2
  - Rank ablation Qwen3 2-bit: k/n 0 / 1/128 / 1/64 / 1/32 | 15.30 (2.01 b) / 11.87 (2.08) / 11.30 (2.11) / 11.01 (2.20) | - | - | Table 4
  - MMLU/GSM8k at 2-bit Qwen3-30B | MMLU 68.19, GSM8K 78.92, HumanEval 66.89 | FP16 79.5, 85.44, 71.5 | -11.3 / -6.5 / -4.6 pt | Table 17
- **Catalog / first-pass numbers checked:**
  - 3-bit VQ of Qwen1.5-MoE-A2.7B scores 67.99 vs FP16 68.07 -> **yes**: Table 1/10
  - VQ on MoE viable for quality at 3 bit -> **yes**: within 0.08-1.95 pt on four models
  - No CPU kernel -> **yes**: speed claim is GPU only and gives no kernel description
  - ICLR 2026 -> **yes**: v2 header says 'Published as a conference paper at ICLR 2026'
  - first-pass: Qwen3-30B 2-bit 63.37 vs 70.24, PPL 11.87 vs 8.70 -> **yes**: Table 1
  - first-pass: 'at 2 bit GPTQ beats it on several models (Qwen1.5 66.36 vs 62.78)' -> **no**: 66.36 is GPTQ at 3 bit; GPTQ at 2 bit is 49.07 on Qwen1.5 vs KBVQ 62.78. At 2 bit KBVQ beats GPTQ everywhere. At 3 bit GPTQ beats KBVQ only on DeepSeek-V2-Lite (68.89 vs 68.73 avg acc)
  - first-pass: Table 6 22.31 -> 35.24 (1.58x), 10.85 -> 17.37 (1.60x) -> **yes**: baseline is BF16, not an int8/Q4 kernel
  - first-pass: ~2.08 effective bits; RTX A6000 (appendix says A100) -> **yes**: Table 4/8; A.10 says 'NVIDIA RTX A100'
- **Transfers to our regime:** no. GPU, speed baseline BF16 (decode on GPU is not our CPU/NVMe regime); vector length 4 codebook plus shared SVD part kept in higher precision, quantiser needs calibration and fine-tuning-free but non-trivial Hessian-free k-means pipeline; quality is only acceptable at 3 bit (~3.1 effective bits incl. overhead is not 2); models are 14-47B, not 125B/512 experts/top-10. No decode kernel is described, so nothing about LUT vs dequant cost.
- **Quality loss:** 3-bit: -0.08 to -1.95 pt avg acc; 2-bit: -2.9 to -7.6 pt (4 models), MMLU -11 pt on Qwen3-30B (Table 17). 2.08 bit is competitive but not near-lossless.
- **Ablation:** Yes: IDRE vs BCOS (Table 2), KLT vs plain SVD (Table 3), rank (Table 4), plugged into GPTVQ/VPTQ (Table 5). Nothing on cache/layout/prefetch or kernels.
- **Limitations stated:** Authors: SVD rank chosen empirically; decoder-only MoE only; 1-bit / 1.5-bit untested. Efficiency: states affine correction <0.1% of expert FLOPs.
- **Contradicts our findings?** No. Supports D.md line 349/359: ~2 bpw plain VQ is the break-even; KBVQ gets to 2.08 b with shared-SVD + bias-correction and still loses 3-8 pt. Our probe (plain 8-dim, 256 words, 1.0 bpw rel. error 0.58) is a weaker quantiser (vector length 8 vs their 4).

## P-0146: A Multi-Engine Dataflow for MoE Decoding on AWS Tranium [sic, in the title; Trainium in the text] (CARDAN). Bin Ma (UC Merced), Wenjie Fan (Yotta Labs), Jonny Liu (Yotta Labs), Dong Li (UC Merced / Yotta Labs)

- **Version read:** arXiv 2609.21137v4 (20 Oct 2026; v1 17 Sep 2026), full PDF via pdftotext -layout and raw mode. First pass could only read the abstract  (raw text: True)
- **Setup:** AWS Trainium3 (8 NeuronCore-v4, 144 GiB HBM at 4.7 TB/s, 64 MiB SBUF per logical core), TP=4, batch-1 TPOT. Models (Table 2): Qwen3-30B-A3B (48 layers, 128 experts, top-8), DeepSeek-V2-Lite (64, top-6), Nemotron-3-Nano-30B (128, top-6), OLMoE-1B-7B (64, top-8), Gemma-4-26B-A4B (128, top-8). Baselines: PyTorch/XLA and AWS BF16 dense MoE megakernel (production AWS/NxDI for Qwen3 and Gemma, ported design for the others). Representation: gate and up matrices only = per-layer shared VQ codebook (K=256, uint8 index maps per expert, block d=8 or 4, looked up by GpSimd 'indexed selection') + shared-basis low-rank (rank r, per-expert coefficients); down projection stays BF16. Initialised closed-form, then knowledge distillation (10,000 x 256-token windows). PPL on WikiText-103, 32 x 512 tokens.
- **Headline numbers (what | value | baseline | ratio | where):**
  - Batch-1 decode speedup vs dense megakernel, 5 models | 1.15-1.31x (geomean 1.24x); Qwen3 1.25x | dense megakernel | as stated | Fig 9, Sec 5.2; abstract 'matches or improves BF16-teacher PPL'
  - vs PyTorch/XLA | 1.09-2.07x (geomean 1.47x) | XLA | as stated | Fig 9
  - Qwen3-30B-A3B TPOT | 6.278 ms (CARDAN) ; dense 7.851 ms | dense BF16 | 1.25x (computed) | Table 4A
  - Qwen3 attribution | VQ+LR per-expert shared proj 6.90; +shared-proj reuse 6.56; +cross-expert gather batching 6.28 | dense 7.85 | 12.1%, +4.9%, +4.3% => 20.0% cumulative | Table 4A
  - Fusion on/off, dense->CARDAN ms/token | DS-V2-Lite 5.48->4.95, 4.82->4.28; Nemotron 6.94->6.25, 6.73->6.03 | dense | 1.11x, 1.13x, 1.11x, 1.12x (computed) | Table 4B
  - HBM traffic | MoE-projection HBM reads -26 to -50%; DMA-active time -10 to -31%; total HBM reads -10 to -32% | dense | as stated | Fig 10a, Sec 5.3
  - Batch scaling OLMoE-1B-7B | 1.31x (B1) -> 1.67x (B8) -> 1.70x (B16); HBM reads 3.33 vs 7.02 GB (B8), 6.13 vs 13.56 GB (B16) | dense | as stated | Sec 5.6, Fig 12
  - Compression of gate/up | 3.92x-4.92x (4.06 bits/weight for Qwen3 (8,384)) | BF16 | as stated | Table 5/6
  - Quality, PPL teacher -> closed-form -> KD | Qwen3 9.82 -> 12.51 -> 9.56; DS-V2-Lite 8.37 -> 12.49 -> 8.22; Nemotron 8.55 -> 9.66 -> 8.29; OLMoE 8.33 -> 9.11 -> 8.24; Gemma 10.98 -> 13.37 -> 10.44 | BF16 teacher | KD PPL -1.1% to -4.9% (better than teacher) | Table 5
  - Ablation: VQ-only d=8 (1.00 b) / d=4 (2.00 b) / d=2 (4.00 b) / LR-only r=512 (4.08 b), no training | PPL 451.53 / 22.47 / 10.44 / 34.91 | teacher 9.82 | +4496% / +129% / +6.3% / +255% | Table 6
  - Equal-rate VQ-only d=2 full 48-layer model | 8.075 ms/token | CARDAN 6.278; dense 7.851 | 28.6% slower than CARDAN, 2.9% slower than dense | Sec 5.5; 'GpSimd selection time exceeds overlapping DMA time'
  - Gemma-4 equal-rate VQ-only (2,0) vs CARDAN (8,512) | TPOT 12.162 vs 10.571 ms; VQ-only reads 3.9% fewer bytes, GpSimd time +199% | dense 11.48 | VQ-only 6% slower than dense | Table 3, Sec 5.3
  - Built-in W4A8 (MXFP4 gate/up) on Qwen3 | 11.75 ms/token | dense BF16 7.85 | 1.50x slower | Sec 5.5
  - Downstream accuracy | Qwen3-30B-A3B loses at most 3.2 pt on four zero-shot tasks | BF16 | - | Sec 5.7, App. A.5
- **Catalog / first-pass numbers checked:**
  - Catalog title: 'Shared-codebook VQ with single table lookup for MoE experts on tensor accelerators' -> **partly**: paper is VQ (per-layer shared codebook, one lookup per block) PLUS a shared-basis low-rank part; descriptions of 'single table lookup' relate to Sec 2.3 sentence about VQ. Accelerator = Trainium3, so 'tensor accelerators' correct
  - Blockwise VQ of expert matrices with lookup, aimed at accelerators, not CPU -> **yes**: no CPU; on gate/up only, down projection stays BF16
  - first-pass summary_accuracy 'wrong' -> **partly wrong**: the first pass said the abstract has no codebook-lookup language; the body (Sec 2.3, 3.1) does describe weight code books, index maps and indexed selections. The abstract says 'vector-quantized component'
  - first-pass: 1.15-1.31x batch-1, up to 1.7x at batch 16 -> **yes**: 1.7x is the OLMoE-only batch-16 result
  - first-pass: 'matches or improves BF16-teacher PPL on all five' -> **yes, with training**: requires knowledge distillation; closed-form init alone loses +27% PPL on Qwen3 (Table 6)
- **Transfers to our regime:** no. Accelerator with software-managed scratchpad and DMA engines (the bottleneck is HBM DMA at 4.7 TB/s, not NVMe at ~3-7 GB/s, i.e. ~1000x different), tensor-parallel 4-way, BF16 weights, 30B-class models with 64-128 experts, training/distillation required, gate/up only. The result that does transfer: lookups are not free. When the block size shrinks to d=2 to keep quality, the extra indexed selections (GpSimd) made decode slower than the dense BF16 baseline despite 12% fewer bytes. That is the same trade our gather-LUT microbenchmark found on CPU.
- **Quality loss:** With distillation PPL is at or below the BF16 teacher on all five (-1.1% to -4.9%); without training (closed form) +27% PPL on Qwen3, +49% DS-V2-Lite; VQ-only d=8 collapses (PPL 451). Downstream: Qwen3 loses up to 3.2 pt on four zero-shot tasks.
- **Ablation:** Yes, the most useful of the batch: representation (VQ-only vs LR-only vs hybrid, Table 6), kernel mechanisms (Table 4A), fusion (4B), d=2 equal-rate lookup cost (Table 3, Sec 5.5), batch scaling. No cache or prefetch ablation (it argues prefetch is awkward on a scratchpad).
- **Limitations stated:** No dedicated limitations section found; boundary conditions stated in Sec 5.5 (d=2 is slower, K fixed at 256 for uint8 index, only gate/up compressed). Related: MoBE (arXiv 2508.05257) for the shared basis, cited as ref [5].
- **Contradicts our findings?** No. Independent support for our refutation of the compute half (lookup work cancels byte savings once blocks are small), and the closest published combination of our families 2 and 5 (shared codebook + shared low-rank basis across experts) - but gains are 1.15-1.31x at batch 1 on an HBM accelerator.

## P-0148: llama.cpp Discussion #18244 'I have a 950k prompt, how do I not reprocess it every time?' (General). Started 21 Dec 2025 by jerkstorecaller; replies by abc-nix (21 Dec 2025) and vimalnakrani08 (6 Jul 2026)

- **Version read:** https://github.com/ggml-org/llama.cpp/discussions/18244 (HTML, tags stripped; 3 replies + 1 comment read)  (raw text: True)
- **Setup:** Not a benchmark. Poster wants to reuse a 950k-token prompt on Nemotron-3-Nano-30B-A3B (1M context, a MoE) in llama.cpp; says processing would take several hours. Replies: llama-cli --prompt-cache FILE then --prompt-cache-ro; llama-server --slot-save-path DIR plus POST /slots/<n>?action=save|restore with {filename}. abc-nix uses GLM-4.6 at 64k ctx whose KV file is 36 GB. July 2026 reply: build b9871 on an M3 Max, '46-cell portability matrix' (data at a third-party repo, not read).
- **Headline numbers (what | value | baseline | ratio | where):**
  - KV file size, GLM-4.6 64k context | 36 GB | - | ~0.55 MB/token (computed) | abc-nix reply
  - Cross-build restore | files from a build ~5 weeks older restored fine in both directions | - | - | vimalnakrani08 reply (b9871, M3 Max)
  - Flash-attention on/off mismatch | restore refused; KV type also part of the match; changing -ngl still worked | - | - | same
  - Time for 950k tokens | 'several hours' (author's estimate) | - | - | original post
- **Catalog / first-pass numbers checked:**
  - Stock llama.cpp has llama-cli --prompt-cache / --prompt-cache-ro -> **yes**: both named in the post
  - llama-server --slot-save-path with POST /slots/<id>?action=save|restore -> **yes**: abc-nix scripts
  - Automatic ... (truncated in catalog) -> **absent**: the thread says llama-server has nothing automatic; users script it
  - first-pass: 36 GB for 64k GLM-4.6; b9871 M3 Max; restore may silently re-prefill, check prompt_n -> **yes**: matches the thread
- **Transfers to our regime:** yes. Mechanism and file size info apply to any local llama.cpp; applies as a TTFT-only feature: decode tok/s unchanged. Caveats: restore can silently re-prefill (SWA models without --swa-full, check prompt_n); no format guarantee across builds; FA/KV-type mismatch hard-fails; files are large (our OLMoE measurement 131,072 B/token; this thread ~0.55 MB/token for GLM-4.6).
- **Quality loss:** none (exact state); thread notes only compatibility failures.
- **Ablation:** none
- **Limitations stated:** No official file-format guarantee (author of reply); cross-machine transfer untested.
- **Contradicts our findings?** No. Consistent with D.md rows P-0148/P-0151 (--prompt-cache works, byte-identical output).

## P-0149: hearth-llm 0.1.1 'Keeps your local LLM contexts warm: persistent KV-cache proxy for llama.cpp's llama-server', Daniel Soromou, MIT; releases 0.1.0 and 0.1.1 both 24 Jul 2026; repo github.com/caporalCoder/hearth

- **Version read:** PyPI JSON API (pypi.org/pypi/hearth-llm/json) which carries the full README; the pypi.org project page itself returns a JS challenge to curl. Repo scripts/tests not read  (raw text: True)
- **Setup:** Python stdlib proxy in front of llama-server: hash chain of chat messages, finds longest saved prefix, /slots restore, saves after the reply, LRU cap 20 GB default. Benchmark: Qwen2.5-7B-Instruct Q4_K_M, Apple M3 Max, llama.cpp b10090, mean of 3 runs, 48-token answer cap, server killed and restarted between cold and warm. Single slot (--parallel 1), requests serialised.
- **Headline numbers (what | value | baseline | ratio | where):**
  - 30.3k-token context: time to answer cold / warm | 99.0 s / 1.79 s | cold re-prefill | 55x (computed 55.3) | README Benchmark table
  - 15.3k tokens | 34.0 s / 1.01 s | - | 34x | same
  - 3.7k tokens | 6.8 s / 0.55 s | - | 12x | same
  - Prefill-only comparison, 30k | ~98 s cold prefill vs 0.46 s restore | - | ~213x (computed) | methodology notes
  - Snapshot size / KV per token / restore throughput, 30.3k | 1.62 GB / 56.2 KB per token / 3.5 GB/s (0.46 s); save 2.04 s | - | - | Memory and storage table; 15.3k: 4.4 GB/s; 3.7k: 5.0 GB/s
  - Across models at ~15k tokens | Qwen2.5-1.5B 29x, Phi-4-mini 24x, Llama-3.2-3B 20x, Mistral-7B 28x, Qwen2.5-7B 32x, Gemma-4-E2B 1.1x (13.8 s vs 12.08 s) | - | as stated | Across models table
  - 500-token answer instead of 48 | the 55x becomes roughly 9x | - | - | methodology notes
- **Catalog / first-pass numbers checked:**
  - 99 s cold vs 1.8 s warm on a 30k-token document, M3 Max -> **yes**: 99.0 s vs 1.79 s, Qwen2.5-7B Q4_K_M
  - Persists KV snapshots to disk automatically -> **yes**: after each response, restores longest matching prefix
  - first-pass: 30k snapshot ~1.6 GB; LRU at 20 GB cap; Gemma-4-E2B ~1.1x -> **yes**: 1.62 GB; 20 GB default; 1.1x
  - first-pass: 'prefill ~98 s vs 0.46 s restore' -> **yes**: 
- **Transfers to our regime:** yes. TTFT-only feature, and exactly the same idea (exact KV on disk keyed by prefix hash). Conditions: dense 7B on a Mac with a fast SSD, 5.3-6.2 GB server RSS; restore throughput measured with a warm OS page cache (author says a post-reboot restore 'will be somewhat slower'), so 3.5-5 GB/s is an upper bound for NVMe cold reads. Not tested on MoE streamed from NVMe; in that regime cold prefill is slower (all experts touched) so the ratio would be larger, but restore must compete with the expert stream for NVMe bandwidth (our assumption, not in the paper).
- **Quality loss:** None claimed; 'warm answers equal cold answers including mistakes', one deterministic question per size, not an evaluation.
- **Ablation:** Across models only; no cache-vs-layout. Gemma SWA is the failure case (engine cannot reuse restored SWA prefix).
- **Limitations stated:** Alpha MVP; single slot; message-granular matching (token-level planned); snapshots unencrypted plaintext equivalent; self-reported, single machine.
- **Contradicts our findings?** No. Confirms the idea is built (novelty 'known'). Its 56 KB/token for Qwen2.5-7B equals the analytic formula as our 131,072 B/token for OLMoE does.

## P-0150: DiskLLM 0.1.1 ('Ollama-style GGUF runner with SSD-backed llama.cpp KV cache'), DiskLLM contributors; 0.1.0 and 0.1.1 released 4 Jun 2026; repo github.com/shivnathtathe/diskllm

- **Version read:** PyPI JSON API (pypi.org/pypi/diskllm/json) carrying the README. The project page itself fails (JS challenge). Repo docs/benchmarks.md and the patch were NOT read  (raw text: True)
- **Setup:** Patched llama.cpp fork adding --kv-backend ssd: KV cache lives in an mmap file under ~/.diskllm/kv_cache with only a 2048-token window hot; run flags -ngl 0 --kv-window 2048 --no-repack. With --session NAME it also uses --slot-save-path and slot save/restore, session file <name>.mmap. Benchmarks: CPU-only Windows 11, Intel i5-12450H, 16 GB RAM, NVMe SSD (README).
- **Headline numbers (what | value | baseline | ratio | where):**
  - Private RAM, Qwen2.5 7B 65K ctx | 258 MB | stock ~6,000 MB | 23x | README Results table (self-reported)
  - Qwen2.5 7B 256K ctx | 258 MB at 2.5 tok/s | stock 14,900 MB | 57x | same
  - Qwen2.5 3B 65K / 14B 65K / 32B 65K / LFM2.5 8B 128K | 200 MB, 8.9 tok/s / 289 MB, 1.7 / 424 MB, 0.1 / 172 MB, 10.0 | stock ~2,000 / ~10,000 / ~20,000 / 1,845 MB | 10x / 34x / 47x / 10.7x | same
  - 7B restored session | 3.46 s to healthy + restore; 34 cached prompt tokens; 7.13 tok/s generation; session file 7.33 MB | - | - | README Benchmarks
  - Generation speed, 7B 65K ctx | 4.5 tok/s | stock tok/s not given | - | Results table
- **Catalog / first-pass numbers checked:**
  - Persists context across restarts through llama-server slot save/restore -> **partly**: true for --session, but the headline feature is a patched mmap KV-on-SSD backend that cuts private RAM; no stock-llama.cpp-only claim
  - first-pass: GGUF runner storing KV on SSD, --session uses --slot-save-path and a patched server binary -> **yes**: matches README
  - first-pass: unverified (page failed to load) -> **resolved**: README readable through the PyPI JSON API
- **Transfers to our regime:** partly. Not a TTFT feature: it is a RAM-footprint trick for long contexts on small-RAM CPU-only laptops (the same device class as ours), and the restore path is the stock slot API. 'Private RAM' excludes file-backed mmap pages, so model weights and KV pages still compete for page cache (my inference; the README does not state total RSS or page-cache use). Decode is slow (0.1-10 tok/s) and no quality or correctness tests are reported.
- **Quality loss:** Not reported.
- **Ablation:** none in README
- **Limitations stated:** Alpha, Windows-first, patched fork, self-reported, single machine; docs/benchmarks.md not read.
- **Contradicts our findings?** No. Relevant prior art: KV-on-SSD for small-RAM CPU boxes already exists (relevant to any idea of paging KV to disk), and it is a patched llama.cpp like ours.

## P-0151: llama : save and restore kv cache for single seq id (llama.cpp PR #6341), kaetemi; opened 27 Mar 2024, merged 8 Apr 2024 by ggerganov as commit beea6e1b16e783a0886e78dec01002a8c00db24d

- **Version read:** GitHub API: pulls/6341 body, 12 comments, commit beea6e1 message and file list. The catalog URL (gitlink.org.cn mirror) was not opened; the same commit hash was verified through the GitHub API  (raw text: True)
- **Setup:** Core API: llama_get_seq_size / llama_copy_seq_data / llama_set_seq_data (later renamed llama_state_seq_*); server endpoints /slot/save, /slot/restore, /slot/erase (final commit 'adjust endpoints to preferred style' => /slots/{id}?action=), --slot-save-path to restrict the folder. Files touched: llama.cpp, llama.h, examples/server (server.cpp, README, tests/features/slotsave.feature), examples/main, examples/save-load-state. Related issue #5843.
- **Headline numbers (what | value | baseline | ratio | where):**
  - Performance numbers in the PR | none for this feature; the bot comment is the generic server bench (T4, phi-2 q4_0, 538 iterations, 8 users) | - | - | PR comments
- **Catalog / first-pass numbers checked:**
  - per-sequence state save/restore in llama.cpp since 2024 -> **yes**: merged 8 Apr 2024
  - first-pass: merge commit beea6e1, 8 Apr 2024 -> **yes**: commit date 2024-04-08T12:43:30Z
  - first-pass: only perf numbers are an unrelated server benchmark -> **yes**: 
- **Transfers to our regime:** yes. Mechanism only: it makes exact KV restore possible. No TTFT measurement. Review thread (martindevans) notes the first file format had no header/magic/version/model check and asserted on model mismatch; changed to return 0 on mismatch.
- **Quality loss:** none (exact state)
- **Ablation:** none
- **Limitations stated:** Reviewer notes missing file-format metadata; author: needs tests (added slotsave.feature).
- **Contradicts our findings?** No. Consistent with D.md: mechanism behind --prompt-cache / slots.

## P-0152: Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving. Shi Qiu, Yifan Hu, Xintao Wang, Wenhao Zhu, Jianqin Yan, Hao Chen, Kaiqiang Xu, Kai Chen, Yiming Zhang (Xiamen University, SJTU, HKUST). cs.OS

- **Version read:** arXiv 2605.03375v1 (5 May 2026), full PDF via pdftotext  (raw text: True)
- **Setup:** GPU-centric KV object store with GPU io_uring and slack-aware I/O scheduling, integrated into vLLM. Server: 64-core Xeon 6530, 512 GB RAM, 2x H100 80 GB, 4x Solidigm D7-PS1010 7.68 TB (2 SSDs: 29 GB/s read, 12 GB/s write peak); 256 GB pinned DRAM, 14 TB SSD per GPU. Models Llama3-8B (main), GLM-4-9B-Chat-1M (2 GPUs TP). Workloads LEval (3k-200k tokens) and LooGLE (>100k), Poisson arrivals. Baselines vLLM HBM-only, LMCache-DRAM-LW, LMCache-SSD, LMCache-GDS (vLLM 0.12.0 + LMCache 0.3.9, and vLLM 0.17.0 + LMCache 0.4.1).
- **Headline numbers (what | value | baseline | ratio | where):**
  - TTFT, LEval, vLLM 0.17.0, highest load | -78.3% vs LMCache-GDS; -69.1% vs DRAM | GDS / DRAM | as stated | Sec 4.1 text; Fig 8; the abstract's 'strict SLO' wording is the abstract's, the 78.3% is the highest-load point
  - TTFT, old version (vLLM 0.12.0) | -71.8% vs GDS at highest load | GDS | as stated | Sec 4.1
  - Request rate under 1 s TTFT SLO | +50% vs DRAM, +100% (2x) vs GDS | DRAM / GDS | as stated | Sec 4.1
  - Cost per 1M tokens, LooGLE 0.5 QPS | -66.2% vs LMCache-SSD; ~27% vs LMCache-GDS | GDS | as stated | Sec 4.x / Fig 14 ($5/h per H100, $0.0088/GB/h DRAM, $0.000082/GB/h SSD)
  - Cache hit rates HBM / DRAM / SSD | LEval 8/53/84%, LooGLE 4/24/86% | - | - | Table 1
  - Write bandwidth | ~10 GB/s (9.8 at 128K tokens) vs LMCache-GDS ~7 GB/s; read up to 25.9 GB/s vs ~11.9 for LMCache with two SSDs | - | - | Sec 4.x bandwidth microbenchmark
  - Motivation: GPU stalls | 70-80% GPU stall with CPU-centric SSD restore | - | - | Sec 1 citing [41]
- **Catalog / first-pass numbers checked:**
  - TTFT cut 78.3% vs SSD-backed LMCache -> **yes**: 78.3% vs LMCache-GDS (the GDS-enabled SSD baseline), LEval, newer vLLM, at the highest request rate; vs plain LMCache-SSD the figure differs
  - 2x request rate -> **yes**: under a 1 s TTFT SLO vs GDS (1.5x vs DRAM)
  - server setting -> **yes**: 2 H100, multi-user Poisson load
  - first-pass: -27% serving cost -> **partly**: 27% is the cost advantage vs GDS on LooGLE at 0.5 QPS; -66.2% vs LMCache-SSD
  - first-pass: 'hardware/models not named on abstract page' -> **resolved**: named above
- **Transfers to our regime:** no. GPU-centric (GPUDirect Storage, GPU io_uring), 2x H100, 8-9B dense models, multi-tenant request-rate serving, SSD tier of 14 TB with ~29 GB/s; nothing about CPU-only, a single user, MoE or batch 1. The transferable observation: restore from SSD is only worthwhile if I/O is efficient; their baseline shows SSD restore can be slower than recompute (Fig 2).
- **Quality loss:** none (exact KV).
- **Ablation:** Yes (component ablation on bandwidth, GPU bubbles, SGL commands: 31x/91x gain at small I/O sizes), all GPU-side.
- **Limitations stated:** Not stated in the part read beyond future work in Sec 5.
- **Contradicts our findings?** No.

## Family A: LUT / codebook / vector-quantised expert kernels (P-0140 to P-0146)

### Further prior art found (read at the depth stated)
- **FASQ: Flexible Accelerated Subspace Quantization** (Qiao et al., arXiv 2605.04084, 22 Apr 2026), full text read. Product quantisation of LLM weights, GPU (RTX 3090): decode 45.2 tok/s at effective 4-bit vs FP16 43.9, 1.6-1.8x AWQ. Key for us: its decode GEMV is *LUT-free* (direct centroid dot-product accumulation) because a per-token LUT of K_s=256 partial products per subspace wastes 99.6% of the table at batch 1; the LUT is kept only for prefill GEMM where tokens amortise it. Independent confirmation of our gather-LUT refutation (different hardware). GPU only.
- **Highly Optimized Kernels and Fine-Grained Codebooks for LLM Inference on Arm CPUs** (Gope, Mansell, Loh, Bratt, Arm; arXiv 2501.00032, Dec 2024), full text skimmed (abstract, Table 1-3, kernels). Optimised 4-bit group-quant kernels give 3-3.2x prefill and up to 2x decode on a single Arm core vs llama.cpp Q4_0 (Graviton3, 64 cores: no gain at 64 threads for decode, bandwidth-bound); also group-wise non-uniform codebook quantisation. Shows the dequant baseline T-MAC beats is itself improvable (the llama.cpp repacked kernels).
- **EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture** (Duan et al., Duke, arXiv 2605.24144, May 2026), abstract/intro read only. Accelerator design; computes input-codebook dot products then structured lookups; claims large gains over lookup-based VQ SOTA. Not CPU.
- **MoBE: Mixture-of-Basis-Experts** (Chen et al., arXiv 2508.05257, Aug 2025), abstract/intro read. Shared basis matrices across experts in a layer: 24-30% parameter reduction on Qwen3-235B / DeepSeek-V3 / Kimi-K2 with 1-2% accuracy drop. Closest published prior art for the shared-basis low-rank part; no latency results read.
- **llama.cpp IQ2/IQ3 family** (P-0144), **QuIP#, GPTVQ, VPTQ** (cited inside KBVQ-MoE, not opened), **bitnet.cpp, LUT-GEMM, FLUTE** (search snippets only; not opened; from memory as to details).
- Search found no paper combining *one codebook shared across all experts of a layer + CPU LUT kernel + NVMe streaming*. CARDAN (P-0146) and KBVQ-MoE (P-0145) share codebooks/low-rank parts across experts but target accelerator/GPU.

### What is new in OUR variant (if anything)
- Byte lever (2.0-2.6 bpw grid/codebook quants for MoE experts running on CPU llama.cpp): **known**, shipping since Jan 2024 (IQ2_XXS/IQ2_XS, Mixtral works).
- Compute lever (8-dim codebook index used as table index, gather+add instead of FMAs, on CPU, at batch 1): published CPU kernels either use bit-serial tables (T-MAC, vs a dequant baseline) or FP32 baselines (AQLM); two independent papers (FASQ for GPU decode; CARDAN where d=2 lookup made decode slower than the dense BF16 baseline) report that lookup work is what cancels the byte saving. Our measurement (0.06-0.17x of int8 dot, AVX2 gather) agrees; no published paper reports the same AVX2-gather-vs-int8-dot comparison, so the measurement is novel, the idea is not.
- Closest read-in-full papers: CARDAN (shared per-layer codebook + shared low-rank across experts; accelerator; needs distillation), AQLM (additive 2x8 codebooks, CPU kernel, FP32 baseline), T-MAC (CPU tables), KBVQ-MoE (shared SVD + VQ on MoE).
- Verdict: novelty known/variant; published evidence does not support a speedup in our regime (no paper measures against an int8 Q4 kernel with experts streamed from NVMe); our own measurement is consistent with the papers once the baselines are normalised.

## Family B: persist and reuse the exact KV cache of a finished prompt (P-0148 to P-0152)

### Further prior art found
- **CachedAttention / AttentionStore** (Gao et al., USENIX ATC 2024, arXiv 2403.19708), abstract/intro read (full PDF fetched): hierarchical KV store (HBM / DRAM / SSD) for multi-turn conversations with layer-wise preload and async save and scheduler-aware placement; TTFT down by up to 87%, prefill throughput up to 7.8x, cost down up to 70%. Server GPU setting. Established the SSD KV tier idea in 2024.
- **Prompt Cache: Modular Attention Reuse for Low-Latency Inference** (Gim et al., MLSys 2024, arXiv 2311.04934v2), abstract/intro read: precomputed attention states of prompt modules reused; TTFT 8x on GPU, **60x on CPU**; no model change. Stores states in server memory rather than disk, but the CPU result is the closest quantitative prior art for TTFT on a CPU.
- **TensorRT-LLM KV cache connector example** (search result, docs page not opened in full): persistent KV on disk keyed by token-hash plus salt. 
- **LMCache** (Tutti's baseline; described in Tutti), **vLLM prefix caching, SGLang RadixAttention, Mooncake** (from memory, not opened).
- Practical tools read: hearth (P-0149) and DiskLLM (P-0150) on PyPI, llama.cpp --prompt-cache / slots (P-0148, P-0151).
- Side evidence read in PR #4773 comments (Kawrakow, 10 Jan 2024): for prompt processing a Mixtral MoE touches all experts so it is 'equivalent to a 46B model', which supports the claim that MoE prefill is expert-I/O-bound when experts stream from disk.

### What is new in OUR variant
- Exact-KV persistence keyed by token-prefix hash is **known** and shipped (llama.cpp since Apr 2024, hearth Jul 2026, AttentionStore 2024, Prompt Cache 2023).
- Not found in any paper read: the regime of a 125B-class MoE whose cold prefill is bound by streaming experts from NVMe on an 8-32 GB CPU laptop (so the saved work is I/O-bound expert traffic, not FLOPs), and keying on router/quantisation identity. Published numbers are dense 7-9B models (hearth: 55x end-to-end at 30k tokens; Tutti/AttentionStore: server GPUs). Novelty: variant (regime), not mechanism.
- Evidence for the regime: none directly; hearth's restore throughput (3.5-5 GB/s, warm page cache) would compete with the expert stream for NVMe bandwidth.
- Our D.md measurement (--prompt-cache on OLMoE: 419-token prompt, 6.0 s cold, 0 tokens evaluated warm, byte-identical, 131,072 B/token) is consistent with hearth's analytic KV/token formula and with all papers.

## Corrections to the first pass (P07.md / P07.json) and catalog

1. P-0140: the 2.8x e2e number is internally inconsistent in the paper (intro says 2-bit; Fig 8 order gives 4-bit 2.8x, 2-bit 6.7x on RPi5); the catalog's 'avg 3.6x at 1-2 bit' is not restricted to 1-2 bit; the paper's prose says FA costs 0.3% accuracy but Table 4 shows -3.0 pt WinoGrande.
2. P-0141: catalog tok/s 9.3 / 28.4 / 18.7 are wrong: 9.39 is the *llama.cpp* 2-bit baseline in the T-MAC paper Table 7; 28.4 and 18.7 appear nowhere. Blog says 30 / 20 (paper 31.83 / 21.63).
3. P-0142: the PPL 6.59 / 3.94 quoted are for the 1x16 GPU-kernel models, not the 2x8 CPU-kernel config (7.61 untuned / 6.57 tuned, 70B 4.21); the PDF is readable; added e2e CPU tok/s (Table 14, 2.24-3.25x vs FP32).
4. P-0144: catalog 21.4 (AVX2) / 19.5 (NEON) tok/s appear nowhere in PR 4773, its 41 comments, the commit message or PR 4856; true: 24.3 vs 15.4 tok/s on Ryzen 5975WX; NEON 8.9 tok/s at merge.
5. P-0145: first pass claim 'at 2 bit GPTQ beats it (66.36 vs 62.78)' is wrong (66.36 is GPTQ 3-bit; GPTQ 2-bit is 49.07). At 3 bit GPTQ wins only on DeepSeek-V2-Lite by 0.16 pt. ICLR 2026 is confirmed (v2).
6. P-0146: first pass 'summary wrong / no lookup language' was based on the abstract; the paper does use per-layer shared weight code books with one lookup per block plus shared low-rank; catalog description is roughly right but omits the low-rank part, gate/up only, and the distillation requirement.
7. P-0150: no longer unverified; README read via PyPI JSON: main feature is mmap KV-on-SSD (RAM 258 MB for 7B at 65K, 4.5 tok/s on an i5-12450H 16 GB), session persistence is secondary.
8. P-0151: catalog mirror URL not opened but the commit beea6e1 is verified (merged 8 Apr 2024 by ggerganov).
9. P-0152: 78.3% is vs LMCache-GDS at the highest LEval load with vLLM 0.17.0; 27% cost is vs GDS on LooGLE 0.5 QPS; hardware now known (2x H100).

## Results that change what we should believe about our own measurements

- Nothing contradicts our measurements. Two items strengthen them: CARDAN (full text) shows equal-rate VQ-only with more lookups (d=2) is 2.9% slower than the dense BF16 megakernel despite 12% fewer bytes (Qwen3 8.075 vs 7.851 ms), and FASQ drops the LUT in B=1 decode for the same reason. The AQLM CPU kernel itself runs at ~17 GB/s of code bytes vs ~98 GB/s for its FP32 baseline (my derivation), i.e. it is lookup/compute-bound. So the LUT-compute half should be treated as refuted by three independent sources, not only by our AVX2 gather test.
- One caution for our KV-persist arithmetic: hearth's restore throughput (3.5-5 GB/s) is with a warm OS page cache; cold NVMe reads, and competition with expert streaming, will be slower.
