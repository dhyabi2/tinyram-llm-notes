# deep_P02a: raw full-text re-check of batch P02a (family s1 #4, #5, #6 rows)

Method: all papers read from the arXiv PDFs (pdftotext), Reddit snapshot, linked write-up and hackmd as raw text. 10 of 10 rows raw-read, 0 unread. Numbers marked 'computed' are my ratios from the tables. Remembered facts are labelled 'from memory'.


## P-0032: DraftExpert: Expansion-Aware Self-Speculative Decoding for End-Device MoE Inference (Dengke Han; arXiv 2607.24434v1, 27 Jul 2026, cs.LG, 9 pp, no venue stated)

- id read: 2607.24434v1 (PDF, pdftotext)
- raw text read: True
- setup: Models: DeepSeek-V2-Lite (15.7B total, 2.45B active, 27 layers, 64 routed + 2 shared, top-6; routed pool 14.4B = 26.8 GiB BF16, non-routed 2.4 GiB) and Moonlight-16B-A3B (same layer/expert shape). Platforms: CG = RTX 4090 24 GB, BF16, PyTorch 2.11, routed experts in host RAM copied over PCIe on demand; MN = Xiaomi MIX Flip 2, Snapdragon 8 Elite, Hexagon HTP v81 NPU, llama.cpp Q4_0, routed experts in Flash staged into shared DRAM. Baseline AR = frozen target, experts loaded on demand, 'not kept in a persistent accelerator cache' (no cache at all; fresh process = no GPU expert cache). Other baseline: best training-free shared+top-r self-drafter (r in 1..3). Prompts: chat, GSM8K, MBPP, summarisation, instruction following. Draft length K up to 16, batch 1 implied (single-user), exact verification. Needs one trained draft expert per layer (self-distillation from frozen target); training cost, data size and GPU-hours NOT reported. Flash bandwidth not measured/stated.

- numbers (what | value | baseline | ratio | where):
  - Decode TPS DS, CG | DraftExpert 2.99 TPS | AR 2.19 TPS | 1.365x (paper 1.36x) | Table 2, p.6
  - Decode TPS DS, MN (Q4_0 Flash) | 15.47 | AR 10.18 | 1.520x (paper 1.52x) | Table 2
  - Decode TPS ML, CG | 2.50 | AR 1.94 | 1.289x (1.29x) | Table 2
  - Decode TPS ML, MN | 13.69 | AR 8.50 | 1.611x (1.61x) | Table 2
  - Average speedup | 1.45x | AR | mean of four = 1.446x | Table 2
  - Training-free shared+top-r exact drafter | 1.19 / 4.80 / 1.01 / 3.82 TPS | AR 2.19 / 10.18 / 1.94 / 8.50 | 0.54 / 0.47 / 0.52 / 0.45 (slower than AR in all four cells) | Table 2
  - DraftExpert vs best training-free top-r | 2.99/1.19, 15.47/4.80, 2.50/1.01, 13.69/3.82 | top-r | 2.51 / 3.22 / 2.48 / 3.58x | Table 2 (computed)
  - MN resident-memory AR reference | about 14.36 TPS | Flash-staged AR 10.18 | 1.41x = the loading headroom | p.6 text
  - Draft acceptance K=4 | DS 85.0%, ML 83.0% | top-1 22.5% / 21.8%; top-3 45.9% / 44.0% | 3.8x over top-1 | Table 3
  - Unique routed layer-experts touched by drafter K=4 | DS: DraftExpert 88 | top-1 84, top-2 183, top-3 244 | 1.05x top-1; top-3 is 2.9x | Table 3
  - Verify-side cost (normalised to one AR token) DS | fixed K 2.35x (CG) / 3.10x (MN); conf+expansion 1.48x / 1.70x; +prefetch 1.25x / 1.35x; hit/waste 88/12% | fixed K | fixed-K block costs 2.35-3.30x one AR token | Table 4
  - Verify union growth | about 59% of all layer-experts covered at K=16 | - | - | Fig 5 (figure label)
  - Expert-load share of decode time | 87% (CPU->GPU RTX 4090), 49% (Flash->NPU) | - | - | Fig 1b
  - Training ablation DS | acc 47.7% -> 79.5% (+residual) -> 87.1% (+logit/token) -> 87.0% (+router); prefetch hit 61% -> 69% -> 73% -> 88% | top-1 draft | - | Table 5
- catalog and first-pass numbers:
  - [yes] 1.45x average decode speedup: 1.45x (Table 2), but it is vs expert-offloaded AR with NO expert cache
  - [yes] 84-87% draft acceptance: 85.0% DS / 83.0% ML at K=4 in Table 3 (Table 5 gives 87.1%/84.5%); abstract says 84-87%
  - [yes] 86-88% prefetch hit rate: 88% DS, 86% ML (Tables 4-5)
  - [yes] verifying a block loads the union of its experts: Eq 3, Fig 5
  - [no] [first pass] 'baseline has no stated LRU size': Stated: no persistent expert cache in the AR baseline (p.6 setup)
  - [yes] [first pass] 'training cost not reported': not reported anywhere in the 9 pages
  - [yes] [first pass] 'about 12-14% wasted prefetch': Table 4 hit/waste 88/12 and 86/14
- transfers: no
- why: Assumptions that break: (1) needs a trained per-layer draft expert (self-distillation) and a 125B model would need that training; (2) baseline AR has no cache, so the 1.45x is a gain over cache-less decode, which is exactly the regime where our own simulation (A.md) also finds a gain (k=8 vs k=1 at 0 footprints 2.19x) and none at 2-4 footprints; (3) 16B/64 experts/top-6 vs our 512 experts/top-10; (4) GPU/NPU compute with BF16/Q4_0, host RAM or UFS Flash at unknown bandwidth, not a CPU matmul bound; (5) acceptance 83-87% comes from the trained draft path; (6) the draft path keeps shared+top-1 experts resident, a memory budget we do not have at 8 GB. The paper's premise (verify of K tokens activates the union; MoE verify 2.4x-5.8x one token as K grows per Fig 4 labels) supports our byte finding.
- quality_loss: None claimed: exact verification, output identical to target (greedy-style accept). No quality benchmark; speedups only.
- ablation: Table 3 (drafter footprint vs r), Table 4 (fixed K vs confidence+expansion truncation vs +prefetch), Table 5 (loss terms). No ablation isolating cache vs prefetch since there is no cache baseline; no AR + LRU baseline.
- limitations: No limitations section. Paper itself says exact verification 'still exposes misses, prefetch waste and non-expert runtime overhead' (p.6). Only two models, two platforms, no variance/CI reported.
- contradicts_our_findings: No. Consistent with A.md s1#4: gain is against a no-cache baseline; paper never compares to AR+LRU.

## P-0033: AcceptMoE: Commitment-Weighted Self-Sizing Verifier Expert Sets for Efficient MoE Speculative Decoding (Shuang Liang, Hao (Mark) Chen, Zhiwen Mo, Qianzhou Wang, Guoyu Li, Lingxiao Ma, Wayne Luk; Imperial College London, Tile-AI; arXiv 2608.02989v1, 4 Aug 2026)

- id read: 2608.02989v1 (PDF; Table 1 and Fig 3/4 also read as rendered page image)
- raw text read: True
- setup: Targets: Qwen3-30B-A3B-Instruct-2507, Qwen3-Coder-30B-A3B (128 experts, top-8) and GPT-OSS-120B (128 experts, top-4 per its model card, 36 layers; from memory for top-4), each with its EAGLE-3 draft; draft tree 5 steps, EAGLE top-k 10, 64 draft tokens; SGLang 0.5.12.post1, batch 1, greedy, 256 tokens/prompt, 20 prompts (offload) or examples 0-49 (resident), median of 3 fresh-server repetitions. Resident: RTX PRO 6000 Blackwell. Offload: RTX 5090 32 GB, X=48 expert slots per layer (37.5% of 128 experts) with LRU, full experts in pinned host RAM (Qwen BF16, GPT-OSS native MXFP4), non-expert weights, draft model and KV on GPU, overlap/CUDA graphs disabled. Baselines: Vanilla AR, Standard SD (EAGLE-3 natural routing), Count-B and MoE-Spec at matched budget B0 (23-34 experts), AcceptMoE-B (matched budget). Benchmarks: GSM8K, MATH500, HumanEval, MBPP. Needs offline commitment-probability estimates from training-split traces (no model training).

- numbers (what | value | baseline | ratio | where):
  - Throughput, all experts resident | mean 1.290x (per pair 1.217-1.339x; per-model 1.249-1.318x) | Standard SD | as stated | Fig 3, text p.6
  - Throughput under offload (RTX 5090, X=48) | mean 2.06x | Standard SD | as stated; AcceptMoE also beats Vanilla AR on all 12 pairs | Fig 4, text p.6
  - Offload tok/s read from bars (approx.) | AcceptMoE roughly 18-52 tok/s | Vanilla AR roughly 9-30 tok/s; Standard SD roughly 9-31 | AcceptMoE/AR about 1.3-2.4x by bar eyeballing (not stated numerically) | Fig 4 (read from rendered bars; approximate)
  - H2D expert traffic per output token (MiB), Qwen-Inst / Coder / GPT-OSS | AcceptMoE 633 / 464 / 298 | Standard SD 2394 / 2025 / 1243; Vanilla AR 888 / 721 / 439 | vs SD 0.264 / 0.229 / 0.240 (-73.6/-77.1/-76.0%); vs AR 0.713 / 0.644 / 0.679; SD vs AR 2.70 / 2.81 / 2.83x | Table 2 (ratios computed)
  - Cache hit rate (LRU, 48 slots) | AcceptMoE 83.8 / 87.0 / 88.5% | Standard SD 73.0 / 74.4 / 74.3%; Vanilla AR 81.0 / 83.5 / 81.0% | +10.8 / +12.6 / +14.2 pts vs SD; +2.8 / +3.5 / +7.5 pts vs AR | Table 2
  - Fetch stall share of wall time | Standard SD 65-78%; AcceptMoE 27-35% | - | - | text p.6, Fig 5a
  - Standard SD vs AR throughput under offload | SD slower than AR on 7 of 8 Qwen pairs; faster on all 4 GPT-OSS pairs (MXFP4) | Vanilla AR | - | text p.6
  - Residency-aware pruning ablation | traffic -38.6% to -48.6%, hit +6.9 to +8.2 pts, throughput +4.6% to +15.1% | same selector, pruning off | - | Fig 5c/d, text p.7
  - Accuracy (12 pairs, mean) | AcceptMoE 90.63; AcceptMoE (Offload) 90.25 | Vanilla AR 91.01; Standard SD 90.90; MoE-Spec 88.15; Count-B 82.72 | resident -0.27 pts vs SD (-0.4% vs AR); OFFLOAD config -0.65 pts vs SD and -0.76 pts vs AR | Table 1 (computed)
  - Per-pair accuracy delta from pruning | -1.83 to +1.22 pts, mean -0.27 | pruning off | - | text p.7
  - Self-sizing vs best fixed budget | 0.97 pts below on average, at most 1.83 | 5 budget-swept pairs | - | text p.7
  - Worst per-pair accuracy vs AR (resident) from Table 1 | e.g. Qwen3-Coder MATH500 86.80 vs 88.00; Inst HumanEval 92.07 vs 93.29; OSS MBPP 78.20 vs 78.60 | Vanilla AR | - | Table 1
- catalog and first-pass numbers:
  - [yes] 2.06x vs standard spec decoding under physical expert offloading: mean 2.06x vs Standard SD (Fig 4); baseline is Standard SD, not AR+good cache
  - [yes] 73.6-77.1% less host-to-device traffic at batch 1: -73.6 / -77.1 / -76.0% vs Standard SD (Table 2); vs Vanilla AR the saving is only 28.7 / 35.6 / 32.1%
  - [partly] [first pass] mean accuracy 0.27 points below Standard SD: true for the resident config (90.63 vs 90.90); the offload config actually reported in Table 1 is 90.25 = 0.65 below SD and 0.76 below AR; 0.27 is also the pruning-on vs pruning-off delta
  - [yes] [first pass] 'Standard SD moves MORE bytes than vanilla AR, lower hit rate': 2.70-2.83x more bytes, hit 73-74% vs 81-83.5% (Table 2)
  - [yes] [first pass] 'Standard SD is faster than AR on all four GPT-OSS pairs': text p.6
- transfers: partly
- why: Direction yes, numbers no. Independent measurement that under an LRU-cached offload path Standard SD (64-node tree) moves 2.7-2.8x the bytes of AR, which supports our k>1-loses-to-LRU finding. AcceptMoE's own win comes from restricting which experts the target may use (a lossy change to routing: -0.65 to -0.76 pts mean accuracy, per-pair offload-row deltas vs AR in Table 1 reach about -4.3 pts, e.g. OSS HumanEval 94.51 -> 90.24 and OSS MBPP 96.20 -> 93.40), on a 32 GB GPU with 37.5% of experts resident as 48 slots/layer and PCIe host RAM; our cache is a few percent of the 512-expert pool on CPU/NVMe. Even so vs AR the traffic saving is 29-36% and throughput gain unspecified in text. Tree-draft with EAGLE-3 needs a trained draft head; batch 1.
- quality_loss: Lossy routing restriction. Mean accuracy 90.25 (offload) vs 91.01 AR (-0.76 pts, -0.8% rel); resident self-sizing -0.27 vs SD. MoE-Spec at the same budget -3.1% vs AR and Count-B -9.1% (Table 1), MoE-Spec cells down by up to 13 pts vs AR (Coder HumanEval 93.29 -> 80.49).
- ablation: Yes: fixed-budget vs self-sizing, matched-budget AcceptMoE-B vs MoE-Spec (Table 1), residency-aware pruning on/off (Fig 5, Table 2: isolates the cache-aware part: -38.6 to -48.6% traffic, +4.6 to +15.1% throughput). No cache-size sweep.
- limitations: Masking changes the target router's logits, 'any such selector is an approximation rather than distribution-preserving' (p.3). Model architecture and kernels differ across targets so GPT-OSS vs Qwen comparison does not isolate fan-out (p.6). B0 chosen from AcceptMoE's own mean sizes.
- contradicts_our_findings: No. Supports A.md s1#4 verdict (union/SD loses to LRU bytes); new quantitative support: SD/AR bytes 2.7-2.8x at 37.5% slots.

## P-0034: MoE-Spec: Expert Budgeting for Efficient Speculative Decoding (Bradley McDanel, Steven Li, Sruthikesh Surineni, Harshit Khaitan; Franklin & Marshall College / Meta Reality Labs; arXiv 2602.16052v1, 17 Feb 2026, cs.LG, preprint 19 Feb 2026)

- id read: 2602.16052v1 (PDF)
- raw text read: True
- setup: Models (FP16): OLMoE-1B-7B (64 experts, k=8, EAGLE-3 draft, 1 GPU), Qwen3-30B-A3B (128 experts, k=8, EAGLE-3, 1 GPU), Mixtral-8x7B (8 experts, k=2, original EAGLE, 2 GPUs). NVIDIA A100 80 GB, experts fully resident in HBM, PyTorch 2.1/CUDA 12.1. Draft tree 63 tokens (sweep 3-511), T=0 and T=1 (5 seeds), 80 samples per benchmark, 1024 max gen length, 5 benchmarks (GSM8K, MATH500, HumanEval, MBPP, CNN/DM). Baselines: AR and EAGLE/EAGLE-3. Method: training-free; rank experts per layer by aggregate router mass over the tree, keep top-B, substitution policy (or truncation); batched MoE kernel implemented for all methods. No offloading experiment.

- numbers (what | value | baseline | ratio | where):
  - Fraction of experts active in a tree | OLMoE 127-token tree activates 54 of 64 experts per layer | - | 84% | Fig 2b / text p.1
  - Heavy tail | top 32 of 64 experts capture 93% of routing mass (tree 63) | - | - | Fig 2c
  - Configs where MoE-Spec beats EAGLE | 27 of 30 | EAGLE/EAGLE-3 | - | Sec 4.2
  - Average relative gain over EAGLE | Mixtral 27%, Qwen3-30B 16%, OLMoE 6% | EAGLE | - | Sec 4.2
  - Mixtral MBPP T=0 | 2.3x | EAGLE 1.7x; AR 1.0x | 1.35x (paper: 35% relative) | Table 1
  - Qwen3-30B GSM8K T=0 | 2.4x | EAGLE-3 1.9x | 1.26x | Table 1
  - Qwen3-30B MBPP T=0 | 1.7x | EAGLE-3 1.7x | 1.00x | Table 1
  - OLMoE MBPP T=0 / CNN-DM T=1 | 2.2x / 1.0x | EAGLE-3 2.2x / 1.0x | 1.00x | Table 1
  - Tree 255 on OLMoE | EAGLE-3 54 experts, 1.85x; MoE-Spec B=32, 2.1x | AR | 1.14x | Fig 4 / text p.6
  - Acceptance length change | -1.4% on average | EAGLE | - | Sec 4.2
  - Quality at max speedup | Qwen3 about 95% relative quality at 2.0x; Mixtral about 78% at max speedup; OLMoE steepest decline | AR = 100% | - | Fig 3 text p.5
  - Qwen3-30B HumanEval T=0 | 78 pass@1 | AR 82, EAGLE-3 82 | -4 pts | Table 1
  - Router-ranking overhead | 2-3% | - | - | Sec 6
- catalog and first-pass numbers:
  - [yes] the target must load the union of experts of all draft nodes: Eq 3, Sec 3.1
  - [yes] caps experts at a quality/efficiency cost: budget B per layer; quality 'within one std dev' of EAGLE at chosen B
  - [yes] [first pass] 27 of 30 configs, 27/16/6% averages, 2.3x vs 1.7x, 54 of 64, 54 vs 32 experts at tree 255 (1.85x vs 2.1x), 10-30% abstract: all confirmed in text/Table 1
  - [yes] [first pass] Mixtral on 2 GPUs, A100 80 GB, 80 samples, 5 benchmarks: Tables 2-3
  - [yes] [first pass] 'EAGLE uses q(x)=1 so T=1 not lossless even for baseline': footnote 3, p.4
- transfers: no
- why: HBM-bound A100 with every expert resident, comparison to EAGLE-3 (a speculative baseline), lossy substitution of experts, trained EAGLE draft. Authors state it 'targets HBM bandwidth when experts fit in GPU memory', offloading methods target PCIe (p.8) and that batch inference changes trade-offs (Sec 6). The premise (union grows to 84% of experts at tree 127; tail experts are stragglers) transfers as a measurement of the union fraction and agrees with our union measure; the speedup does not.
- quality_loss: Substitution policy lossy; reported benchmark deltas per cell in Table 1: e.g. Qwen3-30B HumanEval 82 -> 78 at T=0, OLMoE GSM8K 74 -> 66 at T=0 (AR 74, EAGLE-3 69), Mixtral HumanEval 20 -> 27 (up). Average gap to EAGLE kept within one std dev by budget selection. Aggressive budgets: relative quality about 78% (Mixtral) at max speedup.
- ablation: Yes: token policy (truncation vs substitution), ranking (static vs router vs oracle: oracle reduces expert requirement by about 25%), budget sweep, tree size. None isolates cache or offload.
- limitations: Budget hyperparameter per deployment; 2-3% ranking overhead; single-request A100 only, gains may shrink for batches; softmax routing only (sigmoid routers like DeepSeek-V3 may need adaptation); no per-layer budgets (Sec 6).
- contradicts_our_findings: No. Reports tree-of-127 union 54/64 experts per layer; our A.md union at k=8 consecutive tokens 0.454 of experts (different construct: tree vs chain).

## P-0035: Reddit post 'I spent three days trying to make a 754B MoE usable on 63 GB of fast memory. It ends at 0.9 tok/s, and I can now prove that's the floor.' by u/Sleepybear2611; snapshot mirrored 2026-07-17 18:53 UTC (reddit.sentinel-team.org), linked write-up ARTICLE.md in repo dadwritestech/bytes-per-token (read raw)

- id read: mirror HTML snapshot + raw ARTICLE.md from GitHub (the repo's analysis scripts and figures not opened)
- raw text read: True
- setup: GLM-5.2 (754B MoE, top-4 routing, 365 GB at IQ4_XS, about 3.88 bpw), llama.cpp fork with experts on CPU path and attention on GPUs; RTX 5080 16 GB + RTX 5060 Ti 16 GB (63 GB 'fast memory' = 32 VRAM + 31 RAM), Ryzen 9700X (4 cores saturate), 31 GB RAM, WD SN7100 NVMe, 5.7 GB/s sustained at deep queue depth; 12 GB RAM expert LRU cache; batch 1 decode. Self-reported; 7 comments, 3 captured.

- numbers (what | value | baseline | ratio | where):
  - Final decode speed | 0.9 tok/s | 0.2 tok/s start (4.5x) | 4.5x | post text
  - Bytes touched per token | about 3.4 GB of expert weights (after a 12 GB LRU cache) | drive 5.7 GB/s | ceiling 1.55 tok/s (write-up later quotes an I/O ceiling of 1.41) | post / ARTICLE
  - CPU expert compute wall | 0.36 s/token, 4 cores -> 2.8 tok/s even with zero disk | - | - | post / ARTICLE
  - Smaller quant | UD-Q2_K_XL = 0.70x size: still 0.9 tok/s, same 3.4 GB/token streamed | IQ4_XS | 1.00x | post
  - Hot-slot skew | top 1.7% of expert slots take 25% of routings | - | - | post
  - Speculation tree byte cost | every real tree shape (2x4, 4x2, 3x3, 2x6) reads 2.8-3.3x its accepted path; tolerance R* = 1 + t_fix/t_io = 1 + 0.36/0.71 = 1.51 | greedy decode reads exactly its needed experts | 2.8-3.3x over a 1.5x tolerance, at perfect acceptance | ARTICLE.md (spec section)
  - Sibling union | amortisation A(K) grows about sqrt(K) on a 35B top-8 and the 754B top-4 | - | - | post / ARTICLE
  - Prompt-predicted pinned working set | covers 27-44% of generation-time touches | reactive LRU 42-62%, near oracle | LRU 1.4-1.6x better | post
  - One 96-token generation touches | 38% of all experts | - | - | post
  - Fully resident small model on same box | GLM-4.5-Air 47 GB Q2_K_XL: 19.5 tok/s | 0.9 | 21.6x | post
  - Cross-check cited from another project (colibri) | about 11 GB/token cold top-8 on about 1 GB/s: 0.05-0.1 tok/s; cold-cache speculation inflates expert-loads/token about 660 -> 1100 | - | 1.67x | ARTICLE (second-hand)
- catalog and first-pass numbers:
  - [yes] argues spec decoding is dead under SSD offload: 'Speculative decoding is dead in this regime, the whole family'
  - [yes] argues greedy decoding already reads only needed experts: stated verbatim: 'greedy decode reads exactly the experts it uses, so greedy is already I/O-optimal' (first pass said not on the page: wrong)
  - [partly] a commenter disputed its RAM figures: one commenter (Technical-Earth-3254) asked the author to elaborate his 'ai slop ram numbers' and another flagged it as AI-written; a third praised it. The 63 GB = 32 GB VRAM + 31 GB RAM
  - [yes] [first pass] 0.9 tok/s, 3.4 GB/token, 1.55 ceiling, 2.8 tok/s CPU cap, 0.70x quant same speed, 1.7% slots take 25%: confirmed in the post
  - [yes] [first pass] 'top-8 -> top-4 override': post lists 'overriding top-8 routing to top-4' as one of four changes
- transfers: partly
- why: Qualitatively yes: NVMe-bound, batch 1, tiny RAM share (63/365 GB = 17%, 12 GB cache = 3.3%), reactive LRU is the frontier, speculation loses on bytes. Quantitatively one anecdotal run: 754B top-4 (not 125B top-10 with 512 experts), two GPUs doing attention, 5.7 GB/s drive, AI-assisted write-up, not peer reviewed, spec-test details only in a repo I did not execute. Its R* formula (tolerance 1.5x because compute t_fix 0.36 s vs I/O 0.71 s) is regime dependent: our 125B CPU box has different t_fix/t_io.
- quality_loss: top-8 -> top-4 override is lossy; no quality number given for it. Uniform Q3_K experts: +1.7% PPL at about 2x fewer bytes on a Q6 model (post).
- ablation: Stepwise ablation of the four speed-ups is in the write-up (each measured before the next; mmap -> direct-read streamer gave the final 0.9). Byte-accounting isolates file size vs streamed bytes. No cache-size sweep.
- limitations: Single machine; the author states two walls (I/O 3.4 GB/token, CPU 0.36 s/token); did not test router-lookahead prefetch, which colibri reports as about 72% predictable from the post-attention state.
- contradicts_our_findings: No; agrees with our byte finding (A.md s1#4/#5). One difference: it reports sibling-candidate union about sqrt(K) but consecutive-token union uncorrelated with depth, whereas our traces measure consecutive tokens (union 0.454 at k=8).

## P-0037: MoE-Spec: Expert Budgeting for Efficient Speculative Decoding (same paper as P-0034; arXiv 2602.16052v1, abs page url in catalog)

- id read: 2602.16052v1 (PDF, same text as P-0034)
- raw text read: True
- setup: Identical to P-0034 (A100 80 GB, OLMoE/Qwen3-30B-A3B/Mixtral, EAGLE-3/EAGLE tree 63, experts resident, FP16).

- numbers (what | value | baseline | ratio | where):
  - Abstract claim | 10-30% higher throughput than EAGLE-3 at comparable quality | EAGLE-3 | - | Abstract
  - Per-model average | 27% / 16% / 6% (Mixtral / Qwen3-30B / OLMoE) | EAGLE | - | Sec 4.2
  - Acceptance length | -1.4% on average | EAGLE | - | Sec 4.2
  - Detailed numbers | see P-0034 |  |  | 
- catalog and first-pass numbers:
  - [yes] Verification of a draft tree loads the union of experts: Sec 3.1
  - [partly] top-B budgeting gives 10-30% over EAGLE-3 on OLMoE, Qwen3-30B-A3B, Mixtral: abstract says 10-30%; Mixtral uses EAGLE-1 (not EAGLE-3) and per-model averages are 6-27% relative gain; Qwen3 and OLMoE use EAGLE-3
  - [yes] [first pass] same as P-0034: duplicate
- transfers: no
- why: Same as P-0034. The row's idea (small resident draft + confidence gate) is not what MoE-Spec does: it is a verification-time per-layer expert budget on a trained EAGLE draft, with no gate that skips the big model.
- quality_loss: See P-0034 (lossy substitution; within one std dev of EAGLE at chosen B; larger drops at aggressive B).
- ablation: See P-0034.
- limitations: See P-0034.
- contradicts_our_findings: No.

## P-0038: SP-MoE: Speculative Decoding and Prefetching for Accelerating MoE-based Model Inference (Liangkun Chen, Zijian Wen, Tian Wu, Xiaoxi Zhang, Chuan Wu; Sun Yat-sen University / HKU; arXiv 2510.10302v2, 6 Nov 2025, cs.DC; ACM-style 'Conference'17' template, no venue)

- id read: 2510.10302v2 (PDF)
- raw text read: True
- setup: Target/draft pairs: Mixtral 8x7B (8 experts, top-2, expert 336 MB, 14 ms load) / Mistral 7B; Phi-3.5-MoE (16 experts, top-2, 150 MB, 6 ms) / Phi-mini-MoE; DeepSeek-V2-Lite (64 experts, top-6 + 2 shared, 16 MB, 0.6 ms) / DeepSeek-Lite-AWQ. Env1 RTX 3090 24 GB + Xeon 6348 512 GB; Env2 RTX 4090 24 GB + Xeon 6338N 1 TB; Env3 A100 40 GB + Xeon 8358P 1 TB; PCIe 4.0 x16 (32 GB/s). Batch 1, 100 output tokens, chain drafting (draft token length 1 baseline; tree drafting listed as future work), greedy accept; acceptance 97.42 / 98.15 / 97.01% (Table 1). Datasets HumanEval, BigBench, WikiText-103, MMLU-Pro. Baselines (each given SD): Mixtral-Offloading (LRU), MoE-Infinity, AdapMoE. LRU cache inside SP-MoE; experts on host DRAM, not SSD.

- numbers (what | value | baseline | ratio | where):
  - Abstract: TPOT speedup | 1.07x-3.5x over baselines | 3 baselines | - | Abstract
  - Mixtral, 4 datasets, 3 envs | 1.35x average over baselines; peak 1.75x vs Mixtral-Offloading (HumanEval, 3090); min 1.12x vs AdapMoE (Wikitext, 4090) | baselines | - | Fig 9 caption
  - Average reduction vs each baseline | 34% (29-43%) vs Mixtral-Offloading; 19% (14-25%) vs MoE-Infinity; 12% (8-17%) vs AdapMoE | - | 1.52x / 1.23x / 1.14x implied | Sec 5.1
  - By GPU | average 1.41x (3090), 1.32x (4090), 1.21x (A100) over AdapMoE-class baselines | - | - | Sec 5.1
  - Model types | avg 2.3x; peak 3.5x vs Mixtral-Offloading, DeepSeek, A100; DeepSeek vs AdapMoE only 6.9% / 8.8% / 12.6% lower TPOT on 3090/4090/A100 | - | 1.07-1.14x | Fig 10, Sec 5.1
  - Mixtral TPOT reductions 3090/4090/A100 | 42.8/45.2/43.6% vs Mixtral-Offloading; 19.8/29.9/29.4% vs MoE-Infinity; 13.2/14.6/9.5% vs AdapMoE | - | - | Sec 5.1
  - Hit rate (GPU-cached), average | Mixtral: MO 15.02, MI 15.87, AdapMoE 42.14, SP-MoE 20.89%; Phi: MO 27.60, MI 14.67, AdapMoE 49.43, SP-MoE 42.72%; DeepSeek: MO 17.67, MI 16.73, AdapMoE 21.85, SP-MoE 40.06% | - | SP-MoE hit rate is BELOW AdapMoE on Mixtral and Phi | Table 3 (columns read from table)
  - Expert prediction | cosine sim of draft/target attention outputs 94.59% (DeepSeek), 59.82% (Mixtral), 56.59% (Phi); top-1 expert prediction accuracy 88.94% DeepSeek, about 88% Mixtral and Phi (as text states) | - | - | Fig 7a/b, Sec 3
  - Ablation TPOT | baseline (SD offloading, draft len 1): 1.40 s Mixtral, 0.48 s Phi, 0.32 s DeepSeek; worker prefetch 1.68x/1.39x/1.96x; + batched I/O 1.80x/1.59x/1.96x | vanilla offloading+SD | as stated | Fig 12, Sec 5.4
  - Memory sweep DeepSeek, Env3 | SP-MoE TPOT 0.18 -> 0.10 s as GPU memory 7 -> 39 GB; Mixtral-Offloading 0.27 -> 0.10 s; AdapMoE flat 0.19-0.20 s | - | - | Fig 11, Sec 5.3
  - Draft share of decode time | 16.2% on average | - | - | Sec 2
- catalog and first-pass numbers:
  - [yes] Rejected draft tokens still use I/O bandwidth with offloaded experts; low acceptance wastes the gain: intro: 'when the acceptance rate is low, loading experts for ultimately rejected tokens competes for I/O bandwidth' (first pass could not find it: wrong). But measured acceptance is 97-98%
  - [yes] proposes preloading during drafting: layer-to-layer draft-attention + target gate prefetch with cutoff layer
  - [no] [first pass] 'top-1 prediction accuracy 88.9% DeepSeek, 56-60% Mixtral/Phi': 56.59-59.82% are attention-output cosine similarities; the text gives about 88% top-1 accuracy for Mixtral and Phi as well (figure 7b not independently re-read)
  - [yes] [first pass] 336 MB (Mixtral) to 16 MB (DeepSeek) per expert; 1.35x avg, peak 1.75x, min 1.12x, 2.3x avg, peak 3.5x; TPOT -6.9% to -71.4%: confirmed (150 MB for Phi-MoE added)
  - [yes] [first pass] A100 40 GB, host RAM 512 GB-1 TB: Table 2
- transfers: no
- why: Experts in host DRAM over PCIe 4.0 x16 (32 GB/s), a GPU does compute, small models (16-336 MB experts), a trained same-family draft model with 97-98% acceptance, chain not tree, baselines are other GPU offloaders. It is a prefetch idea (draft gate predicts target experts), which needs a same-family small draft and a GPU/CPU overlap; our 125B on CPU has no such draft and its decode is I/O bound by NVMe where prefetching only overlaps, cannot add bandwidth. The paper's own Table 3 shows its hit rate is lower than AdapMoE's on 2 of 3 models, so the gain is overlap, not a better cache.
- quality_loss: None claimed (exact verification with greedy accept). No quality benchmark.
- ablation: Yes (Fig 12): vanilla prefetch vs worker prefetch vs +batched I/O; cutoff layer sweep (Fig 14, U-shaped for Mixtral/Phi); draft token length sweep (Fig 13: gain shrinks as draft length grows because verification touches nearly all experts); GPU memory sweep (Fig 11). Prefetch vs cache is partly isolated by Table 3.
- limitations: Authors: tree-based drafting and sampling-based decoding not supported; larger batch amplifies I/O bottleneck; future: batching/multi-tenant. Draft token length 1 baseline.
- contradicts_our_findings: No. Fig 13 text agrees that longer drafts reach full-expert activation so prediction accuracy matters less.

## P-0039: Utility-Driven Speculative Decoding for Mixture-of-Experts (Cascade) (Anish Saxena, Po-An Tsai, Hritvik Taneja, Aamer Jaleel, Moinuddin Qureshi; Georgia Tech / NVIDIA Research; arXiv 2506.20675v1, 17 Jun 2025, cs.DC)

- id read: 2506.20675v1 (PDF). Venue MICRO 2025 is NOT visible in the PDF text (first pass took it from the HTML header); UNVERIFIED here.
- raw text read: True
- setup: vLLM v0.6.6 (v0.8.2 for EAGLE), single-batch. RTX 6000 Ada 48 GB + i9-14900K 24 cores 64 GB. Models: Mixtral-8x7B FP8, Phi-3.5-MoE FP8, OLMoE FP8, DeepSeek-V1 FP16, Qwen-1.5-MoE FP16 (all resident in GPU, no offloading). Drafter: n-gram (K up to 3 static baselines; sweep K 0-7 for utility fit); EAGLE only on Mixtral. Tasks: code, math, extraction + mixed workloads (10 min, >=20k tokens). Baselines: static K=1,2,3 and no speculation.

- numbers (what | value | baseline | ratio | where):
  - Verification data movement increase | 2-3x (up to 3x for Mixtral at 3 draft tokens: 6 of 8 experts) | no speculation | - | Abstract, Sec 1
  - Static-K worst-case slowdown | 26% / 38% / 54% for K=1/2/3 | no speculation | 0.79 / 0.72 / 0.65x of baseline speed (1/(1+x) if slowdown is time) | Sec 7, Fig 13
  - Cascade worst-case slowdown | 5% (14% without adaptive back-off) | no speculation | - | Sec 5.5, 7.1
  - Cascade gain over best static K | 7-14% on average (abstract); 7-15% (Fig 13 caption); Mixtral +18% over static K=2 | static K | - | Abstract, Fig 13, Sec 7
  - Mixtral code | Cascade +57% speedup | static K=3 +39% | 1.12x better than static | Sec 7
  - OLMoE | Cascade 29% | static K=2 31%, K=3 33% | 0.97x of best static | Sec 7
  - DeepSeek | marginal 1% gain | - | - | Sec 7
  - Utility predicts speedup | R^2 = 99.4% over 120 points (5 MoEs x 3 tasks x 8 K values) | - | - | Fig 8
  - Max gain vs no speculation | up to 1.6x | no speculation | - | Fig 13 caption
  - n-gram vs EAGLE | EAGLE static K avoids n-gram slowdowns (ETR 1.7 vs 1.3 on math at K=1) | - | - | Sec 7.3
- catalog and first-pass numbers:
  - [yes] Speculation can slow MoE serving because of expert union cost: up to 1.5x slowdown (54% loss) for Mixtral math K=3
  - [yes] decides when to speculate by utility: utility = ETR gain / verification cost; disable if <1
  - [yes] [first pass] 26/38/54%, 2-3x, 5%, 7-14%, 57 vs 39%, 29 vs 31/33%, R^2 99.4% over 120 points: all confirmed
  - [yes] [first pass] RTX 6000 Ada 48 GB, i9-14900K 64 GB, FP8/FP16 split: Sec 6
- transfers: no
- why: All models fit on the GPU (no offloading); FP8 weights; the bottleneck is HBM bandwidth; n-gram drafts. The paper itself says offloading/caching works are 'orthogonal' (Sec 8) and gives no disk experiments. The idea (adaptive K from measured utility) is portable but its thresholds would need re-measuring in a disk-bound regime where verification cost per extra expert is NVMe time, not HBM time.
- quality_loss: None (lossless; rejection sampling).
- ablation: Adaptive back-off ablation (14% -> 5% worst case), test vs set phases, K search. No cache/offload ablation.
- limitations: Slightly worse than best static K on OLMoE; testing phases cost performance; EAGLE case only on Mixtral; draft-model-based overhead grows 5% per unit K (Sec 7.3).
- contradicts_our_findings: No. Supports adaptive-K conclusion; offers no number for a cached-LRU disk path.

## P-0040: Faster Cascades via Speculative Decoding (Harikrishna Narasimhan, Wittawat Jitkrittum, Ankit Singh Rawat, Seungyeon Kim, Neha Gupta, Aditya Krishna Menon, Sanjiv Kumar; Google Research; arXiv 2405.19261v2, 21 Oct 2024 (document dated 23 Oct 2024), cs.CL)

- id read: 2405.19261v2 (PDF, full 27+ pages including appendix E). Venue 'ICLR 2025' in the catalog is NOT in the PDF; UNVERIFIED here (from memory: ICLR 2025).
- raw text read: True
- setup: Dense models only. T5: small 77M drafter with T5-large 800M or T5-XL 3B verifier, supervised fine-tuned on WMT EN->DE, XSum, CNN/DM; Gemma v2 2B drafter with 9B/27B verifier few-shot on 8 benchmarks (WMT, CNN/DM, GSM8K, MBPP, SQuAD2, WebQuestions, NaturalQA, TriviaQA). Latency measured as wall clock, batch 1, 500 validation examples, 3 trials, same TPUv4 device, no model parallelism, block size gamma=5, T=1 for Table 2. Baselines: SeqCascade [Chow], TokenCascade [Chow], SpecDecode [Lossy], BiLD*, SpecCascade [Chow/Diff/OPT/Token].

- numbers (what | value | baseline | ratio | where):
  - Table 2 latency speed-up when matching the large model's quality, small->large WMT/XSum/CNNDM | SpecCascade[Token] 1.85x / 1.18x / 1.89x | SeqCascade 1.55/0.84/0.98x; TokenCascade 1.03/0.93/1.40x; SpecDecode[Lossy] 1.61/1.10/1.57x; BiLD 1.34/1.04/1.38x | vs lossy SpecDecode 1.15 / 1.07 / 1.20 | Table 2, p.9
  - Same, small->XL | SpecCascade[Token] 2.50x / 1.40x / 1.89x; SpecCascade[OPT] 2.61 / 1.34 / 2.21x | SpecDecode[Lossy] 2.17 / 1.28 / 2.07x | 1.15 / 1.09 / 0.91 (Token vs lossy SD) | Table 2
  - Best quality without exceeding the large model's latency, WMT small->large | SpecCascade[Token] BLEU 22.50 | SpecDecode[Lossy] 17.26; BiLD 16.49; SeqCascade 16.56 | +5.2 BLEU over lossy SD | Table 2
  - Gemma 2B->27B | SpecCascade[Token] matches the 27B's quality at lower rejection rate on 7 of 8 tasks (all except TriviaQA), often better than 27B | SpecDecode[Lossy], SpecCascade[OPT] | - | Fig 3, Sec 6
- catalog and first-pass numbers:
  - [yes] Confidence-based deferral to the large model only on hard tokens, combined with speculative verification: Chow-style deferral implemented through speculative execution
  - [absent] (ICLR 2025): not present in v2 PDF text
  - [partly] [first pass] Table 2 numbers; 'hardware not stated in what I read': numbers right; hardware is TPUv4 (appendix E), batch 1
  - [yes] [first pass] 'sequential cascades do better at low latency': Sec 6 text
- transfers: no
- why: Dense T5/Gemma, no MoE, no offloading, TPU; the gain is a quality-cost trade (outputs may differ from the large model; sometimes better than it), not a lossless speedup. For us the gate half (draft emits a token without the big model when the draft is confident) would let tokens skip expert I/O entirely, but their deferral thresholds were tuned on dense T5/Gemma pairs and quality for a 125B MoE vs a small draft is unmeasured.
- quality_loss: By design the output distribution differs from the large model's (lossy); quality is reported as BLEU/ROUGE-2/accuracy at matched latency; SpecCascade[Token] is above the large model's quality in several cells (22.50 BLEU vs large-model quality line in Fig 2). Not distribution-preserving.
- ablation: Compares deferral rules (Chow, Diff, OPT, Token), block size gamma (Fig 7), temperature (Fig 6). No MoE/offload element.
- limitations: Authors: higher throughput but higher total compute cost than sequential cascades; plug-in estimators instead of a trained router; only two-model cascades.
- contradicts_our_findings: No.

## P-0041: hackmd note by thc1006 'Tested every llama.cpp speculative-decode mode on Qwen3.6-35B-A3B + RTX 3090: none of them are faster than baseline' with update banners v2 (2026-04-22), A100 (04-26), v3 (05-08), v5 (05-17); repo links thc1006/qwen3.6-speculative-decoding-rtx3090 and qwen3.6-vllm-2x3090 (repos not opened)

- id read: hackmd .md export (raw, about 19 KB)
- raw text read: True
- setup: Qwen3.6-35B-A3B UD-Q4_K_XL (about 3B active, 8-of-256 routed experts), llama.cpp commit 9789512 then master bcb5eeb64, single RTX 3090 24 GB, whole model on GPU, Q8_0 KV, greedy/temp 0.5, batch 1, 5-12 prompts, 200-1000 tokens, N=3 reproduction (stdev 0.1). Modes: ngram-cache, ngram-mod, classic -md with Qwen3.5-0.8B draft (vocab-matched), DFlash drafter (PR #22105). Later: 2x A100 NVLink vLLM, 2x 3090 vLLM with MTP k=1..3.

- numbers (what | value | baseline | ratio | where):
  - v1 baseline | 135.7 tok/s | - | - | v1 table
  - v1 ngram-mod n8-24 | 129-131 tok/s at 100% accept | 135.7 | 0.95-0.97 (-3 to -5%) | v1 table
  - v1 ngcache / draft-model / ngram-cache | 121.3 / 120-121 / 119.1 mean, min 59-67 | 135.7 | 0.89 (-11%); min 0.44-0.49 | v1 table
  - v2 baseline | 139.9 tok/s | - | - | v2 table
  - v2 -md defaults | 55.3-56.5 | 139.9 | 0.40 (-60%) | v2 table
  - v2 --draft-min 2 --draft-max 32 | 65.0 | 139.9 | 0.46 (-54%) | v2 table
  - v2 --draft-min 48 --draft-max 64 | 85.6 | 139.9 | 0.61 (-39%) | v2 table
  - v3 DFlash draft-max 8 / 16 / 4 | 77.0 / 65.8 / 74.9 | 138.9 | 0.55 (-44.6%) / 0.47 / 0.54 | v3 table
  - A100 NVLink vLLM MTP (clean A/B) | -11.4% decode | no MTP | 0.89 | 04-26 banner, later retracted as a flag/prefix-cache confound
  - vLLM MTP k=1..3, 2x3090, prefix cache off | +27.5% decode TPOT | no MTP | 1.275 | v3 banner
  - Counter-evidence quoted | srogmann PR #20075 on Qwen3.5-122B-A10B (10B active) +15-45% with the same machinery | - | 1.15-1.45 | v1 'Why'
- catalog and first-pass numbers:
  - [yes] On Qwen3.6-35B-A3B no speculative mode beat baseline in one config: true for all llama.cpp modes on the 3090 (v1, v2, v3 DFlash); not true for vLLM MTP (+27.5% after fixing flags)
  - [yes] about 54% slower in another: v2: Oleg's --draft-min 2 --draft-max 32 is -54% (65.0 vs 139.9) (first pass said not on the page: wrong)
  - [yes] one tester, not peer-reviewed: author's own errata (v5) list a classifier bug, stub system prompt and undisclosed vLLM confounds
  - [yes] [first pass] 'v1 -3 to -12%; v2 best draft-model 39% slower; v3 DFlash 45% slower; vLLM MTP +27.5%': confirmed; the 39% case is the draft-model --draft-min 48 --draft-max 64 config, DFlash -44.6%
- transfers: no
- why: Whole model on the GPU, 3B active, compute-bound verify; no disk I/O at all. Author attributes the loss to expert union vs HBM bandwidth but the page also shows the loss shrinks or flips with engine, flags and a bigger-active model (122B-A10B +15-45%), so it is weak evidence even for GPU-resident regimes. The one useful fact: for 10B-active models speculation did help on GPU; our model has about 6B active.
- quality_loss: None (lossless speculation).
- ablation: Controls listed: KV quant, output length, draft vocab, commit, stale build. No cache or disk variable.
- limitations: Author errata: prompt confounds, prefix-cache x MTP interaction (vllm #38182), 'hardware-class-independent' claim retracted, N=1-5 prompts.
- contradicts_our_findings: No (direction agrees with bytes finding, but the mechanism here is not bytes from disk).

## P-0043: ReMoE: Boosting Expert Reuse through Router Fine-Tuning in Memory-Constrained MoE LLM Inference (Xiongwei Zhu, Xiaojian Liao, Tianyang Jiang, Yusen Zhang, Liang Wang, Limin Xiao; Beihang University, Huawei; Proceedings of the 43rd ICML, Seoul, PMLR 306, 2026; arXiv 2605.27081v1, 26 May 2026)

- id read: 2605.27081v1 (PDF, including appendices E-F)
- raw text read: True
- setup: DeepSeek-V2-Lite (15.7B, 2.4B active, 27 layers, 26 MoE, 64 routed + 2 shared, top-6) and Qwen1.5-MoE-A2.7B. Gate-only fine-tuning, 2000 steps, AdamW lr 5e-5, seq 2048, micro-batch 1 x accum 8, OpenHermes-2.5 (100k samples), about 7.9 h on one 80 GB GPU; loss = cross-entropy + temporal reuse (lambda 0.2) + smoothness/lag/window terms + Trust-KL anchor (lambda 0.45). Evaluation: (a) trace simulator, 128 ShareGPT prompts, greedy, 64 new tokens, per-layer LRU/LFU/FIFO/Belady with cache C=4..12 experts/layer and request-level reset; (b) vLLM expert offloading, RTX 3090 24 GB, PCIe Gen3 x16, max-num-seqs 1, moe-expert-cache-size 6; (c) Jetson Orin NX 16 GB, llama.cpp, model on aigo NVMe DP35 256 GB over PCIe Gen3 x4, -np 1, -n 128, --mmap (quantisation, cache size and SSD bandwidth not stated); (d) Iluvatar 2x BI-V150 + NVMe (expert-kit, cache 800 experts) and Kunpeng-920 + Ascend 910B3 + NVMe (llama.cpp).

- numbers (what | value | baseline | ratio | where):
  - Expert Overlap Ratio DeepSeek-V2-Lite | 34.5% | 27.3% (CE-only 22.9%) | 1.264x (+7.2 pts) | Table 2
  - EOR Qwen1.5-MoE | 0.2156 | 0.1695 | 1.272x | Sec 5.7
  - uHR LRU C=4 / 6 / 8 / 12 | 0.2374 / 0.3687 (C=6) / 0.4142 / 0.5035 | 0.2058 / 0.3187 / 0.3629 / 0.4519 | 1.154 / 1.157 / 1.141 / 1.114 (+0.032 / +0.050 / +0.051 / +0.052 abs) | Table 3 (uHR = unique-expert hit rate; second value in each pair is ReMoE)
  - Unique misses C=6 | 0.8068 M | 0.8707 M | 0.927 (-7.3%) | Table 3
  - Belady misses C=4 / C=6 | 802,903 / 635,069 | 843,832 / 684,671 | 0.951 / 0.928 | App F.1
  - vLLM offload, RTX 3090 | 3.88 tok/s, TPOT 242.99 ms, uHR 43.4% (text says 43.3%) | 3.58 tok/s, 254.31 ms, 39.4%; CE-only 2.95 tok/s | 1.084x; TPOT 0.955 | Table 4
  - Jetson TPOT ShareGPT / GSM8K / HumanEval (ms) | 306.27 / 346.04 / 337.61 | 554.69 / 613.73 / 672.68 | 1.811x / 1.774x / 1.993x (-44.8 / -43.6 / -49.8%) | Table 5
  - Jetson TTFT (ms) | 5368.77 / 4736.70 / 5233.11 | 7150.12 / 6041.65 / 7185.78 | -24.9 / -21.6 / -27.2% | Table 5
  - Paper's own TPOT proxy (4 GB/s I/O model), median, C=6 | 468.8 ms | 508.6 ms | -7.8% (C=4 -4.2%, C=12 -8.8%) | Table 10 (App F.1)
  - Iluvatar + NVMe decode | 1.66 tok/s | 1.08 | 1.54x (EOR 0.2763 -> 0.3454, +25%) | Sec 5.7
  - Kunpeng + Ascend + NVMe | 4.8 tok/s | 4.3 | 1.116x (EOR +30.2%) | Sec 5.7
  - Inference-time rerouting alternative, C=4 LRU | beta=1.0 uHR 41.66%, PPL 10.60; beta=4 uHR 63.88%, PPL 3607.92 | ReMoE router beta=0 uHR 23.74%, PPL 6.35 | hit x1.75 at +67% PPL; beta=4 collapse | Table 11 (App F.2)
  - Quality: lm-eval | MMLU 57.81, HumanEval 29.27, GSM8K strict 38.13, IFEval strict 16.08 | 57.72, 26.83, 38.89, 17.19 | +0.09, +2.44, -0.76, -1.11 pts | Table 6
- catalog and first-pass numbers:
  - [yes] +26% expert reuse rate: EOR +26.4% relative (27.3% -> 34.5%), Qwen +27.2%; +7.2 points absolute
  - [yes] +15.7% LRU hit rate: relative gain in unique hit rate at C=6 (0.3187 -> 0.3687); absolute +5.0 points; it is uHR (distinct experts), not token hit rate
  - [yes] TPOT -43.6 to -49.8% under llama.cpp on Jetson Orin NX: Table 5; but see why: unexplained by the paper's own I/O model
  - [partly] This is exactly the idea: it is router fine-tuning for temporal reuse (needs training), not an inference-time swap; the catalog's 'from search snippets' numbers are right
  - [yes] [first pass] quality MMLU +0.09, HumanEval +2.44, GSM8K -0.76; vLLM +8.4% (3.58 -> 3.88); 7.9 h on one 80 GB GPU; CE-only EOR 0.2293 vs 0.2730: confirmed (0.2293/0.2730 are the Sec 5.6 figures, Table 2 lists 22.9/27.3%)
  - [no] [first pass] 'cache 6 experts/layer for Jetson': 6 per layer is the vLLM and simulator setting; the Jetson llama.cpp --mmap run has no stated expert cache size (OS page cache)
- transfers: partly
- why: Closest evidence for our regime (SSD, llama.cpp, batch 1, small cache), but: (1) 15.7B/64 experts/top-6 vs 125B/512/top-10; (2) router fine-tuning requires training the gates of a 125B model plus calibration data; (3) the effect on hit rate is modest (+0.05 absolute at C=6, a perturbation much smaller than our simulated f=0.5 swap which moved hit 0.549 -> 0.763 at 2 footprints); (4) the paper's own I/O proxy (Table 10, 4 GB/s) predicts only -4% to -9% TPOT, the vLLM measurement gives -4.5%, yet the Jetson measurement reports -44% to -50% TPOT and also -22% to -27% TTFT, which the cache-miss accounting cannot explain (mmap/page-cache effects, confounds or unreported cache/RAM settings are plausible; my inference); (5) Jetson baseline decodes under 2 tok/s so the system is extremely miss-bound.
- quality_loss: Table 6 (DeepSeek-V2-Lite): MMLU +0.09, HumanEval +2.44, GSM8K strict -0.76 / flex -0.68, IFEval strict -1.11 pts; Qwen1.5 in Table 14 'comparable'. Entropy/CV: routing concentrates (CV 0.041 -> 0.161). CE-only control worsens things (EOR 22.9%, vLLM 2.95 tok/s). Inference-time rerouting alternative: PPL 6.35 -> 10.60 at beta=1.0 with uHR 41.66%.
- ablation: Yes: CE-only (generic fine-tuning), w/o Trust (EOR 0.388 but PPL 3.228 -> 3.263), w/o Reuse (EOR 0.283), w/o Consistency (EOR 0.329) in Table 7; LRU/LFU/FIFO/Belady cache policies; composability with mild cache-conditional rerouting (beta=0.5: uHR 34.07% vs 32.07%, PPL 6.51 vs 6.97). No ablation separates cache size from layout or prefetch beyond C sweep.
- limitations: Requires fine-tuning on OpenHermes-2.5 (chat); the paper concedes the vLLM result is the conservative one ('PCIe path partially hides miss cost'); only two small models; Jetson settings under-specified; chat-tuned checkpoint shows the same locality gap (App F.3); aggressive inference-time rerouting collapses quality (Table 11).
- contradicts_our_findings: Partly informs: (a) our simulated stickiness f=0.5 (hit +21 points) is not shown quality-free anywhere; ReMoE Table 11 shows a +17-18 point uHR gain by inference-time biasing already costs PPL 6.35 -> 10.60; (b) ReMoE's own proxy gives 4-9% TPOT, i.e. far below our 1.33x serial prediction at f=0.5, consistent only because their perturbation is much smaller.

## Per-family sections

### Family s1 #4: k same-sequence positions processed expert-major (P-0032, P-0033, P-0034, P-0035)
Further prior art found and read (abstract plus the relevant tables/sections of the PDF text):
- SpecMoEOff, Accelerating MoE Inference by Hiding Offloading Latency with Speculative Decoding (arXiv 2508.21706v2, Nanjing Univ.): up to 2.5x decode throughput over MoE offloading; large-batch, CPU/GPU roofline, chunked CPU attention kernel; not batch 1 (abstract/intro read; evaluation tables not read in full).
- MoE-SpeQ / Specter (arXiv 2511.14102): quantised draft model for multi-step expert lookahead plus prefetch; 1.50x average, up to 1.88x TPOT over offloading baselines (abstract and intro read, A100-class GPU).
- EVICT (2605.00342), EcoSpec 'Less Experts, Faster Decoding' (2607.12696, up to 1.62x), The Limits of Speculation (2609.22156, offline oracle; A100): downloaded, only abstracts/intros read; all argue the same union-cost point, none disk-bound CPU.
- Cited inside AcceptMoE (not opened): XShare 2602.07265, Opportunistic expert activation 2511.02237, Self-Speculative Decoding for On-device MoE (Zheng, Xu, Wang 2026), MoE-SpAc 2603.09983, MoESD 2505.19645.
What is new in our variant: none of the above studies a CPU-only, NVMe-streamed, batch-1 decode with an LRU of only 2-4 footprints; all baselines are either cache-less (DraftExpert), GPU slot caches covering 37.5% of experts (AcceptMoE), or HBM-resident. Our finding that k>1 gives no byte gain once an LRU exists is independently supported by AcceptMoE Table 2 (Standard SD moves 2.7-2.8x the bytes of AR with a lower hit rate) and by the Reddit write-up (trees read 2.8-3.3x the accepted path). The closest papers are AcceptMoE and DraftExpert.

### Family s1 #5: small draft + union verify + confidence gate (P-0037 to P-0041)
Further prior art: DraftExpert, SP-MoE, SpecMoEOff, Specter (above), MoE-Spec, Cascade; cascade/deferral literature (Faster Cascades read in full; its gate acts on dense T5/Gemma, lossy). Nobody combines a resident small draft with a confidence gate that skips the big model's expert I/O for a disk-streamed MoE; closest are Faster Cascades (the gate, dense) and SP-MoE/DraftExpert (draft-predicted prefetch, GPU). New in ours: gate-skip against expert I/O, which is lossy and unmeasured for quality; the byte side is dead per A.md and per every paper read.

### Family s1 #6: router with temporal stickiness (P-0043)
Further prior art read: Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference (Skliar et al., arXiv 2412.00099, TMLR June 2025): training-free cache-prior logit boost z' = z + lambda*delta_avg*mask; reports cache miss rate down by over 50% with PPL +0.1-3% and downstream accuracy <0.1%, 2x on-device speedup on Snapdragon phones with llama.cpp; the cache holds 30 or 45 of 60 experts per layer (50-75% of the pool), a much larger share than ours. ReMoE Table 11 shows the same trick at C=4 costs PPL 6.35 -> 10.60 at beta=1.0. Also cited by ReMoE: Oracle-MoE (locality-preserving routing, from-scratch training; not opened). SliceMoE 2512.12990 (surfaced by search, not read). New in ours: none for the idea itself (ReMoE trains it, Skliar does it at inference); our contribution would be only the small-cache regime on 512 experts and a quality measurement.

## Corrections to the first pass (P02.json)
- P-0035: the page DOES state greedy is I/O-optimal; added spec-tree numbers (2.8-3.3x, R*=1.51) from the linked ARTICLE.md; comment was a challenge to 'ai slop ram numbers'.
- P-0038: top-1 prediction accuracy about 88% for all three models; 56-60% were attention-output cosine similarities; the catalog's first clause is in the paper (intro); Table 3 shows SP-MoE hit rate below AdapMoE on Mixtral (20.9 vs 42.1%) and Phi (42.7 vs 49.4%).
- P-0041: the catalog's about 54% slower is on the page (v2, -54%).
- P-0032: baseline AR has no persistent expert cache (not 'unstated LRU').
- P-0033: Table 1 offload row accuracy 90.25 = -0.65 vs SD and -0.76 vs AR; 0.27 is the resident config / pruning delta.
- P-0040: hardware is TPUv4; venue not in PDF.
- P-0043: Jetson run has no stated 6-expert cache; paper's own proxy predicts only -4 to -9% TPOT.
- P-0039: venue MICRO 2025 not in PDF.
