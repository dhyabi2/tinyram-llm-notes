# Deep re-check P02b (raw full text, no digests)

Batch P02b: rows P-0044, P-0045, P-0046, P-0047 (family s1 #6) and P-0049, P-0050, P-0051, P-0052, P-0053 (family s1 #7). First-pass files: P02.md / P02.json. All text below was read from raw PDFs / API JSON in the scratch folder; numbers are copied from tables and figures with locations; ratios are computed by me.


## P-0044: Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference  (family s1 #6)

- Authors / venue / date: Andrii Skliar, Ties van Rozendaal, Romain Lepert, Todor Boinovski, Mart van Baalen, Markus Nagel, Paul Whatmough, Babak Ehteshami Bejnordi (Contextual AI / Qualcomm AI Research). 'Published in TMLR, June 2025' on the PDF header; arXiv 2412.00099v2, 24 Jun 2025 (PDF read = v2; first pass cited v3).
- Read: arXiv 2412.00099v2 (PDF, 26 pages, pdftotext); raw text read = True

- Setup read from the paper: Training-free 'Cache-Prior' routing: logits z' = z + lambda * Delta_avg * m_t, where m_t is the bitmask of experts currently in the LRU cache and Delta_avg a running mean of the logit range; the biased logits only re-rank, the mixing weights use the original logits; optionally the top-J experts are always kept (Eq. 9, Fig. 3). Models (Table 1): Mixtral-8x7B (8 experts, top-2, 46.7B), Phi-3.5-MoE (16, top-2, 41.9B), DeepSeek-V2-Lite (64+2 shared, top 6+2, 15.9B), Qwen1.5-MoE-A2.7B (60+4 shared, top 4+4, 14.3B). Quality evaluation is simulation of the cache (Wikitext perplexity, MMLU 5-shot, GSM8K 8-shot) with cache = half of the experts (Appendix D adds one quarter). On-device (Sec. 4.5, App. C, F): llama.cpp on two Snapdragon / Android 14 phones, mlock on, batch 1; 12 GB phone with 2 GB reserved (10 GB usable), 4-bit Qwen1.5-MoE, cache 30 of 60 experts per layer; 16 GB phone, 8-bit Qwen, cache 45 of 60. Best LRU cache size found by sweep (Fig. 14: larger caches lose speed because the OS pages out KV/activations). 10 runs per box plot. Cache-prior applied only to decode (prefill unchanged). No absolute tok/s anywhere; every throughput is normalised to the best LRU = 1x.

- Numbers (what | value | baseline | my ratio | where):
  - Miss rate, original routing + LRU vs Cache-Prior lambda=0.5, Qwen1.5-MoE cache 30/60 | 35% -> 16% | 35% | 0.46 (-54%) | Fig. 9 (a table), p.13
  - same, DeepSeek-V2-Lite cache 32/64 | 28% -> 7% | 28% | 0.25 (-75%) | Fig. 9, p.13
  - same, Phi-3.5-MoE cache 8/16 | 22% -> 9% | 22% | 0.41 (-59%) | Fig. 9, p.13
  - same, Mixtral-8x7B cache 4/8 | 40% -> 21% | 40% | 0.53 (-48%) | Fig. 9, p.13
  - Mean expert lifetime in cache (tokens), original vs Cache-Prior | Qwen 26 -> 58; DS 19 -> 76; Phi 22 -> 55; Mixtral 5 -> 10 | original | 2.2x / 4.0x / 2.5x / 2.0x (text says '5-9x longer' for Qwen and DeepSeek: does not match its own table) | Fig. 9 and Sec. 4.6, p.13
  - On-device relative throughput, best Cache-Prior vs best LRU (12 GB phone/4-bit and 16 GB phone/8-bit) | 'over 2x' (box plots, 10 runs) | best LRU = 1x | >2x (no absolute tok/s given) | Fig. 1 right, Sec. 4.5 p.12
  - Throughput vs hit rate (lambda 0.1..0.9, 16 GB phone) | near-linear increase with hit rate | lambda=0 (LRU) = 1.0 | not tabulated | Fig. 8 left, p.12
  - Quality: Wikitext perplexity increase at lambda=0.5 | +0.1% to +3% (abstract) | original routing | n/a | Abstract; Fig. 4 curves
  - Quality: MMLU / GSM8K accuracy change | '< 0.1%' (abstract and conclusion) | original routing | n/a; Fig. 5/6 y-axis spans are 0.6-1.0 points absolute across the Pareto front (e.g. Phi-3.5 0.776-0.786), so the '<0.1%' cannot be checked from the figures | Fig. 5, Fig. 6
- Catalog-row one-line summary, itemised:
  - 'Up to 2x speedup over LRU on mobile': YES. 'over 2x' / '2x' vs best LRU (best cache size per phone); relative only
  - 'Qwen1.5-MoE, 10 and 16 GB': PARTLY. phones are 12 GB (limited to 10 GB usable by reserving 2 GB) and 16 GB; 4-bit on the first, 8-bit on the second
  - 'no retraining': YES. training-free (learned-delta variant in App. E is optional)
  - 'Inference-time version of the same idea': YES. it is lossy runtime re-ranking of the router, so quality is not preserved exactly
- Transfers to our regime (CPU-only, 8-32 GB, NVMe streaming, batch 1, 125B 512-expert top-10): PARTLY. Matches: batch-1 decode, llama.cpp, storage (flash) behind a RAM expert cache, throughput linear in miss rate (so a serial, I/O-bound model like ours). Breaks: (1) cache is 1/2 of the experts (1/4 in App. D) of a 60-64 expert 14-16B model, not about 4% (20 of 512 experts per layer = 2 footprints at top-10) of a 125B 512-expert model; the paper never goes below 25%; (2) the method needs router logits at runtime and top-J protection; our simulation swaps picks with no logits (random, f of the non-repeating picks), which is a different and more aggressive perturbation than a logit-ranked bias; (3) it is lossy: only measured on 14-47B models with 4-64 experts; (4) baseline 1x is the best LRU, whose cache size was capped by phone memory pressure, which inflates the headline.
- Quality loss: Wikitext perplexity +0.1% to +3%, MMLU and GSM8K below 0.1% according to the abstract (lambda around 0.5); Table 2 (appendix) gives only qualitative text samples. Granular models (Qwen, DeepSeek) tolerate swaps better than Mixtral; large lambda hurts ('transient bias toward the cache state').
- Ablations: Yes, partly: Max-Rank and Cumulative-Sum-Threshold routing baselines (App. B), cache size 1/2 vs 1/4 (Fig. 11, 15), running-average vs calibrated logit range (Fig. 16), learned delta (Fig. 17), Wikitext/MMLU/GSM8K Pareto fronts vs optimal-oracle eviction (Fig. 4-6, 10). It beats the oracle (Belady) bound with original routing, i.e. routing change, not eviction, is what helps. No ablation of cache size on device beyond Fig. 14 (LRU only).
- Limitations stated by the authors: Authors state: very large lambda causes over-reuse; results rely on mlock and device memory behaviour; MoE design (granularity, shared experts) matters; they say retraining/architectural choices are out of scope. Only one model on phones.
- Contradicts our own findings (pubtech/A.md)?: No contradiction. Consistency check: our stickiness simulation f=0.5 at 2 footprints gives miss 0.237 vs 0.451 (ratio 0.53), the same size as the paper's Mixtral 0.53 and Qwen 0.46. But our test row says 'no router logits needed': the published method does need logits and a protected top-J, so our f is an upper-bound style proxy, not a reproduction. The paper's near-linear throughput vs hit rate supports our serial (not overlapped) figure of 1.33x rather than the overlap 1.12x on a machine without I/O overlap.
- Corrections to the first pass:
  - Date/version: PDF read is v2 (24 Jun 2025, TMLR header); first pass said v3.
  - First pass 'Qwen1.5 35%->16%, DS 28%->7%, Phi 22%->9%, Mixtral 40%->21%' verified (Fig. 9 table).
  - Text says experts stay '5-9x longer' for Qwen and DeepSeek, but its own table gives 2.2x and 4.0x (not in first pass).
  - 'MMLU and GSM8K drops under 0.1%' is an abstract claim; figures do not allow checking it.


## P-0045: Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models  (family s1 #6)

- Authors / venue / date: Jingcong Liang, Siyuan Wang, Miren Tian, Yitong Li, Duyu Tang, Zhongyu Wei (Fudan / USC / Huawei / Shanghai Innovation Institute). 'Published as a conference paper at ICLR 2026' on the PDF; PDF read = arXiv 2505.16056v4 (28 Feb 2026; first pass read v3 and could not see the venue).
- Read: arXiv 2505.16056v4 (PDF, 29 pages, pdftotext); raw text read = True

- Setup read from the paper: Analysis paper, no system: two metrics. SRP = best F1 of a segment router that fixes an expert set per m-token segment; SCH = hit rate of an oracle segment cache (cache size = rho x number of active experts, evicts the expert used least in the next m tokens). 20 MoE LLMs (3B-57B incl. Mixtral-8x7B, Qwen3-30B-A3B, DeepSeek-V2-Lite, OLMoE, Phi-3.5-MoE, Qwen1.5-MoE, Qwen2-57B-A14B, NLLB-MoE-54B, SwitchTransformers) plus pretrained 1.43B TOY models for ablations; corpus of mixed domains, 22,528 samples of 512 tokens. No hardware is given. The only systems experiment is App. F: a naive on-demand LRU implementation on the TOY models, throughput overhead vs full-GPU compared with SCH.

- Numbers (what | value | baseline | my ratio | where):
  - SRP (m=16, %) across 20 models | 19.27 (SwitchTF decoder) / 19.33 (encoder) ... 78.16 (LLaMA-MoE-v2) | - | - | Table 1
  - SRP, Mixtral-8x7B / Qwen3-30B-A3B / DeepSeek-V2-Lite / Qwen1.5-MoE | 49.36 / 54.14 / 37.92 / 30.71 | - | - | Table 1/2
  - TOY baseline, hit rate relative to clairvoyant optimum (=100), rho=1 | LRU 56.49, LFU 61.92, SCH(m=16) 80.97 | optimal 100 | LRU/SCH = 0.70 | Table 4
  - same, rho=2 | LRU 67.04, LFU 70.87, SCH 90.55 | optimal 100 | LRU/SCH = 0.74 | Table 4
  - same, rho=3 | LRU 75.26, LFU 78.35, SCH 96.23 | optimal 100 | LRU/SCH = 0.78 | Table 4
  - SCH (oracle, absolute %), rho=2, m=16, 95% CI | Qwen3 76.5-77.5; Mixtral 73.9-74.4; OLMoE 73.4-74.7; DeepSeek-V2-Lite 57.7-58.5; Qwen1.5-MoE 45.8-46.5; LLaMA-MoE-v2 96.9-97.2 | - | - | Table 8, App. E.5
  - Correlation of SCH with realised LRU / LFU hit rate across models | m=16: 90.43 / 88.70 | - | - | Table 3
  - Correlation of offloading overhead (naive LRU system, TOY) with SCH | r about +0.2 prefill, about -0.3 decode | - | weak | App. F, p.28-29
- Catalog-row one-line summary, itemised:
  - 'Local routing consistency is a model property': YES. SRP ranges 19-78 across models, with shared experts and many-small-expert designs lowest
  - 'cache of about 2x the active expert count balances hit rate and efficiency': PARTLY. true for group-1 models (turning point in SCH near rho=2); groups 3-4 (DeepSeek, Qwen1.5, Switch, NLLB...) have no turning point and SCH grows nearly linearly with rho (Sec. 5.1). The statement is about an ORACLE cache, not realised throughput
  - 'which matches our 2-footprint figure': PARTLY. rho counts per-layer cache entries / top-k, so rho=2 is 2 footprints in our terms; but the paper only says 2x is a knee for the most consistent models
- Transfers to our regime (CPU-only, 8-32 GB, NVMe streaming, batch 1, 125B 512-expert top-10): PARTLY. It supports the operating point loosely (2 footprints is a knee for consistent models) and its TOY result that LRU reaches 0.67 of the clairvoyant optimum at rho=2 (Table 4) would put the optimum for our model near 0.82 given our measured LRU hit 0.549, which is inside the range of the paper's oracle SCH values at rho=2 (0.46-0.97, Table 8) (derived guess, not measured). Breaks: oracle cache with look-ahead over m tokens; 512-token texts; no throughput, no hardware, no 512-expert model (largest 128 experts/layer); the only throughput check (App. F) is a naive LRU on TOY models and shows a weak correlation, which the authors themselves say means the model should not be the sole decider.
- Quality loss: Not applicable (lossless analysis). Notes that local routing consistency trades off against local load balance at training time (TOY NoLB SRP 56.4 vs OverLB 36.4, Table 2), i.e. any stickiness imposed by training/inference has a quality or balance cost.
- Ablations: TOY-model ablations of load balance, shared experts, expert combination space (Fig. 5, Table 2); domain vs vocabulary specialisation (Sec. 4, Fig. 15-18). Cache policy ablation only LRU vs LFU vs oracle (Table 4). No cache vs layout vs prefetch ablation.
- Limitations stated by the authors: SCH relies on an oracle; main text has no real throughput, only the App. F informal argument and a naive TOY implementation; authors say throughput depends on the implementation and other factors.
- Contradicts our own findings (pubtech/A.md)?: No. It is consistent with our LRU hit curve and with the claim that realised LRU is far below oracle. It cannot be used as evidence for or against a speed-up at 8-32 GB.
- Corrections to the first pass:
  - Venue: ICLR 2026 is printed on v4; the catalog was right, first pass could not see it on v3.
  - First pass 'size inconsistency 54B vs 57B' is not an inconsistency: NLLB-MoE-54B and Qwen2-57B-A14B are two different models (App. table).
  - First pass 'hardware not given in the part I read': there is none in the whole paper.
  - First pass omitted that the rho=2 knee exists only for group-1 models, and that the throughput appendix finds only r about -0.3 (decode).
  - SRP numbers (19 to 78, Mixtral 49) and TOY Table 4 (90.55 / 67.04 / 70.87) verified.


## P-0046: vLLM issue #38256: [RFC]: Incremental MoE Expert Offloading - GPU Cache + Async Pipeline  (family s1 #6)

- Authors / venue / date: Opened by e1n00r on 2026-03-26, state open, 24 comments through 2026-09-29 (GitHub API JSON read: issue body and all 24 comments).
- Read: https://api.github.com/repos/vllm-project/vllm/issues/38256 and /comments?per_page=100; raw text read = True

- Setup read from the paper: Not a paper. RFC for vLLM: ExpertWeightProvider with a fixed-size GPU slot cache (LFRU eviction score = freq / age) over CPU pinned memory; PR 1 sync H2D (PR #37190), PR 2 async + cross-layer prediction, PR 3 disk tier. Author data (tinyserve, own implementation): RTX PRO 2000 8 GB, GPT-OSS-20B MXFP4, 238 cache slots, single-stream decode. Comments add measurements from DGX Spark GB10 (unified, NVMe 6.9 GB/s), RTX 3050 Ti laptop, RTX PRO 6000 Blackwell, an Apple M4 (BigRig). All numbers are uncontrolled user reports.

- Numbers (what | value | baseline | my ratio | where):
  - Batched prefill dedup: expert loads for 3K context, top_k=4, 32 experts | 32 unique loads | 12K sequential (3K x 4) | 375x fewer (12,000/32 = 375) | issue body, 'Batched prefill'
  - tinyserve decode, GPT-OSS-20B MXFP4 on 8 GB GPU | 30 tok/s; hit 97-100% | HF device_map=auto | 160x (stated) | issue body 'Production data'
  - Cold start hit rate before warm-up | 48-56% for first 80-160 tokens | - | - | comment 2 (e1n00r, 2 Apr)
  - OLMoE (64 experts, top-8): unique experts per forward | decode step 8; single prefill 28-30; 6-prompt batch 57 of 64 | - | - | comment 7 (yasinyaman, 3 Aug)
  - Expert-split vs token-split prefill at 25% capacity (GB10, OLMoE) | 7.2 s -> 0.29 s | token split | 24.8x; but not bit-exact (greedy token identity 39.1% at cap 16 and 58.2% at cap 32 vs 100% for token split) | comment 7 and comment 19 (CRPONCELET)
  - Apple M4, Qwen3.6-35B-A3B: LFU pinning vs plain LRU | 1.4x to 2.3x worse | LRU | 0.43-0.71x | comment 12 quoting arjvnv/BigRig
  - same: miss rate vs resident experts per layer (9.7 GB ceiling) | 38 resident: 62.5%; 72 resident: 64.7%; decode 10.5 -> 9.9 tok/s | - | doubling residency moved miss rate by 2 points | comment 12
  - same: staging predicted experts during attention | 0.94-0.95x (slower) | no staging | 0.94-0.95 | comment 12
  - same: next-layer router applied to current hidden state, recall at three cutoffs (labels mangled by e-mail quoting) | 83.6 / 96.8 / 99.1% | - | - | comment 12
  - RTX PRO 6000, Qwen3.8-Flash-Next-FP8 (131B total / 6B active, top_k=8), 72 GiB VRAM pool + 36 GiB pinned tier | c1 33.56 tok/s; c32 140.32 tok/s; union 480 experts at c1 vs 7,336 at c32 | c1 | 4.18x aggregate for 32x concurrency | comment 11 (CRPONCELET)
  - same: miss cost for 23 experts (107.8 MiB) | host gather 14.08 ms vs 3.94 ms pinned H2D | - | 3.6x | comment 11
  - same: host tier 80 GB vs 16 GB at fixed 56 GB pinned cache | TPOT c1 21.5 vs 29.2 ms; c8 53.8 vs 186.8 ms | 80 GB tier | 1.36x / 3.47x | comment 19
- Catalog-row one-line summary, itemised:
  - 'Prefill touches nearly all experts': YES. OLMoE: single prefill 28-30 of 64, 6-prompt batch 57 of 64 (comment 7); 497-503 of 512 at 256 tokens in a 512-expert model (comment 19)
  - 'load each unique expert once (3K context example: 12K loads to 32 unique)': YES. issue body: '32 loads vs 12K sequential - 375x' (first pass wrongly said '12K to 32' did not appear)
  - 'Edge0 switches to whole-layer ...': WRONG (ATTRIBUTED TO THE WRONG SOURCE). the catalog text is cut off and no occurrence of Edge0 exists in the issue or its 24 comments; 'whole-layer' loading for prefill is a path of Edge0 (arXiv 2609.18063, Sec. executor paths), a different source
- Transfers to our regime (CPU-only, 8-32 GB, NVMe streaming, batch 1, 125B 512-expert top-10): NO. GPU slot cache over pinned host RAM, mostly batched or concurrent serving; only the prefill dedup idea overlaps with ours (and it is trivial: unique-expert loading). The anecdotal Apple-silicon evidence (LFU without ageing 1.4-2.3x worse than LRU, staging 0.94-0.95x, hit rate flat vs residency on a 256-expert model) is the closest to our regime (unified memory, SSD misses, batch 1) but is one user's measurement, unreviewed and with no controlled setup.
- Quality loss: Lossless caching, except expert-split prefill which is not bit-exact (token agreement 39-58% in greedy gen. at capacity 16-32 on one model; comment 19, labelled by its author as measured).
- Ablations: None controlled. Comments contain A/B tests by their authors (zero-copy vs fill path on GB10: 25.4/51.5/62.9 tok/s at C=1/4/8 vs 8.2/13.7/14.6; plain LRU +12-33% and plain LFU +7-39% more disk reads than LFRU on a 24k-call trace).
- Limitations stated by the authors: RFC lists: PR 1 not for latency-sensitive serving (sync H2D, enforce-eager), single GPU only, LFRU only, no bias tensors. Open questions on memory accounting.
- Contradicts our own findings (pubtech/A.md)?: Potentially: comment 12 (Apple M4, 256-expert model) reports miss rate nearly flat vs resident experts (62.5% to 64.7% from 38 to 72 per layer), while our LRU hit curve rises steeply (0.454 / 0.549 / 0.686 at 1.5 / 2 / 3 footprints). The model, routing entropy and the tuner-chosen cache sizes differ (the commenter says the model's routing is spread thin), so it is a flag, not a refutation. It supports our result that prediction/staging does not help when the bus is the bottleneck (0.94-0.95x).
- Corrections to the first pass:
  - First pass said the catalog's '12K loads to 32 unique' did not appear; it appears verbatim in the issue body ('32 loads vs 12K sequential - 375x').
  - 'Edge0 switches to whole-layer' is not in the issue at all; it comes from the Edge0 paper (2609.18063). The catalog row conflates two sources (and the CSV text is truncated).
  - First pass numbers (30 tok/s, 97-100%, LFU 1.4-2.3x worse, staging 0.94-0.95x, 83.6-99.1%, 6.9 GB/s, 7.2 s -> 0.29 s) verified; the three recall figures are at unknown cutoffs (mangled in the e-mail quote).
  - New: a 131B/6B-active top-8 model (same class as ours) on a 96 GB GPU gets 33.6 tok/s at c1 and 4.18x aggregate for 32x concurrency (expert union), and host gather (14.1 ms) dominates miss cost, not H2D (3.9 ms).


## P-0047: ExpertFlow: Efficient Mixture-of-Experts Inference via Predictive Expert Caching and Token Scheduling  (family s1 #6)

- Authors / venue / date: Xin He, Shunkang Zhang, Kaijie Tang, Shaohuai Shi, Yuxin Wang, Zihao Zeng, Zhenheng Tang, Xiaowen Chu, Haiyan Yin, Ivor W. Tsang, Yew Soon Ong (A*STAR CFAR / HKUST / HIT-SZ / HKBU / NTU). arXiv 2410.17954v2, 2 Apr 2026; ACM DAC '26 (Long Beach, 26-29 Jul 2026), 7 pages.
- Read: arXiv 2410.17954v2 (PDF, 7 pages; pdftotext raw mode); raw text read = True

- Setup read from the paper: System: (1) routing path predictor (transformer, trained per dataset; in-domain and cross-domain), (2) token scheduler that regroups tokens with similar predicted paths, (3) expert cache engine (prefetch predicted experts, correct mispredictions at runtime). Models (Table 1): Switch-32/64/128 (top-1), Mixtral-8x7B (8, top-2), Qwen1.5-MoE (60, top-4), DeepSeek-MoE (64, top-6). Hardware: one NVIDIA A40 48 GB + Xeon Gold 6338. Datasets: Alpaca, WMT16, XSUM, AIME2024. Baselines: Cache-MoE (= Eliseev & Mazur per-layer LRU with CPU fallback on miss), SE-MoE and Pregated-MoE (Switch only). Settings are (cache size CS = 2..16 experts per layer, batch size BS = 4..32), i.e. batched, GPU, experts in host RAM.

- Numbers (what | value | baseline | my ratio | where):
  - Throughput vs Cache-MoE (LRU), Mixtral-8x7B | up to 1.99x | Cache-MoE | 1.99x | Sec. 4.2.2 text, Fig. 6
  - same, Qwen1.5-MoE | up to 2.12x (2.21x on XSUM cross-domain at CS=4, BS=16) | Cache-MoE | 2.12x | Sec. 4.2.2, Fig. 6/7
  - same, DeepSeek-MoE | up to 1.94x | Cache-MoE | 1.94x | Sec. 4.2.2
  - Switch-128, CS=16 to 4, vs SE-MoE | 5.86x -> 9.99x ('10x') | SE-MoE | 9.99x | Sec. 4.2.2
  - Hit ratio Switch-32 CS=16, BS=4 (ExpertFlow vs LRU) | 91.96% vs 76.61% | LRU | 1.20x, +15.35 points | Fig. 10
  - same CS=16, BS=8 / BS=16 | 88.84 vs 65.62; 85.91 vs 58.37 | LRU | 1.35x / 1.47x | Fig. 10
  - same CS=8, BS=4/8/16 | 86.35 vs 57.21; 79.88 vs 44.90; 71.89 vs 36.22 | LRU | 1.51x / 1.78x / 1.98x; max gap +35.67 points | Fig. 10
  - Token scheduler, Switch-32/64/128 | 881.91 / 788.30 / 735.35 tok/s | w/o TS 854.19 / 680.72 / 628.36 | 1.03x / 1.16x / 1.17x (text 1.03/1.15/1.17) | Table 2
  - GPU memory vs all-in-GPU | Switch-128 15.26 -> 1.03 GB; Qwen1.5 35.21 -> 6.52 GB; DeepSeek 31.35 -> 6.38 GB; Mixtral OOM in AIG, 15.99 GB with offloading | all-in-GPU | -93.25% for Switch-128 (abstract says -93.72%, not reproduced from the text) | Sec. 4.2.4, Fig. 8
  - Router path predictor accuracy | 'up to 95%' (intro) | - | - | Sec. 1 contributions
- Catalog-row one-line summary, itemised:
  - 'Routing-aware cache reaches 91.96% hit': PARTLY. 91.96% is ExpertFlow on Switch-32 at CS=16, BS=4 (Fig. 10); not a general number
  - 'up to 61% better than LRU': PARTLY. 'outperforming LRU by up to 61.15%' is in the intro, but Fig. 10 gives at most +35.67 points (71.89 vs 36.22; text says '15-36% higher') and 1.98x relative; 61.15 cannot be reproduced from the data shown. The 91.96% and the 61% are not the same operating point (LRU there is 76.61%)
  - 'without changing the router': YES. experts that run are the router's; but tokens are regrouped by a scheduler and the predictor is a separate trained model
- Transfers to our regime (CPU-only, 8-32 GB, NVMe streaming, batch 1, 125B 512-expert top-10): NO. GPU (A40), batch sizes 4-32, 8-128 experts per layer, a separate trained transformer predictor per dataset, caches of 2-16 experts per layer on tiny-expert-count models; the cache-hit result is on Switch-32 (top-1, encoder-decoder). Not a router-stickiness method, so it is not evidence for family #6. Predictor training cost and generalisation to a 512-expert top-10 model are untested.
- Quality loss: None claimed (router unchanged); no quality table.
- Ablations: Hit ratio vs LRU only on Switch-32 (Fig. 10); token scheduler on Switch models only (Table 2: 1.03-1.17x). No ablation separating predictor accuracy from cache policy on Mixtral/Qwen/DeepSeek.
- Limitations stated by the authors: Memory saving is vs all-in-GPU; the baselines SE-MoE / Pregated-MoE run only on Switch; token scheduler benefit grows with expert count and is tiny on Switch-32; no stated limitations section.
- Contradicts our own findings (pubtech/A.md)?: No. It does not test temporal stickiness, and its largest hit-ratio gain is at large batch size where LRU collapses (BS=16, CS=8: 36%), which is a regime we do not have.
- Corrections to the first pass:
  - First pass 'hit 91.96% vs LRU up to 61.15% lower' merges two things: 91.96% pairs with LRU 76.61% (+15.35 points); the 61.15% claim in the intro is not reproducible from Fig. 10 (max +35.67 points).
  - Throughput 1.99/2.12/1.94x, 10x vs SE-MoE on Switch-128 (9.99x), A40 + Xeon Gold 6338 verified. Batch sizes 4-32 and CS 2-16 were not in the first pass.
  - Memory: text gives 15.26 -> 1.03 GB (-93.25%); the 93.72% of the abstract is not derivable from the text.
  - Token scheduler Switch-64 ratio computed 1.158x (text rounds 1.15x).


## P-0049: Fast Inference of Mixture-of-Experts Language Models with Offloading  (family s1 #7)

- Authors / venue / date: Artyom Eliseev, Denis Mazur (MIPT; Yandex School of Data Analysis; Yandex). arXiv 2312.17238v1, 28 Dec 2023 (technical report, 12 pages; no venue).
- Read: arXiv 2312.17238v1 (PDF, pdftotext); raw text read = True

- Setup read from the paper: Mixtral-8x7B(-Instruct). Per-layer LRU cache of k experts (k=2 on RTX 3060, k=4 elsewhere; 8 experts/layer, top-2) plus speculative loading (apply next layer's gate to the current hidden state). Mixed HQQ quantisation: shared attention 4-bit, experts 2- or 3-bit; embeddings, gates, norms stay 16-bit; the 2-bit scheme averages 2.6 bits/param with metadata. Hardware (Sec. 4.3): free-tier Colab T4 16 GB (PCIe3), RTX 3080 Mobile 16 GB (PCIe4) laptop, RTX 3060 12 GB (PCIe3) desktop, A100-80GB SXM (reference; also uses offloading), cloud T4 for 3-bit. Batch 1, OpenAssistant conversations, sampling. Expert offload tier = host RAM over PCIe; no SSD tier.

- Numbers (what | value | baseline | my ratio | where):
  - Table 2, 2-bit experts, full algorithm (tok/s) A100 / 3080 Mobile / 3060 / T4 | 3.061 / 2.655 / 2.278 / 2.092 | naive offloading (accelerate) 1.392 / 1.059 / 0.919 / 0.661 | 2.20x / 2.51x / 2.48x / 3.16x | Table 2, p.7
  - same, w/o expert pre-loading | 2.918 / 2.227 / 2.051 / 1.567 | w/o LRU and pre-loading 2.265 / 1.758 / 1.547 / 1.168 | LRU alone: 1.29x / 1.27x / 1.33x / 1.34x | Table 2
  - same, pre-loading on top of LRU | full vs w/o pre-loading | - | 1.05x / 1.19x / 1.11x / 1.33x | Table 2
  - Table 2, 3-bit experts, full algorithm (A100 / 3080M / 3060 / T4 cloud) | 2.845 / 2.475 / 2.038 / 1.603 | naive 1.246 / 0.914 / 1.791 / 0.580 (the 3060 naive value 1.791 looks like a typo) | 2.28x / 2.71x / (1.14x, suspect) / 2.76x | Table 2
  - MMLU 5-shot (Table 1): FP16 attention + FP16 experts | 70.51% | - | - | Table 1
  - MMLU: FP16 attn, experts 4/3/2-bit | 70.3 / 69.32 / 66.66 | 70.51 | -0.2 / -1.2 / -3.85 points | Table 1
  - MMLU: 4-bit attn, experts 4/3/2-bit (chosen config = 2-bit) | 69.11 / 68.47 / 65.58 | 70.51 | -1.4 / -2.0 / -4.93 points | Table 1
  - MMLU: 3-bit attn, experts 4/3/2; 2-bit attn, experts 4/3/2 | 66.54 / 65.79 / 61.83; 59.0 / 57.67 / 55.26 | 70.51 | - | Table 1 (right block)
  - Model size, 4-bit attn + 2-bit experts / + 3-bit experts | 17.54 GB (Wiki2 4.61, C4 7.42) / 21.37 GB (Wiki2 4.05, C4 6.87) | FP16 86.99 GB, ppl 3.59 | - | Table 1
- Catalog-row one-line summary, itemised:
  - 'Mixtral-8x7B with 2-bit experts (about 2.6 bpw with metadata)': YES. '2.6 bits per parameter' stated for the 2-bit HQQ scheme (group 16, scale group 128)
  - 'plus LRU offloading': YES. LRU cache k per layer; plus speculative loading, which the catalog omits
  - '2-3 tok/s on T4/3060/3080 Mobile': YES FOR 2-BIT. 2-bit: 2.09 / 2.28 / 2.66; 3-bit: 1.60 / 2.04 / 2.48
  - 'MMLU drop at 2-bit': YES. -3.85 points (FP16 attention) or -4.93 (4-bit attention, the chosen config); no 'accuracy at equal RAM' comparison
- Transfers to our regime (CPU-only, 8-32 GB, NVMe streaming, batch 1, 125B 512-expert top-10): PARTLY. Origin of '2-bit experts + LRU'. But the cache tier is GPU VRAM over PCIe host RAM, k is 2 or 4 of only 8 big experts per layer (k=4 is half of the experts, 2 footprints of top-2), the baseline is naive offloading not a bigger-cache 4-bit path at equal RAM, and the paper never isolates 'same RAM, fewer bits' (our byte-capacity simulation does). Their LRU-only gain (1.27-1.34x) is for a 25-50% cache, nothing like 4% of 512 experts.
- Quality loss: MMLU 70.51 -> 65.58 (-4.93 points) for the chosen 4-bit attention + 2-bit experts; 2-bit attention is much worse (55.26). Perplexity: Wiki2 3.59 -> 4.61, C4 6.52 -> 7.42 (Table 1, 4-bit attn + 2-bit experts). Data-free HQQ, no calibration.
- Ablations: Yes (Table 2): naive offloading vs no LRU/no pre-load vs LRU only vs full, on four GPUs and two bit-widths. LRU is isolated (1.27-1.34x), prefetch is isolated (1.05-1.33x), quantisation is not isolated from offloading policy at equal memory.
- Limitations stated by the authors: Authors list: future work on further offloading strategies and speculative prediction; the 'not possible to prefetch next layer reliably without changing architecture' footnote. Mixtral only; no disk tier.
- Contradicts our own findings (pubtech/A.md)?: No. Consistent in direction: LRU helps a lot when the cache is a large share of experts; our simulation finds the 2.6 bpw gain at the same RAM is real only when the cache is small (8 GB 1.49x serial, 1.08x overlapped). Their Table 2 ratios are against naive layer offloading and cannot confirm or refute the equal-RAM comparison.
- Corrections to the first pass:
  - All first-pass numbers (Table 2, Table 1 MMLU 70.51 / 66.66 / 65.58 / 61.83 / 55.26) verified against the raw text.
  - Missing in first pass: ablation ratios (LRU alone 1.27-1.34x, pre-loading 1.05-1.33x), the 3-bit column, and the odd naive 3-bit 3060 value 1.791 (higher than other naive entries, likely a typo).
  - Right-block labels: 61.83 and 55.26 belong to the 3-bit-attention and 2-bit-attention blocks with 2-bit experts; first pass wording was ambiguous.


## P-0050: HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference  (family s1 #7)

- Authors / venue / date: Peng Tang, Jiacheng Liu, Xiaofeng Hou, Yifei Pu, Jing Wang, Pheng-Ann Heng, Chao Li, Minyi Guo (SJTU / CUHK). arXiv 2411.01433v2, 6 Nov 2024 (ACM-style 'Conference'17' template; no venue).
- Read: arXiv 2411.01433v2 (PDF, 15 pages; Fig. 14 also read as a rendered image); raw text read = True

- Setup read from the paper: Built on llama.cpp (token-level dynamic expert loading, layer-level adaptive prefetching, sequence-level multidimensional cache with LRU+LFU+LHU priorities). On a cache miss, the expert's unimportance score s = f(gate) decides: high precision (importance high), low-precision copy (moderate), or skip. For Mixtral thresholds T1=0.6, T2=0.9 give 67% high precision, 30% low precision, 3% skipped (Sec. 3.2); top-1 always high precision. Models: Mixtral-8x7B (45B, 32 layers, 8 experts, top-2, 87 GB fp16) and Phi-MoE (42B, 16 experts, top-2). Hardware: RTX 4090 24 GB + 256 GB RAM + 64 cores + PCIe4 (float16 base, int4 replacement); Jetson AGX Orin 32 GB unified, model on Samsung 980 PRO NVMe (7 GB/s nominal, about 3 GB/s in practice) (int8 base, int2 replacement). Batch 1, 60 Alpaca prompts, in/out lengths 16 or 128 / 32 or 128. Baselines: llama.cpp (mmap), MoE-Infinity, MoE-Offloading (Eliseev&Mazur), Transformers, DeepSpeed, Fiddler.

- Numbers (what | value | baseline | my ratio | where):
  - Decode speed vs MoE-Infinity on Jetson Orin (NVMe) | 3.64x (Mixtral), 9.93x (Phi-MoE) | MoE-Infinity | as stated; bars read from Fig. 14a/b: Mixtral about 0.9 vs 0.25 tok/s, Phi-MoE about 4 vs 0.4 tok/s (approximate, eyeballed) | Sec. 5.2 p.10, Fig. 14
  - Decode speed vs llama.cpp (mmap) on Jetson Orin | 13.0x (Mixtral), 18.9x (Phi-MoE); prefill latency -79% / -80% | llama.cpp about 0.07 tok/s on Mixtral (read from Fig. 14a) | as stated | Sec. 5.2, Fig. 14
  - RTX 4090 vs MoE-Offloading / MoE-Infinity | 3.21x / 3.29x vs MO; 2.30x / 3.92x vs MI (Mixtral / Phi) | MoE-Offloading, MoE-Infinity | as stated; Mixtral HOBBIT about 2.1 tok/s (Fig. 14c) | Sec. 5.2
  - RTX 4090 + CPU vs llama.cpp / Fiddler | 1.31x / 1.42x vs llama.cpp; 0.99x / 1.46x vs Fiddler | llama.cpp, Fiddler | as stated | Sec. 5.4, Fig. 15
  - Abstract headline | 'up to 9.93x decode vs state-of-the-art offloading' | MoE-Infinity on Phi-MoE / Jetson | 9.93x (excludes llama.cpp 18.9x, which they do not count as an offloading system) | Abstract
  - Ablation: dynamic (mixed-precision) loading alone | 1.19x to 1.57x (largest on Jetson, smallest on RTX 4090 + CPU) | same system without dynamic loading | 1.19-1.57x | Sec. 5.5.1, Fig. 16
  - Ablation: adaptive prefetching | about 1.05x decode; about -10% prefill latency | no prefetch | 1.05x | Sec. 5.5.2, Fig. 17
  - Ablation: multidimensional cache policy | 4.69-8.68% lower miss penalty than LRU; 2.13-4.19% than LFU | LRU / LFU | - | Sec. 5.5.3, Fig. 18
  - Accuracy (Table 3): GSM8K Mixtral float16 -> float16+int4; int8 -> int8+int2 | 0.52 -> 0.51; 0.52 -> 0.51 | base | -0.01 each | Table 3
  - Accuracy: Phi-MoE GSM8K | 0.83 -> 0.82 (fp16+int4); 0.83 -> 0.83 (int8+int2) | base | -0.01 / 0 | Table 3
  - TruthfulQA (truth, info), Mixtral | 0.49,0.94 -> 0.49,0.93; 0.50,0.91 -> 0.52,0.90 | base | within 0.02 | Table 3
- Catalog-row one-line summary, itemised:
  - 'Loads INT4/INT2 versions of low-importance cache-miss experts': YES. int4 replacement on the 4090 (fp16 base), int2 on Orin (int8 base); plus a skip class (3% of experts for Mixtral)
  - 'up to 9.93x decode vs offloading baselines': YES. 9.93x is Phi-MoE on Jetson vs MoE-Infinity; vs llama.cpp it is 18.9x
  - 'adaptive prefetch': YES. only about 1.05x in decode, about -10% prefill
- Transfers to our regime (CPU-only, 8-32 GB, NVMe streaming, batch 1, 125B 512-expert top-10): PARTLY. Closest hardware class (llama.cpp, batch 1, NVMe, unified memory). But: (1) low-precision copies are used only for misses; hot (cached) experts stay high precision, and extra storage for multiple precision copies is needed, so it is NOT 'store everything at 2.6 bpw'; (2) the huge ratios are against weak baselines: llama.cpp mmap page-faulting (about 0.07 tok/s) and MoE-Infinity (about 0.25-0.4 tok/s) on a 3 GB/s NVMe; absolute HOBBIT speed is only about 0.9 tok/s (Mixtral) / 4 tok/s (Phi-MoE) on the Orin; (3) 8-16 large experts per layer, 42-45B models; (4) 3% of experts are skipped entirely (not only quantised).
- Quality loss: GSM8K -0.01 (Mixtral) and -0.01/0 (Phi-MoE); TruthfulQA within 0.02; the text says 'decreases by no more than 1%'. Two benchmarks only, only when 30% of loaded experts are replaced and 3% skipped, thresholds T1=0.6, T2=0.9 tuned per model.
- Ablations: Yes (Sec. 5.5): dynamic loading 1.19-1.57x (the part relevant to us), prefetch about 1.05x, cache policy 4.7-8.7% miss-penalty reduction vs LRU. Cache vs layout is not separated.
- Limitations stated by the authors: Stated: focus on GPU-centric computing; CPU helper mode gives smaller gains; Fiddler wins slightly on Mixtral (0.99x). Not stated: only two models, 2 benchmarks, weak baselines.
- Contradicts our own findings (pubtech/A.md)?: No. The ablation (dynamic loading 1.19-1.57x, biggest on the slowest link) is the same order as our byte-capacity result (8 GB: 1.35-1.72x serial, 1.08x overlapped) and shows the same dependence on link speed. The absolute numbers (about 0.9 tok/s Mixtral on Orin) confirm that NVMe-bound decode with 8-GB-class memory is far below our simulated 2-4 tok/s, but those are 87 GB-class fp16/int8 models, not 4-bit.
- Corrections to the first pass:
  - First pass numbers (3.64x/9.93x vs MoE-Infinity, 13.0x/18.9x vs llama.cpp, Table 3 GSM8K) verified.
  - Missing in first pass: 3% of experts are skipped (T1=0.6, T2=0.9; 67/30/3 split), ablation 1.19-1.57x for dynamic loading alone, absolute speeds read from Fig. 14 (about 0.9 tok/s on Orin for Mixtral).
  - First pass 'Jetson Orin 32 GB unified, int8+int2, NVMe about 3 GB/s' verified.


## P-0051: MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts (v1); renamed Specter: Fast MoE Inference via Quantization-Enabled Multi-Step Speculative Offloading (v3)  (family s1 #7)

- Authors / venue / date: Wenfeng Wang, Jiacheng Liu, Xiaofeng Hou, Xinfeng Xia, Peng Tang, Mingxuan Zhang, Chao Li, Minyi Guo (SJTU / HKUST). arXiv 2511.14102 v1 18 Nov 2025; v3 22 Oct 2026 (v3 code link is named specter_asplos; first pass said ASPLOS '27, I could not confirm the venue from the page text).
- Read: arXiv 2511.14102v1 and v3 (both PDFs read); raw text read = True

- Setup read from the paper: A GPTQ INT4 copy of the target runs as the draft model on the GPU, predicts several future tokens and thereby the experts of future layers; experts are prefetched over PCIe while drafting (Expert Lookahead Buffer, three-phase prefetch, layer-aware cache); target verified in parallel (lossless speculative decoding). The offloaded experts stay FP16 ('official FP16 weights for the target and a GPTQ-quantized draft'); quantisation applies to the draft only. Models: Phi-3.5-MoE (42B, 16 experts, top-2, 78 GB), Qwen1.5-MoE (14.3B, 60+1, top-4), DeepSeekV2-Lite (15.7B, 64+shared, top-6). Hardware: one A100 40 GB (PCIe 4.0 x16), 24-core Xeon Silver 4310, 256 GB RAM. Baselines v1: Hugging Face Transformers, Mixtral-Offloading (same cache SC and same memory SM variants); v3 adds SpecMoEOff, vLLM, llama.cpp. Datasets C4, WikiText-2, HumanEval, GSM8K, GPQA; single request plus batch scaling to B=32.

- Numbers (what | value | baseline | my ratio | where):
  - v1 abstract headline | 'at most 2.34x' over state-of-the-art offloading frameworks | - | - | v1 abstract, Sec. 1
  - v1 Phi-MoE average TPOT (high-memory setting) | 163.1 ms | Mixtral-Offloading-SC 536.7 ms; SM 351.8 ms; Transformers 960.5 ms | 3.29x vs SC; 2.16x vs SM; 5.89x vs Transformers | v1 Sec. 4.2 p.11
  - v1 Qwen / DeepSeekV2-Lite | 74.1 ms/token | SC / Transformers | 2.5x vs SC; 3.5x vs Transformers; DeepSeekV2-Lite 4.8x vs Transformers | v1 Sec. 4.2
  - v3 abstract headline | average 1.50x (up to 1.88x) TPOT speedup over SpecMoEOff | SpecMoEOff | - | v3 abstract, Sec. 4.2.1
  - v3 Phi-3.5-MoE average TPOT (high memory) | 163.3 ms | SpecMoEOff 292.1; Mixtral-Offloading-SM 295.1; vLLM 297.5; Transformers 960.6 | 1.79x / 1.81x / 1.82x / 5.88x | v3 Sec. 4.2.1
  - v3 Qwen1.5-MoE average TPOT | 35.3 ms | SpecMoEOff 66.4; MO-SM 74.2; vLLM 176.6; Transformers 258.2 | 1.88x / 2.10x / 5.0x / 7.3x | v3 Sec. 4.2.1
  - Cache hit rate DeepSeekV2-Lite at 16/24/32 GB | 99.85 / 98.62 / 96.25% | LRU reactive (v1 Table 3: 29.2% at cache 6; 50.94% at cache 16; LRU with scaled cache 61.13% / 76.56%) | - | v3 Sec. 4.4; v1 Table 3
  - Expert demand vs speculation depth K, DeepSeekV2-Lite | per-layer experts 6.0 (K=1) -> 31.7 (K=16) | K=1 | 5.3x, nearly half of 64 experts | v3 Sec. 2.3, Fig. 4
  - Cumulative ablation TPOT (DeepSeekV2-Lite) | 69.2 ms full; 83.5 w/o PIO; 94.4 w/o PIO+SIC; 130.4 w/o PIO+SIC+SEE | full | 1.21x / 1.36x / 1.88x | v3 Table 5b
  - H2D transfer hidden under compute (draft+verify) | 55.5% / 60.5% / 22.4% (DS-V2-Lite / Qwen / Phi) | - | - | v3 Table 5a
- Catalog-row one-line summary, itemised:
  - 'Quantized draft predicts experts for prefetch': YES. INT4 GPTQ draft; offloaded experts remain FP16
  - 'up to 2.34x': YES (V1 ONLY). v1 abstract 'at most 2.34x'; v3 says average 1.50x, up to 1.88x vs SpecMoEOff (different baseline)
  - '(3.3x Phi-MoE)': YES (V1 ONLY). 536.7 -> 163.1 ms = 3.29x vs Mixtral-Offloading-SC (not 2.34x: the v1 text gives both numbers against different baselines; first pass said it could not find 3.3x, it is in v1 Sec. 4.2)
  - 'over offloading': YES. baseline is Mixtral-Offloading (Eliseev and Mazur) re-implemented, then SpecMoEOff in v3
  - 'notes naive spec decoding can hurt MoE because of the expert union': YES. v1 Sec. 2: 'naively applying speculative decoding to MoE models is often counterproductive'; v3 Sec. 2.3 'expert explosion', 6.0 -> 31.7 experts per layer
- Transfers to our regime (CPU-only, 8-32 GB, NVMe streaming, batch 1, 125B 512-expert top-10): NO. GPU (A100) + PCIe + FP16 offloaded experts + GPU-resident INT4 draft; the prediction is a draft model forward, which on CPU-only 125B-class decode would cost a full extra small-model pass. Does not support 'requantise the stored experts'. The union-of-experts finding (K=16 touches nearly half of the experts) supports our own union measurements against speculative decoding at 512 experts.
- Quality loss: None for output (lossless verification); the draft's INT4 only affects prediction accuracy. Table 6 (v3) shows TPOT degrades 1.06x/1.31x/1.34x when 25/50/75% of prefetch targets are wrong (prefetch accuracy 0.785/0.575/0.410).
- Ablations: Yes (v3 Table 5b and v1 Table 4): fused kernel (13.02 vs 8.88 tok/s w/o), async prefetch (-5%), PIO/SIC/SEE components. Cache-policy comparison LRU vs LRU-scaled vs single prefetch vs speculative (Table 3 v1 / Fig. 17 v3).
- Limitations stated by the authors: v3 notes the benefit depends on memory regime; dataset diversity; v1 states that Mixtral-Offloading was reproduced and extended to other models.
- Contradicts our own findings (pubtech/A.md)?: No. It agrees with our finding that speculation widens the expert union (our k=2 speculative read 1.34x more bytes per accepted token than k=1 on a 2.6 bpw 8 GB cache).
- Corrections to the first pass:
  - First pass said the 3.3x / 536.7 -> 163.1 ms figure was not found in the text it read: it is in v1 (Sec. 4.2): 3.29x vs Mixtral-Offloading-SC (2.16x vs SM, 5.89x vs Transformers).
  - First pass said the 'naive spec decoding can hurt MoE' note was not verified: it is explicit in v1 Sec. 2 and v3 Sec. 2.3 (6.0 -> 31.7 experts per layer at K=16).
  - First pass v3 numbers verified (163.3 vs 292.1 ms; 1.50x avg / 1.88x max); the 1.88x maximum is the Qwen1.5-MoE ratio (66.4/35.3), Phi is 1.79x.
  - Hardware details added: A100 40 GB, Xeon Silver 4310 24-core, 256 GB, PCIe 4.0 x16.


## P-0052: Making Every Verified Token Count: Adaptive Verification for MoE Speculative Decoding (EVICT)  (family s1 #7)

- Authors / venue / date: Lehan Pan, Ziyang Tao, Xiao Wang, Zhimeng Zhao, Meng Wang, Yanyong Zhang (USTC; Tianyijiaotong Technology). arXiv 2605.00342v2, 5 Sep 2026.
- Read: arXiv 2605.00342v2 (PDF, 16 pages); raw text read = True

- Setup read from the paper: Training-free, lossless truncation of the EAGLE-3 draft tree before target verification: prefix chosen by maximising estimated accepted tokens divided by profiled verification cost C(k), with C(k) reflecting the growth of the activated-expert union. Targets: Qwen3-30B-A3B (main), Qwen3-235B-A22B (TP=8), Ling-flash-2.0 (TP=4), dense Hunyuan-7B; baselines vanilla AR, Lookahead, EAGLE-3, DDD. SGLang with CUDA graphs on A100 SXM4 80GB; draft tree steps=4, topk=8, 32 draft tokens; six benchmarks, T=0 and T=1; batch 1 main, vLLM batch 4-64 in App. C.5. No offloading at all.

- Numbers (what | value | baseline | my ratio | where):
  - Qwen3-30B-A3B average decode, T=0 | EVICT 202.82 tok/s (1.40x vs vanilla) | EAGLE-3 176.34 (1.22x); vanilla 144.46 | 1.15x over EAGLE-3 | Table 1
  - same, T=1 | EVICT 195.17 (1.38x) | EAGLE-3 170.11 (1.20x); vanilla 141.40 | 1.15x over EAGLE-3 | Table 1
  - Qwen3-30B-A3B GSM8K / HumanEval, T=0 | 258.35 / 265.52 | EAGLE-3 208.74 / 209.32 | 1.24x / 1.27x (the text's '1.23-1.27x' is for these high-acceptance benchmarks only) | Table 1, Sec. 4.2
  - Qwen3-235B-A22B average T=0 / T=1 | 147.69 / 143.96 | EAGLE-3 118.24 / 115.66; vanilla 81.89 / 80.82 | 1.25x over EAGLE-3 (1.80x/1.78x over vanilla) | Table 8
  - Ling-flash-2.0 average T=0 / T=1 | 279.96 / 265.76 | EAGLE-3 224.98 / 217.30; vanilla 175.58 / 171.01 | 1.24x / 1.22x over EAGLE-3 | Table 7
  - Maximum speedup over autoregressive | 2.35x | vanilla | 187.43/79.61 = 2.35x (GSM8K, T=0, Qwen3-235B); HumanEval T=1 gives 183.21/76.73 = 2.39x | abstract; Table 8
  - Verified tokens / activated experts / verification latency vs EAGLE-3 | -74.7% / -32.5% / -26.6% on average | EAGLE-3 | - | Fig. 5, Sec. 4.3 (these are EVICT's own numbers, not quoted from another paper)
  - Large batch (vLLM, QA), BS=64 | +34.7% over EAGLE-3, +39.1% over DFlash | same backbones | 1.35x / 1.39x | App. C.5, Table 6
  - Qwen3-30B-A3B activated experts vs tree size | grows with tree size 4 -> 32; figure axis labelled 'Total Experts = 64' | - | - | Fig. 1
- Catalog-row one-line summary, itemised:
  - 'Verification cost grows with the union of experts': YES. Fig. 1: more experts and higher latency with larger trees; target verification dominates latency (Fig. 2)
  - 'pruning draft nodes shrinks it': YES. -32.5% activated experts, -26.6% verification latency (Fig. 5)
  - 'filed under #7 (mixed-precision experts)': WRONG MAPPING. EVICT has nothing to do with expert precision; it belongs with the speculative-decoding families (#4/#5)
- Transfers to our regime (CPU-only, 8-32 GB, NVMe streaming, batch 1, 125B 512-expert top-10): NO. In-GPU serving on A100s with no offloading; reduces verification token count, not bytes read from a slow tier. Useful only as a data point that tree pruning shrinks the expert union far less than the token count (-74.7% tokens vs -32.5% experts), which is the same reason our speculative variants read more bytes.
- Quality loss: Lossless (verification unchanged); lower mean accepted tokens than EAGLE-3 in Table 2 by design.
- Ablations: Yes: eager vs CUDA-graph integration (EVICT slightly underperforms EAGLE-3 in eager mode because of CPU overhead), tree/threshold ablations (Sec. 4.4), dense-model check (Table 9: +2.6-11.3% on Hunyuan-7B).
- Limitations stated by the authors: Stated: depends on cost/benefit estimates and may be hurt by domain shift; benefit at large batch is workload dependent (HumanEval at high batch comparable to the untruncated backbone).
- Contradicts our own findings (pubtech/A.md)?: No.
- Corrections to the first pass:
  - First pass '1.23-1.27x over EAGLE-3 on Qwen3-30B-A3B across temperatures' is the figure for high-acceptance benchmarks (GSM8K, HumanEval) only; the six-benchmark average is 1.15x at both temperatures (202.82/176.34 and 195.17/170.11).
  - First pass attributed '74.7% fewer verified tokens, 32.5% fewer activated experts' to another paper (AcceptMoE); both numbers (and 26.6% lower verification latency) are in EVICT itself (Fig. 5).
  - Abstract: 2.35x max and 1.21x average (mean of 1.15, 1.22, 1.25 on the three MoE backbones) verified. Hardware: A100 SXM4 80 GB, TP=4/8 for the big models.
  - Mis-mapped to #7 (confirmed).


## P-0053: Bandwidth-Efficient Adaptive Mixture-of-Experts via Low-Rank Compensation  (family s1 #7)

- Authors / venue / date: Zhenyu Liu, Yunzhen Liu, Zehao Fan, Garrett Gagnon, Yayue Hou, Nan Wu, Yangwook Kang, Liu Liu (RPI / UMass Amherst / GWU / Samsung Semiconductor). arXiv 2512.17073v1, 18 Dec 2025 (8 pages).
- Read: arXiv 2512.17073v1 (PDF; Fig. 6-8 read as text and one page image); raw text read = True

- Setup read from the paper: Experts stored low-bit (INT2/INT3, GPTQ/HQQ-style) in host RAM; per-expert low-rank compensators (rank budget average 32 for Mixtral, 64 for DeepSeek; kurtosis-guided rank allocation) precomputed offline; at inference only the router's Top-n experts (n=1 for Mixtral, 3 for DeepSeek-MoE) get their compensator transferred and added. Models: Mixtral-8x7B, Mixtral-8x22B, DeepSeek-MoE-16B (64 experts, top-6 + 2 shared). Hardware: GPU-only = H100 PCIe 80 GB + DDR host memory (experts loaded on demand); GPU-NDP = H100 plus a SIMULATED near-data-processing device (512 GB/s, 512 GB) modelled with Ramulator following MoNDE. Input 256, output 128/256/512. Baselines: GPTQ, HQQ (accuracy); Mixtral-Offloading, HOBBIT (GPU-only speed); MoNDE (GPU-NDP speed).

- Numbers (what | value | baseline | my ratio | where):
  - GPU-only, Mixtral-8x7B decode (in 256 / out 512) | 12.27 tok/s (3-bit) and 18.11 tok/s (2-bit) | Mixtral-Offloading 2.37 tok/s | 5.18x and 7.64x (paper: 5.17x / 7.64x) | Fig. 7, Sec. 4.3
  - same vs HOBBIT | 25.43 and 36.97 tok/s | HOBBIT 6.75 tok/s | 3.77x and 5.48x | Sec. 4.3
  - GPU-only, Mixtral-8x22B | 5.18x (3-bit), 7.63x (2-bit) | Mixtral-Offloading 0.79 tok/s | - | Sec. 4.3
  - GPU-only, DeepSeek-MoE-16B | 4.38-5.93x vs Mixtral-Offloading; 2.95-3.78x vs HOBBIT | - | - | Sec. 4.3
  - GPU-NDP, Mixtral-8x7B | 54.96 (3-bit) / 77.33 (2-bit) tok/s | MoNDE 11.56 tok/s | 4.75x / 6.69x (simulated) | Sec. 4.3, Fig. 7
  - Overall claim | 3x-8x end-to-end | - | mixes real H100 PCIe offloading and simulated NDP | Sec. 4.3
  - Average accuracy (7 commonsense tasks + MMLU), Mixtral-8x7B | Original 70.03; GPTQ INT3 67.67; HQQ INT3 68.51; Ours INT3 68.96; GPTQ INT2 34.41; HQQ INT2 46.27; Ours INT2 54.51 | Original 70.03 | Ours INT2 is -15.5 points; +8.24 over HQQ INT2; INT3 +0.45 | Fig. 6, Sec. 4.2
  - MMLU, Mixtral-8x7B | Original 70.38; HQQ INT2 37.03; Ours INT2 47.53; Ours INT3 69.19 | 70.38 | Ours INT2 is -22.9 points; +10.50 over HQQ INT2 | Fig. 6, Fig. 8a
  - Average accuracy, DeepSeek-MoE-16B | Original 59.63; HQQ INT2 48.56; Ours INT2 50.86; GPTQ INT2 38.50; Ours INT3 58.81 | 59.63 | Ours INT2 -8.8 points; +2.30 over HQQ INT2 | Fig. 6
  - MMLU, DeepSeek-MoE-16B | Original 45.03; HQQ INT2 27.02; Ours INT2 28.75 (top-3 restored); top-1 only 25.80 | 45.03 | Ours INT2 -16.3 points, close to the 25% chance level | Fig. 6, Fig. 8a, Table 2
  - Mixtral-8x7B restoring Top-1 vs Top-2 only (INT2) | MMLU 47.53 vs 25.26; WikiText ppl 6.32 vs 16.76 | - | - | Table 2
  - Compensator transfer cost | 0.32 MB (rank 16, 0.75% of an INT2 expert) to 2.53 MB (rank 128, 6.03%) | - | - | Sec. 4.4 (2)
- Catalog-row one-line summary, itemised:
  - 'Low-rank compensation to keep the savings of 2-3 bit offloaded experts': YES. this is the method; but 'keeps savings' holds only with a compensator transfer of 0.75-6% of an expert and only for Top-n experts per token
- Transfers to our regime (CPU-only, 8-32 GB, NVMe streaming, batch 1, 125B 512-expert top-10): NO. Speed: H100 + DDR over PCIe (2.37 tok/s baseline) plus simulated NDP; no CPU-only, no NVMe. As evidence about quality: plain 2-bit loses a great deal (GPTQ INT2 collapses, HQQ INT2 MMLU 37), the compensation recovers only part (Mixtral MMLU 47.5 vs 70.4), and on the fine-grained 64-expert DeepSeek-MoE model 2-bit stays near chance on MMLU (28.75 vs 45.03). That model type is the closer analogue of a 512-expert top-10 model. It also requires an offline compensator per expert (extra storage) and a router-guided runtime restore; DeepSeek needs a larger rank/more experts.
- Quality loss: See numbers: INT2 + compensation, Mixtral-8x7B average -15.5 points, MMLU -22.9; DeepSeek-MoE-16B average -8.8, MMLU -16.3. INT3 + compensation is close to lossless on average (-1.1 Mixtral, -0.8 DeepSeek). Perplexity ablation WikiText 6.32 (rank 32) to 3.77 (rank 128) on Mixtral.
- Ablations: Yes (Sec. 4.4): number/position of restored experts (Top-1 vs Top-2; Top-3 for DeepSeek), rank budget 16-128 vs overhead, kurtosis-guided vs uniform rank (7.05 vs 7.55 ppl at rank 16; 3.77 vs 8.54 at rank 128).
- Limitations stated by the authors: Not stated as a section; the text concedes DeepSeek-MoE gains are small and may need larger rank or more experts; compensation settings are hand-configured per model family; NDP results are simulated.
- Contradicts our own findings (pubtech/A.md)?: No contradiction; it supports our flag that 2-bit quality for the 125B-class 512-expert model is unmeasured and likely severe (our OLMoE Q2_K perplexity +26% fits this direction).
- Corrections to the first pass:
  - All first-pass speed ratios verified (5.17x/7.64x, 3.77-5.48x vs HOBBIT, 5.18x/7.63x for 8x22B, 2.95-5.93x DeepSeek).
  - First pass accuracy: 'uniform GPTQ INT2 collapses 70.03 -> 34.41' verified; added absolute INT2+compensation values: Mixtral MMLU 47.53 (-22.9 vs 70.38), DeepSeek MMLU 28.75 (near chance, -16.3 vs 45.03).
  - First pass 'vs HQQ +0.45% (INT3) and +8.24% (INT2) average, MMLU +1.11 / +10.50' verified; DeepSeek gains +1.63 / +2.30 verified.


## Family s1 #6 (stickiness / cache-aware routing): all further prior art found, and what is new in our variant

Our variant (from pubtech/A.md): simulate a temporal-stickiness term by swapping a fraction f of each token's non-repeating picks for the previous token's experts, evaluate on LRU hit rate of three traces of a 512-expert top-10 model at a cache of about 2 footprints (20 experts per layer, about 4%), turn the hit rate into serial/overlapped tok/s. Reproduced hit gain: f=0.5 gives 0.763 vs 0.549.

Papers read in full or in the relevant parts (raw text, PDF):
- Skliar et al., Cache-Prior routing (arXiv 2412.00099, P-0044): the training-free, logit-bias version. Read in full above.
- Prefetch-Steered Routing, PSR (Siming Huang, HKUST-GZ, arXiv 2610.34721v1, 26 Oct 2026; NEW, not in catalog): deterministic logical cache + at most n next-layer-router-predicted experts per layer and token get the routing bonus; OLMoE-1B-7B on RTX 4090 with 50%/25% of experts resident: 10.7 ms/token at 50% = 2.3x faster than exact routing + LRU (24.7 ms), 25-30% faster than Cache-Prior (lambda=0.2); mean benchmark difference +0.1 / +0.5 points (95% lower bounds -0.8 / -0.3). On Qwen3.6-35B-A3B (256 experts, copies cheap) its latency advantage over Cache-Prior disappears. Authors also report that cache-aware routing combined with asynchronous prefetch makes outputs depend on transfer timing. Also says: prefetch by token history or popularity gives exact routing little and clearly hurts under cache-aware routing. Hardware GPU; OLMoE has 64 experts; batch-1 teacher-forced. Closest threat: it already combines cache-bias routing and prefetch at low residency (25%), not at 4%.
- Sticky Routing / StickyMoE (Ali Kayyam, BrainChip, arXiv 2607.08780, 12 Jun 2026; NEW): a differentiable consistency loss at pre-training time; claims switch rate -59% and cache misses reduced up to 3.92x on small/medium models. Read abstract and introduction only (training-time, small models), so UNREAD beyond that.
- ReMoE (arXiv 2605.27081, P-0043, other row): router fine-tuning for reuse; cited for completeness (abstract level).
- SpecMD (Hoang, Jaiswal, Samragh, Cho; Apple; arXiv 2602.03921; NEW): benchmarking framework; finds expert access 'not consistent with temporal locality assumptions (LRU, LFU)'; proposes Least-Stale eviction (collision misses up to 85x lower than LRU at 1% capacity, over 88% hit rate at 5% capacity (abstract), TTFT -10.7 to -34.7% on OLMoE); collision misses are experts evicted and needed again within the same forward pass, so the gain is a prefill/TTFT effect (read sections on Least-Stale and results). It says routing bonus accounts for 10-20% of the speedup in some configurations (as quoted by PSR; not independently verified by me).
- SMoE (Zhu et al., arXiv 2508.18983; NEW): substitutes low-importance activated experts by functionally similar cached experts, 48% lower decode latency, hit rate above 60% (abstract read). BuddyMoE (arXiv 2511.10054; NEW): substitute a cached 'buddy' expert on prefetch miss, up to 10% throughput (abstract read). MoE-CORE (arXiv 2610.01950, Huang/Huawei; NEW): coordinated offloading and residency with optional substitution (abstract only). Speculating Experts (arXiv 2603.19289) and SP-MoE (arXiv 2510.10302), In-depth analysis of caching and prefetching (arXiv 2511.05814, CMU): only titles/abstracts read.
- Oracle-MoE (Zhou et al., 2025, routing in an attention-derived space; from memory via StickyMoE's citation, not read), MoE-ERAS (Bambhaniya et al., 2024, cited by PSR as also preferring resident experts, not read), MoE-Infinity (arXiv 2401.14361), AdapMoE (arXiv 2408.10284): known, not read at depth.
- WiSP (Zhang et al., Nokia, arXiv 2606.21868v2; NEW): working-set paging on RTX 3090, byte-identical outputs, up to 2.0x over static offload; finds prediction/prefetch does not help in single-stream decode because the bottleneck is PCIe bandwidth (abstract + intro read). Budgeting Bytes (Zhang, arXiv 2609.04238, 29 Jul 2026; NEW): on an 8 GB RK3588 with eMMC and Qwen3-30B-A3B (18 GB 4-bit) decode is pinned at 0.19-0.51 tok/s, even a trace-driven oracle prefetch does not help (0.12 -> 0.13), quantising to fit a 16 GB device gives 11.5 tok/s (22x); frozen-model probe predicts routing at 91.2% (abstract read in full, body skimmed).
- vLLM issue 38256 comments (P-0046): independent Apple-silicon data (staging 0.94-0.95x, LFU 1.4-2.3x worse than LRU) and a 131B/6B-active top-8 model on 96 GB GPU.

What is new in OUR variant: the method (bias or swap toward reusing experts, lossy, training-free or trained) is known: Cache-Prior (Nov 2024) is the inference-time version, PSR (Oct 2026) adds prefetch steering, ReMoE/StickyMoE are training-time versions, SMoE/BuddyMoE are substitution versions. Nothing new in the mechanism. What differs is the regime and the measurement: a 512-expert top-10 model, resident cache of about 4% (2 footprints), CPU-only with experts on NVMe, serial vs overlapped cost model. No paper tested below 25% residency (Cache-Prior App. D) and PSR's gain collapsed on a 256-expert model where copies are cheap; so 'does it still work at 4% and 10 picks' is open, as is quality (a trained router for a 125B model is out of reach; no logits in our traces). Our logit-free swap is not the published method (it needs router logits and a protected top-J), so the test row 'no router logits needed' tests an upper-bound proxy, not Cache-Prior.

Closest papers: Cache-Prior (2412.00099) and PSR (2610.34721), both read.


## Family s1 #7 (mixed 2/3/4-bit experts, byte-capacity cache): all further prior art found, and what is new in our variant

Our variant (pubtech/A.md): requantise all experts (69 GB at 4.3 bpw down to 3.0 / 2.6 / 2.06 bpw), then replay the three traces through a byte-capacity LRU at equal RAM; result 8 GB 1.35-1.72x serial / 1.08x overlapped, about nothing at 16-32 GB; the size 'nearly fits' claim refuted (33 GB at 2.06 bpw vs 27 GB expert cache).

Papers read:
- EdgeMoE (Yi et al., arXiv 2308.14352, IEEE TMC; Aug 2023; NEW to the catalog): the earliest in this family: non-expert weights in RAM, experts on external storage (SSD, HDD, SD card), expert-wise bitwidth adaptation chosen against a user accuracy-loss budget, plus expert preloading; Jetson TX2 and Raspberry Pi 4B; Switch-Transformer-based models up to 26B (128 experts, top-1) and GPTSAN; 1.19x-2.77x over memory-optimised baselines (dynamic expert loading, STI), speed-up 2.64-3.03x over IO-EXP on Jetson TX2 SSD, 4.49-4.76x on HDD; accuracy loss at most 2% (abstract and results read in part). This predates Eliseev and Mazur and already combines low-bit experts with storage-backed offloading.
- Eliseev and Mazur (P-0049), HOBBIT (P-0050), MoE-SpeQ/Specter (P-0051), EVICT (P-0052, off-topic), low-rank compensation (P-0053): read above.
- DyMoE (Huang et al., arXiv 2603.19172; NEW): importance-aware dynamic mixed-precision experts, depth-adaptive, lookahead prefetch; RTX 3090 / Jetson Orin NX / RTX 4070 class; TTFT -3.44x to -22.7x and TPOT up to 14.58x vs offloading baselines (abstract read; baselines weak).
- SliceMoE (Choi et al., KAIST, arXiv 2512.12990; NEW): bit-sliced expert caching (Matryoshka-style nested slices so low-bit and high-bit versions share storage, no duplication), decode latency up to 1.81x/1.64x and energy up to 2.37x/2.85x on DeepSeek-V2-Lite / Qwen1.5-MoE (abstract read). This is the closest to a byte-capacity cache where precision is chosen on demand.
- Dynamic Expert Quantization / DynaExq (Chu et al., arXiv 2511.15015; NEW): online budget-constrained precision allocation keeping hot experts high precision with a low-precision fallback on a single GPU (abstract read).
- RotaryQuant (arXiv 2608.08081, 'Fitting 120B MoE models on consumer hardware', Aug 2026): title only, not read.
- Budgeting Bytes (arXiv 2609.04238): on a bandwidth-walled edge device the lever that works is reducing bytes per token to fit the fast tier (22x), not prefetch; this directly supports the byte-reduction direction and our observation that overlap helps little once compute-bound.
- Edge0 (arXiv 2609.18063, other rows P-0100/P-0114): 35B MoE at about 20 tok/s from SSD with an int4 student, trained prerouter and recovery LoRA; relevant because it quantises to int4 and recovers quality by LoRA (abstract and executor-path sections read).

What is new in OUR variant: also not the method. Per-expert or mixed bit-widths for storage-backed expert offloading date to EdgeMoE (2023) and Eliseev and Mazur (2023); miss-only low precision is HOBBIT (2024); slice-level precision caching is SliceMoE (2025); compensation is P-0053 (2025). What none of them report is the equal-RAM, byte-capacity comparison (same RAM, 4.3 vs 2.6 bpw) at 512 experts / top-10 on NVMe with serial and overlapped cost models; that simulation result (small gain only at 8 GB, none at 16-32 GB) is our own, but it is a measurement, not a method. Quality at 2-3 bpw for a 512-expert model is untested anywhere: P-0053 shows 2-bit remains near chance on the 64-expert DeepSeek-MoE-16B MMLU (28.75 vs 45.03) even with compensation, and E&M lose about 4-5 MMLU points even on Mixtral with a data-free 2-bit scheme.

Closest papers: EdgeMoE (2308.14352, partly read), Eliseev and Mazur (read in full), HOBBIT (read in full), SliceMoE (abstract only).


## Summary of corrections and consequences

Raw text read: 9 of 9 rows (8 arXiv PDFs via pdftotext, 1 GitHub issue via API with all 24 comments; MoE-SpeQ read in both v1 and v3). Unread: 0. Added prior art beyond catalog: PSR (2610.34721), StickyMoE (2607.08780), SpecMD (2602.03921), SMoE (2508.18983), BuddyMoE (2511.10054), MoE-CORE (2610.01950), WiSP (2606.21868), Budgeting Bytes (2609.04238), EdgeMoE (2308.14352), DyMoE (2603.19172), SliceMoE (2512.12990), DynaExq (2511.15015). Of these only PSR, EdgeMoE and parts of SpecMD / Budgeting Bytes were read beyond the abstract/introduction.

Corrections to the first pass (itemised per row above): P-0045 (venue visible in v4, 54B/57B is not an inconsistency, no hardware, knee only for group-1 models), P-0046 ('12K to 32 loads' is in the issue; 'Edge0 whole-layer' belongs to another source), P-0047 (91.96% and 61.15% are different operating points; 61.15 not reproducible), P-0051 (3.3x and the expert-union remark are in v1), P-0052 (1.23-1.27x is high-acceptance benchmarks only, average 1.15x; the 74.7%/32.5% numbers are EVICT's own).

What changes what we should believe about our own measurements:
1. Our stickiness test is a proxy, not Cache-Prior: the published method biases logits (needs router logits and a protected top-J); ours swaps picks at random. The miss-rate reduction we got (ratio 0.53 at f=0.5) is in the range of the paper's (0.25-0.53 at lambda=0.5), but no paper tests below 25% residency, and PSR's gain over Cache-Prior vanished on a 256-expert model when copies are cheap.
2. One anecdote (vLLM issue, Apple M4, Qwen3.6-35B-A3B) shows miss rate nearly flat (62.5% to 64.7%) from 38 to 72 resident experts per layer, whereas our LRU hit curve rises steeply (0.454 / 0.549 / 0.686). Different model and tuner, uncontrolled; but it is the one data point in the literature that contradicts the steepness we rely on for the 2-footprint figure. It is worth checking our hit curve against a second model/trace.
3. The routing-consistency paper (P-0045) finds LRU at 0.67 of the clairvoyant optimum at 2x cache on its TOY model; our LRU hit 0.549 at 2 footprints then implies an optimum near 0.82 for our model, inside the 0.46-0.97 range of its oracle values for real models, so our LRU numbers are plausible (derived, not measured).
4. Our overlap-vs-serial split is supported by P-0044 (throughput near-linear in miss rate on a device with no I/O overlap), by WiSP/Budgeting Bytes (prefetch gives nothing when the bus is saturated) and by the Apple-silicon comment (staging 0.94-0.95x).
5. Quality for 2-3 bpw at 512 experts remains unmeasured by anyone; the evidence that exists (P-0049 -4 to -5 MMLU, P-0053 -16 to -23 MMLU at 2-bit even with compensation, and near-chance MMLU on the fine-grained DeepSeek-MoE-16B) says our 8 GB serial gain of 1.35-1.72x would come with a large quality cost unless a recovery mechanism (P-0053 compensators, Edge0 LoRA) is added.
6. Baseline weakness: HOBBIT's 9.93x/13x and P-0053's 7.64x are against baselines at 0.07-2.4 tok/s; they do not bound what a tuned CPU/NVMe path achieves. The same applies to MoE-SpeQ (against Mixtral-Offloading and Transformers) and ExpertFlow (against LRU per-layer cache at batch 4-32).
