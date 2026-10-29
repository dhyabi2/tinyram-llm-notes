# deep_P12b: raw-text re-check of P-0267..P-0272, P-0274..P-0276 (families 24 and 25)

Raw text read for all 9 rows (pdftotext of arXiv PDFs; curl-and-strip for blog/README pages). Scratch: scratchpad/deep_P12b. Caveat on figure-only values: pdftotext gives data labels of bar charts but not always their mapping; where a mapping is inferred it is flagged.

Our regime for "transfers": CPU-only, 8/16/32 GB RAM, expert weights from NVMe, batch 1 decode, 125B-class 48x512 top-10 MoE. Our simulations: pubtech/F.md (families 24, 25), D.md (P-0184 DynaExq/GEMQ line).

## P-0267 LMSYS blog: Fast JSON Decoding for Local LLMs with Compressed FSM
- Read: https://lmsys.org/blog/2024-02-05-compressed-fsm/ (Liangsheng Yin, Ying Sheng, Lianmin Zheng, Feb 5 2024). Blog, no arXiv id.
- Setup: llama-7B on one NVIDIA A10 (24 GB); vllm v0.2.7, guidance v0.1.0, outlines v0.2.5, llama.cpp v0.2.38 (python binding). Two tasks: character data in JSON from a brief prompt; city info extracted from a long document into JSON. Throughput at each system's max batch, latency at batch 1.
- Numbers (Fig 6 is an image; only text numbers readable): latency "up to 2x" lower, throughput "up to 2.5x" higher vs guidance+llama.cpp and outlines+vLLM (best baseline not stated in text); retokenization overhead "approximately 4%". Ratios cannot be recomputed: no table.
- Catalog/first-pass numbers: 2x latency, 2.5x throughput, 4% overhead, A10 24GB, Llama-7B: all correct. Catalog sentence (forced paths prefilled in one step, not skipped; savings from fewer decode steps): correct. First pass baseline "Guidance+llama.cpp, Outlines+vLLM": correct.
- Conditions: GPU, dense 7B, regex/JSON with long fixed keys; it also says SGLang without jump-forward already beats Outlines+vLLM in throughput on the character task and suspects outlines overhead, i.e. part of the 2.5x is baseline slowness.
- Transfers: mechanism yes (forced runs prefilled as one multi-token forward), numbers no (GPU, dense, weights resident; on our streamed MoE a multi-token forward touches the union of experts of its tokens, which is the cost our F.md models).
- Quality loss: none claimed as lossless, but the blog says different tokenization of the jumped text changes the next-token logits; fix = retokenize the whole text (4% cost). No benchmark.
- Ablation: none in blog (the paper below has 1.6x).
- Limitations stated: tokenization-boundary issues; prefer one comprehensive regex.
- Contradicts our findings: no. Supports that f (forced fraction) is what matters; blog gives no f.

## P-0268 SGLang: Efficient Execution of Structured Language Model Programs
- arXiv 2312.07104v2 (cs.AI, 6 Jun 2024). Authors: Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, Ying Sheng.
- Setup (Sec 6.1, App C): Llama-2 dense, Mixtral MoE, LLaVA, fp16; mostly AWS G5 A10G 24GB; 7B on one A10G, Mixtral-8x7B on 8 A10G tensor-parallel, Llama-70B on 4 A100 80GB. Baselines Guidance v0.1.8 (llama.cpp backend), vLLM v0.2.5, LMQL v0.7.3. Throughput = max batch; latency = one program at a time.
- Numbers: (a) "up to 6.4x higher throughput, 3.7x lower latency" overall (abstract/Sec 6.2; the max is not JSON). (b) Sec 6.3 / Sec 6.4 ablation: compressed FSM "increases the throughput by 1.6x" on the JSON decoding benchmark; without reusing the preprocessed FSM across a batch the throughput is 2.4x lower. Baseline for 1.6x = SGLang without compressed FSM (an ablation, same system). Per-benchmark JSON bars in Fig 5-7 are charts only.
- Catalog/first-pass: "JSON about 1.6x" correct; "6.4x overall not JSON" correct; hardware list correct; catalog "published jump-forward for JSON; forced tokens prefilled" correct.
- Conditions: GPU, batch throughput mode, regex-derived JSON. 
- Transfers: mechanism yes; 1.6x is the only isolated JSON number and is a batched GPU number.
- Quality: none measured. App B.3 states probability skew: forcing a compressed edge can distort choice probabilities (e.g. "Above Average" vs A+); listed as future work. App B.2: retokenize forced text, "minor" overhead.
- Ablation isolating our part: yes, 1.6x with/without compressed FSM (Sec 6.4).
- Limitations: distorted probability (B.3), FSM preprocessing cost.
- Contradicts ours: no. Our F.md serial speedup 1.1x-1.5x (conservative) fits a 1.6x GPU ablation.

## P-0269 XGrammar (MLSys 2025)
- arXiv 2411.15100v3 (12 May 2025). Yixin Dong, Charlie F. Ruan, Yaxing Cai, Ruihang Lai, Ziyi Xu, Yilong Zhao, Tianqi Chen.
- Setup for jump-forward (Appendix B, Fig 11): Llama-3.1-8B-Instruct in SGLang, AMD Ryzen 9 7950X + RTX 4090, batch 1, metric time per output token, JSON-schema task, Outlines vs XGrammar each with and without jump-forward. Main results (Sec 4): H100 + EPYC 7R13, input 139 / output 53 tokens, vLLM+Outlines, llama.cpp, SGLang+XGrammar, MLC.
- Numbers: Fig 11 labels 44.2, 31.5, 6.8, 5.4 ms; Table 1 (SGLang+Outlines vs SGLang+XGrammar, JSON Schema, Llama-3.1 8B) shows 44.2 and 6.8, and DeepSeek-V2-Lite 16B MoE 15.8 vs 4.8 (ms). So mapping (inferred, consistent with Table 1): Outlines 44.2 -> 31.5 with jump-forward (1.40x), XGrammar 6.8 -> 5.4 (1.26x). XGrammar vs Outlines: 6.5x without, 5.8x with jump-forward. Table 1 MoE ratio 15.8/4.8 = 3.3x. Abstract: up to 100x (mask-engine microbenchmark), 80x end-to-end serving.
- Catalog/first-pass: catalog "jump-forward combined with masking gives best TPOT; forced tokens appended without sampling": correct (Appendix B). First pass said "no numeric result readable; values in Figure 11 only": WRONG, the values are readable (above). First-pass "abstract up to 100x": correct. First-pass summary_accuracy "partly": agree only in that jump-forward is an appendix sidenote, not XGrammar's contribution ("orthogonal technique").
- Conditions: GPU TPOT dominated by grammar mask cost (Outlines 44 ms); jump-forward gain is on top.
- Transfers: mechanism yes. The 1.26x is a measured single-stream jump-forward gain on a JSON-schema task on a GPU where forward is ~5 ms, not a streaming regime.
- Quality: none for jump-forward; XGrammar masks are exact.
- Ablation: yes, Fig 11 with/without jump-forward (1.26x / 1.40x).
- Limitations: nothing stated about jump-forward; rollback support noted (Sec 3.5).
- Contradicts ours: no. Observed 1.26x with GPU weights resident is below our 1.1x-2.3x on streamed weights only because our model differs.

## P-0270 llguidance fast-forward tokens
- Read: docs/fast_forward.md and README.md of guidance-ai/llguidance (raw GitHub, main), the gitlink mirror page (dnrops/llguidance, a Chinese mirror, only commit list visible), llama.cpp PR 10224 (API JSON) and llama.cpp docs/llguidance.md. Author of PR: mmoskal.
- Setup: Rust library; no model or hardware benchmark for fast-forward.
- Numbers: mask ~50 us single-core CPU per token, 128k tokenizer, JSON-schema grammars (README); "typical JSON schema about 1.5 ms" (README; the two are for different statistics); p99 0.5 ms, p100 20 ms (llama.cpp doc). Forced tokens share on MaskBench: 12.7% (12.1% with the conservative variant). Non-canonical forcing: with token healing off (max_tokens=0) 97% of fast-forwarded sequences are non-canonical; with max_tokens=1: 23 cases among ~10k tests; >=2: none. llama.cpp merge b4613 on 2 Feb 2025 (README; PR merged_at 2025-02-02).
- Catalog/first-pass: all first-pass numbers correct. Catalog "llama.cpp can use llguidance": true for masking only. The PR body states fast-forward "is not yet hooked up to llama.cpp in this patch" and llama.cpp's llguidance doc never mentions fast-forward: FIRST-PASS "transfers yes (runs in llama.cpp CPU)" is too strong for the forced-token part; first-pass "accurate" -> partly.
- Conditions: MaskBench generic schemas (many with free values), f=12.7% of tokens forced after canonicalization, tokenizer Llama-3 128k.
- Transfers: partly. CPU mask cost trivial for us (50 us vs hundreds of ms per token); the multi-token forward is not available in llama.cpp via this path, would need own code.
- Quality: algorithm designed to avoid distribution damage; shows naive forcing is harmful (non-canonical forced tokens). No accuracy benchmark.
- Ablation: canonicalization depth (above).
- Limitations: tokenizers work on strings; heuristic max_tokens=4 might fail on constructed cases.
- Contradicts ours: YES in magnitude, and it matters: published forced share 12.7% on generic schemas vs our synthetic f 0.44-0.61. Our synthetic outputs have fixed order, compact whitespace, short values and enum/optional tails; MaskBench schemas aren't like that, and llguidance additionally drops trailing forced tokens that could merge with the next token (our "merged" variant, f 0.44, is the closer model). Do not use f>0.4 for real planner/verifier outputs until measured; the speedup is roughly 1/(1-f)-bounded (12.7% -> at most 1.15x).

## P-0271 BentoML: Structured Decoding in vLLM, a gentle introduction
- Read: bentoml.com blog, Aaron Pham, last updated Jan 15 2025 (secondary blog; no experiments of its own).
- Setup: vLLM v0 on GPU, Outlines vs XGrammar backends; benchmark charts are images (credited Michael Goin, Red Hat).
- Numbers: "up to 5x improvement in TPOT under load" (TL;DR), no table, no batch or hardware in text. No jump-forward number.
- Catalog/first-pass: catalog first half "jump-forward needs batched prefill of several tokens, not logit processors": correct ("techniques like jump-forward decoding are currently not possible with logit-processor approach. It requires prefilling a set of k-next tokens"). Catalog second half "The saving is batching, not zero work": ABSENT from the page (an inference, not stated). First pass "wrong" overstates; "partly" is right. First-pass 5x: correct, and it is mask/engine overhead under load, not jump-forward.
- Transfers: no numbers; the one statement (needs k-token prefill) is applicable and matches what we need in llama.cpp (a batch>1 decode call).
- Quality/ablation/limitations: none. Page lists vLLM v1 plan to integrate jump-forward later.
- Contradicts ours: no.

## P-0272 Earley-Driven Dynamic Pruning (Formatron / ZapFormat)
- arXiv 2506.01151v1 (1 Jun 2025), ICML (42nd Intl. Conf. on Machine Learning, per page header). Xintong Sun*, Chi Wei*, Minghao Tian, Shiwen Ni.
- Setup: RTX 3090 24GB + EPYC 7452; gemma-2-9b-it, Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, FP16, Transformers 4.48. Baselines lm-format-enforcer 0.10.9, Outlines 0.1.13, XGrammar. Tasks geoquery, Json schema, Json grammar. Metric "throughput (tokens/s)" of the constrained-decoding pipeline (Appendix Table 6 splits parse/mask stages).
- Numbers: Table 1 (tokens/s), Formatron vs XGrammar: Gemma geoquery 12174.68 / 616.66 = 19.7x; Gemma json_s 7943 / 1474 = 5.4x; Llama3 geoquery 6700.87/2758.98 = 2.4x; Mistral json_s 10639.87 / 8421.43 = 1.26x; but json_g Gemma 8668.69 vs XGrammar 10245.04 (0.85x) and Mistral 12046.54 vs 15273.72 (0.79x). vs Outlines Gemma json_s 7943.34 / 61.32 = 130x. Abstract: "up to 2x"; conclusion "1.5-2x". Ablation Table 3: removing pruning drops throughput 30-50%. Table 4 memory: 1635.92 vs 1655.48 MB (Llama3). Table 7 accuracy comparable to baselines (e.g. Llama3 json schema 0.67 vs XGrammar 0.69).
- Catalog/first-pass: Catalog says "Reviews of constrained-decoding speedups (DOMINO, Formatron)" and "gains 1.5-2x from skipping forced forwards plus cheaper masks": WRONG. The paper is a method paper; it contains neither DOMINO nor jump-forward/forced-forward (searched); gains come from cheaper Earley masking/caching. Gain "1.5-2x" is the authors' headline but the paper's own Table 1 ranges from 0.79x to 19.7x vs XGrammar. First pass "wrong": correct; "Table 1 about 20x on one cell": correct (19.7x).
- Transfers: no (mask cost; irrelevant on CPU+NVMe where mask is microseconds against ~hundreds of ms of expert I/O).
- Quality: Table 7 above, no loss vs baselines claimed.
- Limitations: none beyond impact statement.
- Contradicts ours: F.md says "DOMINO/Formatron report 1.5 to 2x, consistent with 1.3x to 2.3x range": that consistency claim is void; those numbers measure mask speed, not skipped forwards. Real DOMINO (arXiv 2403.06988, read, see family section) reports speculative-token speedup 1.66x-1.91x on schema JSON/template on GPU, which is the number that should have been cited.

## P-0274 DynaExq
- arXiv 2511.15015v4 (page stamp "10 Sep 2026", cs.PF). Kexin Chu, Dawei Xiang, Zixu Shen, Yiwei Yang, Zecheng Lin, Wei Zhang. ACM format, venue blank ("Conference'17" template).
- Setup: single RTX A6000 48GB; HuggingFace Transformers + pytorch, ~3K lines Python; models (Table 3): Qwen3-30B-A3B 30.5B/3.3B active, 48 layers, 128 experts, top-8; Phi-3.5-MoE 42B/6.6B, 32 layers, 16 experts, top-2; Qwen3-Next-80B-A3B (printed "Qwen3-80B (Int4)") 80B/3B active, 41 GB at Int4 (37 GB experts), 48 layers, 512 experts + 1 shared, top-10. Hot tier FP16 (Int4 for 80B), cold tier Int4 (Int2 for 80B). Baselines: static Int4/Int2 PTQ and ExpertFlow (offloading/prefetch). Hotness = EMA of router counts per time interval, per-layer top-n_hi under a fixed HBM budget, async promote/demote, hysteresis.
- Numbers: Table 4 (avg of MMLU-Pro, GPQA, AIME25, GSM8K, HumanEval): 30B FP16 65.29, Int4 63.98, DynaExq 64.38; 80B Int4 78.11, Int2 73.09, DynaExq 77.57 (Int2 -> DynaExq +4.48; Int4 -> DynaExq -0.54); Phi FP16 58.06, Int4 56.31, DynaExq 57.00. Throughput: only prose "1.42x to 2.73x vs ExpertFlow, gap widens with batch"; abstract "2.73x at batch 32"; the throughput table reference is broken ("?? reports") and tok/s are figure-only: no raw tok/s or per-model ratio table. Table 1 decode expert-activation ratio: Qwen3-Next-80B 1.9% at batch 1, 3.7, 6.7, 14.9, 25.4, 39.2% at batch 32.
- Catalog/first-pass: 2.73x "at batch 32": per abstract yes; "vs offload/prefetch baselines (plural)": only ExpertFlow was compared. 1.42x-2.73x correct. 73.09 -> 77.57 correct. "48x512 top-10": correct (plus 1 shared, 3B active). "single baseline": correct. First-pass "raw tok/s in figures only": correct.
- Conditions: GPU HBM budget (not host RAM or NVMe), batch sweep up to 32, 80B Int4 fits in 48 GB, so "offloading" baseline is a stress configuration; dense-activation motivation is explicitly about batch growth.
- Transfers: partly. The idea (per-layer hot set at higher precision, cold at low precision, router-count EMA) is what F.md/D.md already simulate; the throughput claim does not transfer: at batch 1 decode activation is 1.9% of experts (their own Table 1), the regime where they say offloading works fine.
- Quality: Table 4 above; deltas are small because the hot/cold tiers are Int4/Int2 and Int4 is already near FP16.
- Ablation: pruning/hysteresis not isolated; hotness-vs-random assignment is shown only in Fig 3 (perplexity vs low-precision expert ratio); no cache-vs-precision split.
- Limitations stated: none dedicated; (paper is "largely GPU, single device").
- Contradicts ours: no. It supports hot/cold tiering; our replay 1.43-1.52x bytes (D.md) and "2.73x GPU batch 32 not comparable" (F.md) are consistent.

## P-0275 Any-Precision LLM (and SNU-ARC repo)
- arXiv 2402.10517v4 (21 Jun 2024), ICML 2024 (page header: Proceedings of the 41st International Conference on Machine Learning). Yeonhong Park, Jake Hyun, SangLyul Cho, Bonggeun Sim, Jae W. Lee (SNU). Repo SNU-ARC/any-precision-llm README read (seed precision default 3, parent default n, supports precisions [seed, parent], bitplane packing, CUDA).
- Setup: SqueezeLLM (non-uniform, clustering) + "incremental upscaling": 3-bit seed, each step splits each centroid cluster to add one bit; parent 8-bit; weights stored as bit-planes; per-bit-width centroid tables. Models Llama-2-7B, Mistral-7B, OPT-6.7B/2.7B/1.3B. Hardware: RTX 4090, RTX 4070 Laptop, Jetson AGX Orin 64GB (kernel tests); end-to-end in TensorRT-LLM on RTX 4090/4070 Laptop (Fig 12); quantization run on i9-13900K.
- Numbers: Table 1 (Llama-2-7B) memory for supported {3,4,5,6,7,8}: 8.4 GB any-precision vs 29.9 GB separate = 3.56x (for multiple models); {3,6}: 5.6 vs 8.3 GB. Table 3 wikitext2 ppl Llama-2-7B SqLLM vs SqLLM+IU: 4-bit 5.61 vs 5.62, 5-bit 5.50/5.50, 8-bit 5.47/5.47; worst delta OPT-6.7B PTB 4-bit +0.23, OPT-1.3B PTB +0.14; zero-shot within 0.2%. Table 2 (Llama-2-7B wiki): GPTQ+IU 4-bit 12.7 vs GPTQ 5.68; AWQ+IU 4-bit INF: upscaling works only with non-uniform clustering. Table 6 kernel speed over cuBLAS FP16 matvec (4096x11008-ish): 3-bit x4.67 (RTX4090), 4-bit x3.43, 8-bit x1.78; Orin 3-bit x4.20. Table 5: seed+upscale takes under a minute per 7B.
- Catalog/first-pass: Catalog "one stored n-bit model, lower precision from most significant bits; seed model upscaled": correct in spirit, but note lower-bit models use their OWN centroid tables (not literally the top bits of a rounded parent). First pass "no quantitative result on either page": WRONG, the paper has Tables 1-6 and Fig 12 (numbers above). First-pass "seed 3 bits not 2": correct (README default 3; 2-bit only as a uniform-quant appendix experiment, Fig 14, which fails).
- Conditions: GPU CUDA kernel, bit-plane layout makes memory reads proportional to bit-width (8-bit packing does not).
- Transfers: partly. Layout yes: reading k planes reads k/n of the bytes, no storage duplication; kernel no (CUDA, no CPU kernel). Seed 3 bits, so a 2-bit base is out of range of what was shown.
- Quality loss: negligible vs independently quantized models at 4-8 bit; 3-bit is the seed itself (SqueezeLLM 3-bit wiki 6.13 vs 5.47 at 8). Dense models only.
- Ablation: kernel optimizations (Fig 10), IU vs independent (Table 3). No MoE.
- Limitations stated: needs non-uniform quant backbone; engine GPU. 
- Contradicts ours: no. F.md says "layout with no byte saving by itself": consistent (savings are for storing several models; for one precision the byte read is what the chosen plane count costs).

## P-0276 AnyBCQ (ICLR 2026)
- arXiv 2510.10467v2 (2 Feb 2026), ICLR 2026. Gunho Park, Jeongin Bae, Beomseok Kwon, Byeongwook Kim, Se Jung Kwon, Dongsoo Lee (NAVER Cloud).
- Setup: Llama-3.1-8B, Gemma-2-9B, Phi-4-14B; kernel latency on A100 80GB (CUDA 12.6), H100 in appendix; group 128, 512 C4 calibration seqs; baselines AWQ, Any-Precision LLM, ShiftAddLLM, FP16; decode throughput batch not stated in main text.
- Numbers: Table 4 Llama-3.1-8B A100 tok/s at 2/3/4 bits: AnyBCQ 245/212/186, Any-Precision 228/196/169, FP16 105 -> 2-bit 2.33x vs FP16, 1.07x vs AP. Phi-4-14B 2-bit 171 vs FP16 56 = 3.05x (the "up to 3.0x"); Gemma 185 vs 83. Quality 2-bit Llama: wiki ppl 19.01 (AP 1680.77, FP16 6.24), MMLU 35.32 (AP 24.66, FP16 65.35 in Table 4; Table 2 FP16 65.02), Table 2 CSR avg 2-bit 58.71 (multi-prec) vs 58.89 fixed-prec vs FP16 72.72, AP 39.65. 3-bit CSR 69.30 (multi-prec). Table 1 memory (Llama-3.1-8B): multi-model 9.85 GB vs proposed 4.99 GB (-49%), BCQ2 2.19 GB. Table 3 GEMV: 2-bit x4.07 vs cuBLAS FP16 (28672x8192), x1.33 on 4096x4096; AP 2-bit x3.66 on the same. H100: 2-bit 325 tok/s vs FP16 168.
- Catalog/first-pass: all first-pass numbers correct (245/228/105, 3.0x, 58.71 vs 72.72). Catalog "true bit-plane computation": correct (kernel accumulates per-plane partial sums). Note abstract says "1.2x over SOTA multi-precision": Table 4 gives 1.07x-1.17x.
- Conditions: GPU, dense, 2-4 bit; small-matrix gains vanish (x1.33).
- Transfers: no for kernel (GPU dense); BCQ layout (shared binary planes + per-precision scales) is an alternative to Any-Precision's centroid layout and costs 0.24 GB scale overhead at 2-bit.
- Quality: 2-bit loses ~14 CSR points / 30 MMLU points (above): a 2-bit base is not free.
- Ablation: fixed vs multi-precision (Table 2/6), kernel breakdown (Table 7).
- Limitations stated: BCQ capacity limits peak accuracy at higher bit widths; empirical, no theory.
- Contradicts ours: supports our unmeasured-quality caveat in F.md family 25 (2-bit base needs quality measurement); the dense-model 2-bit gap is large.

# Family 24: grammar-constrained jump-forward decoding
Prior art found and read:
- LMSYS compressed FSM / SGLang (above): first jump-forward in a serving runtime (regex FSM, retokenize).
- Guidance interleaved chunked prefill (named in LMSYS and llguidance docs; the commit is linked in fast_forward.md, not opened).
- llguidance fast-forward with canonical-token safeguard (above): closest on tokenization-safety.
- DOMINO (arXiv 2403.06988, ICML 2024, Beurer-Kellner, Fischer, Vechev): read Tables 3 and Fig 5. Speculative decoding on grammar-predicted tokens: schema JSON 1.77x (Mistral 7B) and 1.66x (Llama 13B) throughput vs unconstrained generation, XML schema 1.52x/1.84x, fixed template 1.30x/1.91x, free-form JSON 0.96x/1.12x, C 0.78x/0.85x; transformers backend on A100/H100; llama.cpp grammars 0.74x-0.87x. Mechanism differs: speculate with checks, not forced prefill.
- XGrammar, Formatron (above), SynCode (2403.01632), Flexible and Efficient GCD (2502.05111): mask speed, not forced-token batching (listed from search; the latter two not read in full).
What is new in OUR variant (if anything): none of these treat the cost of a k-token forward in an expert-streaming MoE where the tokens of a forced run must route through experts and the union of their experts must be read from disk; the published work assumes weight-resident GPUs where a k-token forward costs about one token. That cost model (F.md) is ours; the forced-run idea itself is not new. Closest papers read in full: SGLang (2312.07104), DOMINO (2403.06988), llguidance doc. A further caveat the papers raise that our simulation lacks: non-canonical forced tokens degrade the model (llguidance: 97% non-canonical without healing), so f must be counted after canonicalization (published 12.7%).

# Family 25: low-bit base plus higher-precision refinement / hot-cold precision
Prior art found:
- HOBBIT (arXiv 2411.01433, Tang, Liu, Hou, Pu, Wang, Heng, Li, Guo; read): mixed-precision expert offloading in llama.cpp. Mixtral-8x7B (45B/14B, top-2, 32 layers x 8 experts) and Phi-MoE (42B/6.6B); Jetson AGX Orin 32GB (SSD ~3,000 MB/s) and RTX 4090 + 256GB host RAM; batch 1. On a cache miss, an unimportant expert (low gate score vs thresholds T1/T2) is loaded at int4 (from float16) or int2 (from int8) instead of full precision; plus layer-ahead prefetch and a cache policy (LHU) that tracks precision. Results: decoding speedup "avg 13.0x (Mixtral) and 18.9x (Phi-MoE) vs llama.cpp on Jetson", "3.64x / 9.93x vs MoE-Infinity" (the 9.93x headline is vs MoE-Infinity on Phi-MoE). Quality Table 3: GSM8K Mixtral fp16 0.52 vs fp16+int4 0.51; Phi-MoE 0.83 vs 0.82; int8+int2 0.51 / 0.83; TruthfulQA about equal; text says accuracy decreases by no more than 1%. This is the closest to our family (SSD, batch 1, llama.cpp, precision chosen on misses by importance), and it is on-demand lower-precision loading of missing experts, the reverse direction of "2-bit base plus 4-bit tiles read only when above threshold" but the same family of trigger. Its Mixtral/Phi-MoE are small expert counts (8/16), not 512.
- EdgeMoE (cited in HOBBIT; static per-expert bit widths, not opened).
- Matryoshka/D2MoE/BitStack/MC-MoE (named in our F.md/D.md; not opened this pass: UNREAD).
- Any-Precision LLM, AnyBCQ (above), DynaExq (above).
What is new in OUR variant (if anything): a per-slice residual-norm trigger for reading the refinement tile and the NVMe/8-32 GB 512-expert byte accounting. HOBBIT's gate-score trigger already covers "choose precision per expert at miss time" and reports <=1% accuracy loss on 8/16-expert models; our F.md found residual norm uniform across slices (CV 0.01), i.e. nothing to select on, and 1.0x at 16/32 GB. So the novelty is limited to the 512-expert, 8 GB measurement; closest papers read in full: HOBBIT, DynaExq, Any-Precision LLM.
