# deep_P11b: raw-text re-check of rows P-0244, P-0245, P-0246, P-0247, P-0249, P-0250, P-0251, P-0252, P-0253

Date 2026-10-29. All papers read from pdftotext / raw text (scratch folder deep_P11b). Counts: 9 rows, 9 raw text read (P-0249 via PR #4484 JSON + README + llama.cpp docs because the catalog URL is a truncated mirror behind a sign-in wall; P-0252 via the primary papers because the catalog URL is an aggregator page; AcceptMoE unfound). Statements from memory are labelled. Our measurements are from Playbook/pubtech/F.md (families 20 and 21). Page/figure locators are the printed page numbers as best recoverable from pdftotext and may be off by one.

## P-0244

**Title/venue:** Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation. Jingyu Liu, Beidi Chen, Ce Zhang (U Chicago, CMU). Proceedings of the 42nd ICML, Vancouver 2025, PMLR 267 (stated in page footer).

**Read:** arXiv 2502.02789v2 (19 May 2025), pdftotext of full PDF (177 kB text) (raw_text_read=True)

**Setup:** Speculator Llama-3.1-8B-Instruct (BF16) + base Llama-3.1-70B-Instruct (BF16) or Llama-3.1-405B-Instruct-FP8, all dense; vLLM 0.6.3.post1, tensor parallel 8 for both models; 8x H100 (quality runs) or 8x H200 node (efficiency runs; Table 4: 428.2 TFLOPS total, 4052.8 GB/s HBM per GPU, 1123 GB host RAM). Keep rate fixed per task (10/30/50/70/90%), chunk-wise selection from the speculator's attention scores, optional look-ahead steps. Baselines: base model, sentence-RAG (RAG-Llama), LLMLingua, MInference. Benchmarks LongBench, RULER, standard short tasks (MMLU, IFEval, GSM8K, HumanEval, MBPP, ARC, GPQA). TTFT measured with vLLM latency script, max decode step 1, batch x sequence products from 128x4K to 16x32K.

**Numbers (what / value / baseline / ratio / where):**
- Max TTFT speedup, 405B-FP8, 10% kept (Fig 3 bar, the 766% cell): 7.66x | baseline: vanilla 405B prefill, same batch/seq | ratio: 7.66x (paper's analytic bound 7.72x, Appendix E / Fig 9, batch 16 x 32K) | Fig 3 p8, Appendix E
- TTFT speedup 405B at 30% / 50% / 70% kept (largest cell per row): 3.21x / 2.04x / 1.42x | baseline: vanilla | ratio: 3.21 / 2.04 / 1.42 | Fig 3 p8
- TTFT speedup 70B at 10% kept (range over 4 batch x seq settings): 3.44x-3.67x | baseline: vanilla | ratio: 3.44-3.67 | Fig 3 p8
- Max end-to-end QPS on 405B, Multi-Doc QA, accuracy kept: ~7x QPS at >95% accuracy | baseline: vanilla 405B | ratio: ~7x | Sec 4.7.1 p8, abstract
- LongBench avg, Llama-70B, SpecPrefill 10% / 30% / 50% kept: 52.74 / 53.92 / 53.59 | baseline: full-prompt baseline 53.55 | ratio: 0.985 / 1.007 / 1.001 | Table 1 p7
- Same table, RAG baseline at 10.38% / LLMLingua ~10%: 35.49 / 29.85 | baseline: baseline 53.55 | ratio: 0.66 / 0.56 | Table 1 p7
- RULER 70B, 10% kept, average without aggregation, 4K...128K: 88.2, 91.7, 91.3, 90.8, 90.4, 65.6 | baseline: 70B full prompt: 92.3, 91.5, 90.6, 89.8, 87.9, 60.3 | ratio: 0.96, 1.00, 1.01, 1.01, 1.03, 1.09 | Table 2 p8
- RULER 128K, multi-hop tracking and QA at 10% kept: 55.6 / 55.3 | baseline: full 56.1 / 48.2 | ratio: 0.99 / 1.15 | Table 2

**Catalog and first-pass numbers checked:**
- 7.66x TTFT on Llama-3.1-405B-FP8 by pruning the prompt -> YES. 7.66x at 10% kept, batch x seq up to 16x32K, 8x H200, vs unpruned 405B
- first-pass: 8x H100/H200 TP8, vLLM, 8B speculator -> YES. H100 for quality, H200 for efficiency runs
- first-pass: ~7x QPS at >95% accuracy -> YES. Multi-Doc QA only
- first-pass: aggregation tasks degrade at 10% keep -> YES. Table 2 aggregation column (CWE/FWE) at 10% keep: 77.9 vs full-prompt 99.7 at 4K, 70.0 vs 96.9 at 32K; the average column excludes aggregation

**Transfers to our regime:** partly. The prune-the-prompt idea transfers; the 7.66x does not. (1) GPU TP8 where prefill is compute-bound, so cost falls ~linearly with kept tokens (paper: 'reduces total FLOPS proportional to the percentage of token drop'); on NVMe-streamed experts, a 500-token prefill touches most experts anyway, so bytes saturate and our own bound is 1.48x/1.80x/2.77x at 50/25/12.5% kept (F.md). (2) The speculator is an 8B dense model run over the FULL prompt first; its overhead is 2% of a 405B but 8B dense = ~16 GFLOP/token vs ~12 GFLOP/token active for the 125B MoE (my arithmetic from F.md's 12 GFLOP), so on our target the speculator would cost MORE compute than the target; a 0.6-1.7B speculator (as in P-0245) is needed, and its RAM competes with 8-32 GB. (3) Batch of 16-128 sequences, speedups grow with batch*seq. (4) Needs the speculator resident plus 2-pass logic; no multi-turn recompute (stated limitation).

**Quality loss:** Lossy. LongBench 70B: 98.5% of baseline at 10% kept, 100.7% at 30% (Table 1). RULER at 10% kept: aggregation tasks drop (CWE/FWE 77.9 vs 99.7 at 4K); standard short tasks degrade at low keep rate because prompts are information dense (Appendix Fig 6, text Sec 4.6). No perplexity/logit outputs (cannot score prompt tokens).

**Ablation:** Variants SpecPrefill / Full / Full+look-ahead (Fig 2, Fig 8), keep-rate sweep, batch x seq sweep (Fig 3/4), vs MInference (Fig 5: 2.54x-6.54x relative speedup). Nothing isolating I/O or MoE.

**Limitations stated:** Stated (Sec 5): no explicit logits for all input tokens, so only generative evaluation; multi-turn conversations can fail without KV recompute; need a better method to choose how many tokens to keep; aggregation tasks are information-dense.

**Contradicts / supports our findings:** No contradiction. It supports our F.md statement that prompt pruning only pays when compute (not bytes) scales with tokens; the paper's 7.66x is within 1% of its own FLOP-ratio bound (7.72x), i.e. it is a pure compute-scaling result. Our 'bytes saturate' bound for MoE-from-disk is a regime the paper never tests.

## P-0245

**Title/venue:** Cross-Family Speculative Prefill: Training-Free Long-Context Compression with Small Draft Models. Shubhangi Upasani, Ravi Shanker Raju, Bo Li, Mengmeng Ji, John Long, Chen Wu, Urmish Thakker, Guangtao Wang (SambaNova). v3 header: 'Published as a conference paper at ICLR 2026'

**Read:** arXiv 2603.02631v3 (13 Mar 2026), pdftotext of full PDF (62 kB text); ICLR virtual page 2026/10018734 also surfaced by search (raw_text_read=True)

**Setup:** Targets: Llama-3.1-8B, Qwen3-8B, DeepSeek-R1-0528 and DeepSeek-V3/V3.1 (671B MoE, 37B active, from memory for the sizes); drafts Llama-3.2-1B, Llama-3.1-8B, Qwen3-0.6B/1.7B/4B-2507. Same attention-importance chunk selection as SpecPrefill but spans re-tokenised by the target and re-numbered contiguous positions. Benchmarks LongBench v1/v2, RULER (128k), InfiniteBench Code Debug. Greedy. Hardware: SambaNova custom RDU (not GPU/CPU), max target-side context 32k; KV in 4k blocks. TTFT numbers only in prose (no table).

**Numbers (what / value / baseline / ratio / where):**
- RULER 128k -> 16k (12.5% kept), DeepSeek-V3 target: TTFT: ~2.5 s | baseline: 46 s (full prompt, with SnapStream KV compression) | ratio: 46/2.5 = 18.4x (paper says ~18x) | Sec 5.2 p5 (text only; no table)
- Same, compress to 32k: TTFT: 4.3 s | baseline: 46 s | ratio: 10.7x | Sec 5.2 p5
- RULER accuracy, DeepSeek-V3, 12.5% kept, draft Qwen3-4B / Llama-3.1-8B: 89.67 / 93.36 | baseline: full prompt 80 (baseline degraded by SnapStream KV compression) | ratio: 1.12 / 1.17 (>1 because baseline is handicapped) | Table 4 p6
- LongBench v1, target Qwen3-8B, draft Llama-3.2-1B-Instruct, 45% kept: accuracy: 55.98 | baseline: full prompt 57.34 | ratio: 0.976 | Table 3 p5
- LongBench v1, target Llama-3.1-8B, draft Qwen3-1.7B, 45% kept: 54.29 | baseline: full prompt 55.31 | ratio: 0.982 | Table 3 p5
- LongBench v2, Llama-3.1-8B, 25% kept: Llama-3.2-1B draft / Qwen3-1.7B draft: 29.4 / 29.6 | baseline: full prompt 31.2 | ratio: 0.94 / 0.95 | Table 1 p5
- LongBench v2, DeepSeek-R1, 6% kept (Qwen3-4B / Llama-3.1-8B draft) and 3% (Qwen3-1.7B): 53.3 / 54.1 / 45.9 | baseline: full prompt 58.3 | ratio: 0.91 / 0.93 / 0.79 | Table 1 p5
- Code Debug, DeepSeek-V3.1, 20% / 15% kept: 64.72 / 59.13 | baseline: full 67.51 | ratio: 0.96 / 0.88 | Table 2 p5
- Code Debug, DeepSeek-R1-0528, 30% / 25% / 15% kept: 70.30 / 68.02 / 62.44 | baseline: full 74.37 | ratio: 0.95 / 0.91 / 0.84 | Table 2 p5

**Catalog and first-pass numbers checked:**
- Compressing a 128k prompt to 16k cut TTFT from 46 s to about 2.5 s on custom RDU hardware -> YES. 46 s to ~2.5 s, DeepSeek-V3 target; the 46 s baseline already uses SnapStream KV compression
- Accuracy 55.98 vs 57.34 with 45% of the prompt kept -> YES. Table 3: target Qwen3-8B, draft Llama-3.2-1B-Instruct, 45% kept, average over 8 LongBench-v1 subtasks
- catalog '(Llama-3.1-8B dra...' (truncated) for the 55.98 row -> NO. The 55.98 row uses a Llama-3.2-1B-Instruct draft for a Qwen3-8B target; Llama-3.1-8B is the TARGET in the other Table 3 row (54.29 vs 55.31, Qwen3-1.7B draft). Catalog text is truncated so the exact wording cannot be judged
- first-pass: venue is an ICLR 2026 workshop -> NO. v3 header says 'Published as a conference paper at ICLR 2026' and the ICLR virtual site lists it; first pass was wrong or read an older version
- first-pass: baseline used KV compression -> YES. SnapStream on RDU for the 128k RULER baseline (Sec 5.1)

**Transfers to our regime:** no. Custom dataflow RDU accelerator with a 32k target-context cap, TTFT numbers are prose-only and the baseline is handicapped (SnapStream KV compression, which the paper says causes retrieval loss, so the >100% accuracy is partly baseline damage). Use as quality evidence only: a 1-4B draft is enough to pick tokens for an 8B target, which matters for us because it shrinks the speculator cost (0.6-1.7B = 1-3 GFLOP/token vs 12 for the 125B MoE's active path). It does not touch expert bytes. Prompts of 8k-128k tokens, whereas our interactive prompts are ~64-500 tokens where selection overhead exceeds the saving.

**Quality loss:** Reported 90-100% of full-prompt accuracy typically (abstract); concrete worst cases: DeepSeek-R1 LongBench-v2 3% kept = 79% of full; Code Debug 15% kept = 84-88%. LongBench v2 numbers are 29-32% absolute (near-chance multiple choice), so deltas are noisy.

**Ablation:** Draft model size/family sweep (Sec 5.3, Table 1: in-family vs cross-family draft differ by <=0.6 points at 25-50% kept). No timing ablation beyond the two TTFT values.

**Limitations stated:** Appendix A.5 task-specific constraints (aggressive compression hurts long-context code debugging, Table 2); RDU max 32k target context; keep rate per sample varies after re-tokenisation.

**Contradicts / supports our findings:** None. Does not report any MoE-specific cost; DeepSeek-V3 is MoE but resident in RDU memory.

## P-0246

**Title/venue:** Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve. Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, Ramachandran Ramjee (MSR India, Georgia Tech). OSDI 2024 (v3 acknowledges 'OSDI reviewers', artifact branch named osdi-sarathi-serve; year 2024 is from memory/arXiv date).

**Read:** arXiv 2403.02310v3 (17 Jun 2024) AND v1 (4 Mar 2024), both pdftotext of full PDF (raw_text_read=True)

**Setup:** Mistral-7B (1x A100 80GB), Yi-34B (2x A100, TP2), LLaMA2-70B (8x A40 48GB, TP4-PP2), Falcon-180B (2 nodes x 4 A100, TP4-PP2, 100 Gbps ethernet). Baselines vLLM and Orca. Workloads openchat_sharegpt4 (median prompt 1730, median output 415) and arxiv_summarization (median prompt 7059, output 208), Poisson arrivals. SLO on P99 TBT: strict 0.1/0.2/1/1 s, relaxed 0.5/1/5/5 s for the four models (Table 3), median scheduling delay <2 s. Token budgets 512 (strict) and 2048 (relaxed). Own system on vLLM base.

**Numbers (what / value / baseline / ratio / where):**
- Capacity, Mistral-7B 1xA100 (v3 abstract and v1 abstract): 2.6x | baseline: vLLM | ratio: 2.6x | abstract v1 & v3
- Capacity, Yi-34B 2xA100: v3: up to 3.7x; v1: up to 2.8x | baseline: vLLM | ratio: 3.7x (final) vs 2.8x (v1) | v3 abstract/intro p1-2; v1 abstract
- End-to-end capacity Falcon-180B with pipeline parallelism: v3: up to 5.6x; v1: 6.9x | baseline: vLLM | ratio: 5.6x (final) vs 6.9x (v1) | v3 abstract; v1 abstract
- Capacity bars in Fig 10/11 (examples read from the figure text): 2.78x, 2.15x, 4.00x, 2.44x, 5.54x, 6.31x, 4.69x, 5.62x | baseline: vLLM or Orca per bar | ratio: n/a (bar labels, which model/SLO each belongs to is not recoverable from pdftotext) | Fig 10-11 p12-13
- Median TTFT, Yi-34B, token budget 1024, sharegpt4: hybrid-batching-only / chunked-only / combined: 0.53 / 1.04 / 0.76 s | baseline: hybrid-only 0.53 | ratio: chunking alone raises TTFT 1.96x vs hybrid-only; combined 1.43x | Table 4 p15
- Same on arxiv_summarization (P50 TTFT): 3.78 / 5.38 / 3.90 s | baseline: hybrid-only 3.78 | ratio: 1.42x / 1.03x | Table 4 p15
- Chunked-prefill compute overhead, Yi-34B: <= ~25% at chunk 512 | baseline: no chunking | ratio: <=1.25x prefill time | Fig 14 / Sec 5.4.1 p14

**Catalog and first-pass numbers checked:**
- up to 2.8x serving capacity on Yi-34B -> YES (V1 ONLY). 2.8x is the v1 abstract (the catalog URL is /html/2403.02310v1); the published/v3 text says up to 3.7x for Yi-34B
- first-pass: capacity up to 2.6x (Mistral-7B), 2.8x (Yi-34B), 6.9x (Falcon-180B) -> PARTLY. All three are v1 numbers; v3 (final): 2.6x, 3.7x, 5.6x
- first-pass: Mistral-7B/Yi-34B/LLaMA2-70B/Falcon-180B on A100/A40 -> YES. Table 1
- 'does not emit tokens before the whole prompt is processed' -> YES. Sec 2.2: prefill produces the first output token; chunked-prefill-only raises TTFT (Table 4)

**Transfers to our regime:** no. Serving-capacity method for dense models on multi-GPU clusters with many concurrent requests under TBT SLOs; the core mechanism (mixing a prefill chunk into a decode batch so decodes do not stall) has no meaning at batch 1 on a laptop. No MoE and no weight streaming. Its relevance is negative: chunking adds compute overhead (<=25% at 512) and its own Table 4 shows chunking alone raises TTFT.

**Quality loss:** None (lossless scheduling).

**Ablation:** Table 4 / Sec 5.4.2: hybrid-batching-only vs chunked-only vs combined; chunk overhead Fig 14; token budget sweep Fig 12-13. None relevant to expert reads.

**Limitations stated:** Token budget must be profiled per model/hardware ('a complex' choice, Sec 4.3); chunk size sensitivity (257 vs 256 costs 32% prefill, Sec 4.3); designed for TP/PP GPU deployments.

**Contradicts / supports our findings:** None. Note it is the origin of the token-chunked prefill whose expert reloads P-0247 quantifies.

## P-0247

**Title/venue:** From Tokens to Layers: Redefining Stall-Free Scheduling for MoE Serving with Layered Prefill. Gunjun Lee, Jiwon Kim, Jaiyoung Park, Younjoo Lee, Jung Ho Ahn (header: Proceedings of the 9th MLSys Conference, Bellevue WA, 2026)

**Read:** arXiv 2510.08055v2 (16 Apr 2026), pdftotext of full PDF (172 kB text) (raw_text_read=True)

**Setup:** Qwen3-30B-A3B (128 experts, top-8, 30B) and GPT-OSS-20B (32 experts, top-4), bf16 (Table 3); 2x H100 80GB (default), extra rows on 2x A100, 8x H100 with Qwen3-235B-A22B and gpt-oss-120b (fp4); custom serving stack on vLLM-like scheduler; datasets ShareGPT (mean input 2340, output 438) and arXiv summarization (mean input 9194, output 231); Poisson arrivals; SLO TTFT 5-10 s, TBT 100-125 ms (Table 5); baseline chunked prefill, chunk 512; layered prefill with Nlg layer groups (default 16 for Qwen). Decode batch sizes 32-128 in stable regime.

**Numbers (what / value / baseline / ratio / where):**
- Total expert-weight loads for 100 requests, Qwen, arXiv: layered 21.7 TB | baseline: chunked(512) 35.6 TB | ratio: 35.6/21.7 = 1.64x more loads for chunked (= 39% fewer for layered) | Table 7 p12
- Same, ShareGPT: layered 25.1 TB | baseline: chunked 28.5 TB | ratio: 1.14x (12.0% fewer) | Table 7 p12
- Qwen arXiv at 1.3 req/s: mean TTFT / p99 TTFT: 1.24 / 4.10 s | baseline: chunked 2.80 / 8.65 s | ratio: 2.26x / 2.11x (-55.7% mean) | Table 6 p11
- Same: mean TBT / p99 TBT: 21.5 / 37.1 ms | baseline: chunked 32.9 / 51.1 ms | ratio: 1.53x / 1.38x lower | Table 6
- GPT-OSS, same request rate (2.1 req/s): mean TTFT reduction: >70% | baseline: chunked | ratio: >3.3x | Sec 5.6 p13 (text)
- gpt-oss-120b fp4, 2xH100, arXiv 1.2 req/s: mean TTFT: 1.090 s | baseline: chunked 3.110 s | ratio: 2.85x (-65%) | Table 9 p13
- Qwen end-to-end latency (single request trace): 5.5 s | baseline: 9.4 s | ratio: 1.71x (-41%) | Sec 5.5 p12
- Energy per output token, Qwen arXiv, sustained 1.6 vs 1.3 req/s: 44.2 mJ | baseline: 56.6 mJ | ratio: -22% | Table 8 p13
- Expert coverage of a decode batch (Qwen): batch 1: 6.25%, 8: 29.0%, 32: 54.7%, 128: 86.3%, >=512: >=98% | baseline: 128 experts | ratio: batch 1 = 8/128 | Table 1 p4
- Dense control Qwen3-8B 1xH100, arXiv 1.5 req/s: mean TTFT: layered 1.45 s | baseline: chunked 0.955 s | ratio: layered 1.52x SLOWER | Sec 5.13 p15
- Peak offline throughput H100x2: 391.59 tok/s | baseline: 315.36 tok/s | ratio: 1.24x | Sec 5.9 p14

**Catalog and first-pass numbers checked:**
- Token-chunked prefill reloads MoE expert weights for each chunk -> YES. Sec 3.1-3.2: per-chunk MoE load roughly constant regardless of chunk size (Fig 2)
- The paper moves to layer-wise chunking for MoE -> YES. layer groups Nlg as scheduling unit
- first-pass: TTFT up to -70% (2.80 s to 1.24 s, Table 6) -> NO (CONFLATED). 2.80 -> 1.24 s is -55.7% (Table 6); 'more than 70%' is the GPT-OSS statement in Sec 5.6 and the abstract's 'up to 70%'
- first-pass: latency -41% -> YES. 9.4 -> 5.5 s single-request trace, Sec 5.5
- first-pass: expert loads 35.6 vs 21.7 TB (Table 7) -> YES. arXiv workload; ShareGPT 28.5 vs 25.1
- first-pass: 39% framing mixed in paper -> YES. abstract: chunked prefill 'increase[s] memory traffic by up to 39%' but Table 7 gives layered 39% lower than chunked, i.e. chunked is +64% over layered
- first-pass: batch 32-128 -> YES. decode batch avg <32 on arXiv, <128 on ShareGPT (Sec 3.1)
- our own 4.4-6.0x re-read figure vs this paper -> DIFFERENT QUANTITY. paper measures total expert loads (prefill+decode, 100 requests, GPU HBM) = 1.64x on arXiv with ~18 chunks per prompt; our 4.4-6.0x is prefill-only, 8 chunks of 64 tokens, on a tiny cache (F.md). Not comparable, not a contradiction

**Transfers to our regime:** partly. Directionally yes (the mechanism, redundant expert reloads per token chunk, is precisely the cost that dominates NVMe streaming), but all numbers are for GPU HBM-resident experts with concurrent requests, 512-token chunks, SLO-bound TBT and decode batches of 32-128. Batch-1 laptop prefill has no decode to interleave with, so the 'stall-free' goal does not exist; the useful kernel is simply 'for a prompt, process layer-major (all tokens through layer l, read each expert once) instead of token-chunk-major', i.e. our expert-major prefill idea (B.md #9). Assumptions that break: GPU, batch>1, SLO scheduling; the paper does not use disk or CPU.

**Quality loss:** None (scheduling change, lossless).

**Ablation:** Group-size Nlg sweep (Table 11: Nlg 2/4/8/16 TTFT 0.48/0.57/0.77/1.27 s, p99 TBT 138/84.5/54.6/35.7 ms); dense-model control (slower); disaggregated serving comparison (Table 10: TTFT 3.00 chunked, 3.94 disaggregated, 1.24 layered); relaxed-TBT convergence (chunk 2048 == Nlg 4, TTFT 2.47 s both). Closest to isolating the 'redundant reload' cost.

**Limitations stated:** Conclusion: multi-GPU extension and adaptive layer grouping are future work; the gain vanishes with large chunks (relaxed SLO) and is negative for dense models; layer count must divide into groups.

**Contradicts / supports our findings:** No contradiction; supports our 'token-chunked prefill re-reads experts' finding qualitatively. The magnitude (1.64x) is much lower than our 4.4-6.0x because their chunks are 8x larger (512 vs 64) and loads include decode.

## P-0249

**Title/venue:** Prompt Lookup Decoding: llama.cpp PR #4484 'Prompt lookup decoding' (LeonEricsson, opened 2023-12-15, merged 2023-12-22 by ggerganov) + apoorvumang/prompt-lookup-decoding README (Apoorv Saxena, Nov 2023) + llama.cpp docs/speculative.md (current master)

**Read:** PR: GitHub API JSON for pull 4484 and all 34 issue comments; README raw from raw.githubusercontent.com; docs/speculative.md raw (24 kB). The catalog URL itself (cdn04132025.gitlink.org.cn/.../commit/7082d24cec35e9ce9147535a2224df, truncated) returns a 'Sign In - Trustie' page: UNREAD. (raw_text_read=True)

**Setup:** README: Mistral-7B-Instruct-v0.1, single A100 40GB, greedy, max n-gram 3, continuation length 10; datasets CNN/DM (100), HAGRID context QA (100), MT-Bench (80); baseline plain greedy; numbers are in image plots, the text states the result. PR thread: Mistral-7B-instruct-v0.2 f16/Q5_K_S/Q4_0, Solar-10.7B Q5_K_M, Deepseek on Apple M1 Max 32GB (apoorvumang) and M2 Pro 16GB (x4080), llama.cpp Metal, no CUDA numbers.

**Numbers (what / value / baseline / ratio / where):**
- README: average speedup on summarization and context-QA: ~2.4x (stddev bars; 'always more than greedy or within error') | baseline: greedy, A100 | ratio: 2.4x | README Results, plot image, p. text only
- README tagline: 2x-4x on input-grounded tasks | baseline: greedy | ratio: 2-4x | README TLDR
- PR thread, M1 Max 32GB, --draft 10 (Q5_K_S): decode 167 tokens: 9.749 s, 17.13 t/s, accept 9.231% (72/780) | baseline: --draft 0: 144 tokens in 4.226 s, 34.07 t/s (different output length: nondeterminism flagged in thread) | ratio: 0.50x speed | comment 2023-12-19
- PR thread, Q4_0 --draft 1, 197 tokens: 10.959 s, 17.98 t/s, accept 48.696% (56/115) | baseline: --draft 0: 3.980 s, 49.49 t/s | ratio: 0.36x speed | comment 2023-12-20
- PR thread, 'prompt that took ~8 s takes 13 s': 13 s vs 8 s | baseline: main | ratio: 0.62x | comment 2023-12-19 (apoorvumang)
- Final remark by maintainer: performance varies with acceptance rate, n_draft and batch decode speed; 'Probably no reason to go above 4-5 for n_draft' | baseline: n/a | ratio: n/a | comment 2023-12-21 (ggerganov)

**Catalog and first-pass numbers checked:**
- Exact mechanism already exists in llama.cpp: n-gram match in prompt, draft, batch verify -> YES. PR #4484 (examples/lookup) and today's docs/speculative.md list ngram-cache, ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod (flag --spec-type)
- flags --spec-type ngram-cache/ngram-simple/ngram-map -> YES (VERIFIED FROM RAW DOCS, FIRST PASS USED DIGESTS ONLY). docs also list ngram-map-k4v and ngram-mod; ngram-mod example says 'MoEs require long drafts' (n-match 24, n-min 48, n-max 64); ngram-mod pool is shared across server slots (~16 MB)
- Gains depend on repetition (code refactoring, editing) -> YES. docs: use for code rewriting, reasoning-model repeats, summarization; README: roleplay worst, coding turn 2 best
- first-pass: PR slowdown on M1 Max 13 s vs 8 s, accept 9.2%-48.7% -> YES BUT INCOMPLETE. Numbers are real but come from the first, disputed implementation (maintainer-contributor exchange of 2023-12-20 questions whether verification was batched) and from two different runs (9.2% at draft 10, 48.7% at draft 1); not a clean measurement of the merged code. Other users on M2 Pro 16GB and a Mac laptop also reported slowdown for Solar-10.7B
- first-pass: README ~2.4x summarization/QA -> YES. A100 40GB; PR #4484 itself reports no GPU numbers

**Transfers to our regime:** partly. The mechanism is the thing our idea (P-0248) is, so it transfers as prior art. The speedup does not: README is a dense 7B on an A100 where verifying 10 tokens costs about the same as 1; the Apple laptop reports in the PR are slowdowns because prompt-eval per token was not much faster than decode per token ('latency when doing a multi-token decoding step', #4624 cited by the author). On our MoE a k-token verify reads the UNION of experts (our F.md: pass cost 0.40 s for 1 token, 1.79 s for 8 at 8 GB), the same effect, stronger. Assumption broken: verify pass is nearly free.

**Quality loss:** None claimed (lossless greedy); thread shows different outputs at temp 0 between --draft 0 and >0 for Q5_K_S (not for Q4_0), attributed to numerics, unresolved in thread.

**Ablation:** None systematic; README: MT-bench turn 0 vs turn 1 (self-match only vs prompt match) and by category (extraction/coding high, roleplay lowest).

**Limitations stated:** README: hyperparameters 3 and 10 chosen with little testing; need real FLOPs analysis; sampling not tested. PR TODOs: batching multiple matches, better search, vary n-gram size.

**Contradicts / supports our findings:** No. Agrees with F.md: low-overlap text gives speedup <=1.0 at 8 GB. The laptop slowdowns in the thread are dense-model results that independently agree.

**Unread parts:** Catalog URL itself UNREAD (truncated mirror, sign-in wall); the substitute sources above were read raw.

## P-0250

**Title/venue:** SAM Decoding: Speculative Decoding via Suffix Automaton. Yuxuan Hu, Ke Wang, Xiaokang Zhang, Fanjin Zhang, Cuiping Li, Hong Chen, Jing Zhang (Renmin U of China, Tsinghua). Venue not printed in the PDF text; ACL 2025 is from the catalog / memory.

**Read:** arXiv 2411.10666 v3 (16 Dec 2024) full text; v1 and v2 also read (for the abstract numbers) (raw_text_read=True)

**Setup:** Main: Vicuna-7B-v1.3, one RTX A6000 48GB, 20-core CPU, PyTorch 2.3.0, fp16, greedy, batch 1; Spec-Bench (6 task types), HumanEval, HAGRID; also Vicuna-13B/33B and Llama3-8B, an A800 repeat (Table 9). Static suffix automaton built from Vicuna-7B outputs on Stanford-alpaca, GSM8K, python-instruct-18k plus dynamic automaton over the live context; draft size 40 (16 for code); l_bias=l_threshold=5; auxiliary drafters Token Recycling [T] and EAGLE-2 [E2]. Baselines: autoregressive, Lookahead, REST, PIA, PLD, Token Recycling, EAGLE-2.

**Numbers (what / value / baseline / ratio / where):**
- Spec-Bench Vicuna-7B: SAM-Decoding (alone): 1.84x, MAT 2.30, 69.37 tok/s | baseline: PLD 1.56x (59.02 tok/s); Token Recycling 1.84x (69.65) | ratio: 1.84/1.56 = 1.18x over PLD; equal to Token Recycling | Table 1 p6
- Spec-Bench: SAM-Decoding[T]: 2.27x (85.73 tok/s) | baseline: Token Recycling 1.84x | ratio: 1.23x over Token Recycling | Table 1 p6
- Spec-Bench: SAM-Decoding[E2]: 2.58x (97.56 tok/s, MAT 4.62) | baseline: EAGLE-2 2.38x (90.14) | ratio: 1.08x over EAGLE-2 | Table 1 p6
- HumanEval / HAGRID, SAM alone: 2.29x / 2.24x | baseline: PLD 1.52x / 1.29x | ratio: 1.51x / 1.74x over PLD | Table 1
- Llama3-8B Spec-Bench: SAM / PLD / Token Recycling / SAM[T] / EAGLE-2 / SAM[E2]: 1.51 / 1.28 / 1.91 / 2.05 / 1.98 / 2.08x | baseline: autoregressive | ratio: SAM = 1.18x PLD | Table 3 p13
- Vicuna-13B Spec-Bench: SAM vs PLD: 1.84x vs 1.59x | baseline: autoregressive | ratio: 1.16x | Table 5 p14
- A800 Vicuna-7B: SAM[E2] vs EAGLE-2: 2.58x vs 2.21x | baseline: autoregressive | ratio: 1.17x | Table 9 p15
- v1/v2 abstract values: SAM[T] 2.27x, SAM[E2] 2.49x (EAGLE2 2.38x) | baseline: autoregressive | ratio: - | v1 Table / v2 abstract; superseded in v3 by 2.58x

**Catalog and first-pass numbers checked:**
- 2.27x over autoregressive on Spec-Bench with Token Recycling -> YES. SAM-Decoding[T] 2.27x, Table 1 (v1, v2, v3)
- 2.49x with EAGLE2 -> YES FOR V1/V2, NO FOR V3. v1 and v2 abstracts state 2.49x; v3 Table 1 says 2.58x (and the cited URL .../html/2411.10666v3 is v3)
- first-pass: '2.27x/2.49x not found' -> WRONG. 2.27x is in v3 Table 1; 2.49x is in v1/v2 only. First-pass digest missed these
- first-pass: 1.84x vs PLD 1.56x; SAM+EAGLE2 2.58x vs EAGLE-2 2.38x; HumanEval 2.29x -> YES. Table 1
- first-pass: Vicuna-7B, A6000, batch 1 -> YES. plus A800 repeat
- 'dense models, I/O not binding' -> YES. all models dense, fp16, resident on GPU

**Transfers to our regime:** no. Dense 7-33B models fully resident on an A6000/A800, batch 1, where verifying a 40-token draft is nearly free. On a streamed-expert MoE, a long draft (default 40) makes the verifier read the union of the experts of every drafted position: pass cost scales with draft length (F.md: 0.40 s for 1 token to 1.79 s for 8 at 8 GB). Also the static corpus is built from Vicuna outputs on three datasets: the retrieval source would need to exist for our model (paper's own limitation). The dynamic-context automaton part (equivalent to ngram-map in llama.cpp) is what would apply, and the paper's own PLD comparison (1.56x) already shows gains are mostly repeat-in-context.

**Quality loss:** None (lossless, greedy verification).

**Ablation:** Table 2: draft generation methods (SAM vs PLD vs REST etc.), effect of l_threshold and draft size (figures), static vs dynamic automaton (Fig). No I/O ablation.

**Limitations stated:** Sec 7: depends on task type and the quality of the retrieval corpus (corpus built from Vicuna-7B outputs on Alpaca, GSM8K, python-instruct; may not match other LLMs); gains when combined with other SD methods can shrink.

**Contradicts / supports our findings:** None. The 1.84x vs PLD 1.56x gap shows suffix match beats fixed n-gram on dense GPU, but this was not tested on text with <20% overlap, where our F.md finds f=0.11-0.16.

## P-0251

**Title/venue:** Speculative Decoding: Performance or Illusion? Xiaoxuan Liu*, Jiaxiang Yu*, Jongseok Park, Ion Stoica, Alvin Cheung (UC Berkeley). Venue not printed.

**Read:** arXiv 2601.11580v2 (18 Mar 2026), pdftotext of full PDF (201 kB text) (raw_text_read=True)

**Setup:** vLLM v0.10.1.1 on NVIDIA H100 80GB (1 GPU for 8B, 4 GPU TP4 for 70B and 106B); models Llama3.1-8B-Instruct, Llama3-70B-Instruct, Qwen3-8B (+Thinking), GLM-4.5-Air-106B; variants n-gram (3 or 5 draft tokens, prompt_lookup max 7 min 3), EAGLE, EAGLE-3, draft-model (3 tokens), MTP; datasets ShareGPT, GSM8K, CNN/DailyMail, InstructCoder (non-reasoning), AIME22-24, GPQA-Main (reasoning); batch sizes 1-128; 500 requests per dataset; all vLLM optimisations on (continuous batching, chunked prefill, CUDA graphs). Metric throughput ratio (speedup). Also an oracle upper-bound simulator.

**Numbers (what / value / baseline / ratio / where):**
- Mean generated tokens per verification step, n-gram: InstructCoder / CNN-DM / ShareGPT / GSM8K: 7.27 / 2.33 / 2.02 / 1.41 | baseline: EAGLE-3 4.18 / 3.08 / 3.06 / 3.02 (EAGLE: 4.24 / 2.32 / 2.82 / 2.56) | ratio: n-gram/EAGLE-3 = 1.74 / 0.76 / 0.66 / 0.47 | Table 4 p21 (averaged Llama3.1-8B and Llama3-70B)
- Same, reasoning: GPQA-Main / AIME22-24 (Qwen3-8B-Thinking): 4.20 / 3.53 | baseline: EAGLE-3 2.78 / 2.62 | ratio: 1.51 / 1.35 | Table 4
- n-gram std of that length: 4.19 (InstructCoder), 1.87 (ShareGPT), 0.45 (GSM8K) | baseline: EAGLE-3 0.99 / 0.87 / 0.47 | ratio: std 2-5x higher than EAGLE (text) | Table 4, Sec 6 p9
- BLEU-4 overlap above which n-gram beats EAGLE/EAGLE-3 at all batch sizes (Llama3.1-8B, InstructCoder): > ~0.6 | baseline: EAGLE, EAGLE-3 | ratio: n-gram up to +53% speedup over EAGLE (3 draft tokens), up to +100% (5 draft tokens) | Sec 7 p10, Fig 9
- EAGLE speedup on GSM8K, Llama3.1-8B, batch 1 -> larger batch: 1.73x -> 1.21x | baseline: no-SD | ratio: -30% | Sec 4 p5 (intro text)
- Tree k=21 at batch 64: falls below 1x on all workloads | baseline: no-SD | ratio: <1.0 | Sec 4 (tree-style verification) p5
- Reasoning workloads, draft-model/EAGLE-3 speedup (for scale): 1.50x-1.58x (n-gram comparable), 1.64x-1.80x best | baseline: no-SD | ratio: - | Sec 4 p7

**Catalog and first-pass numbers checked:**
- N-gram speedup tracks prompt/output overlap -> YES. Sec 7: higher BLEU-n strongly correlates with n-gram speedup (Fig 9)
- low-overlap requests lose -> PARTLY. Text says n-gram 'underperforms' EAGLE/EAGLE-3 at low overlap and is 'generally less effective than other SD approaches across most workloads'; the text does not state it falls below plain decoding (Fig 1 values are only in plots, not recoverable from the extracted text). 'Lose' is true against EAGLE, unproven against no-SD
- first-pass: InstructCoder 7.3, CNN/DM 2.3, ShareGPT 2.0, GSM8K 1.4 (Table 4) -> YES. 7.27 / 2.33 / 2.02 / 1.41; these are mean tokens produced per step (accepted + bonus), averaged over two models
- first-pass: BLEU>0.6 beats EAGLE -> YES. specifically Llama3.1-8B on InstructCoder
- first-pass: bursty acceptance, std 2-5x -> YES. Sec 6

**Transfers to our regime:** partly. The overlap law transfers (it is what our F.md measured: f 0.11-0.16 prose/code, 0.59 JSON/edit). Numbers do not: GPU vLLM H100 with CUDA graphs, batch 1-128, drafts of 3-5 tokens, verification time dominated by the dense forward, not weight bytes. The paper shows verification cost dominates execution time even on GPU (Sec 5), which is consistent with our expert-union cost. Our regime has bigger penalty per rejected token (whole expert union from NVMe).

**Quality loss:** None in theory (lossless); observed output differences attributed to nondeterminism, with the SD generation length not systematically longer or shorter (Table 1).

**Ablation:** Execution-time breakdown (drafting/verification/rejection, Sec 5), per-position acceptance (Fig 11), tree vs chain (SGLang), proposal length 3 vs 5, oracle upper bound (Sec 8). No MoE-expert ablation except GLM-4.5-Air MTP vs n-gram.

**Limitations stated:** BLEU overlap measured between prompt and full output only, not the evolving context (Sec 7, left for future work); evaluated on vLLM only; results sensitive to numerics.

**Contradicts / supports our findings:** None; consistent with F.md (low-overlap = no gain). Table 4 values imply our f=0.107 for prose is near the paper's ShareGPT/GSM8K regime (1.4-2.0 tokens per step incl. bonus), i.e. accepted drafts 0.4-1.0 per step, a higher acceptance than our stand-in prose corpus (0.11). Mild: our stand-in corpora (Python docs) may under-estimate real chat overlap.

## P-0252

**Title/venue:** MoE-Spec: Expert Budgeting for Efficient Speculative Decoding. Bradley McDanel, Steven Li, Sruthikesh Surineni, Harshit Khaitan (header: arXiv v1, 17 Feb 2026). Catalog URL (emergentmind.com/topics/moe-spec) is an aggregator page, NOT read; primary papers read instead: MoE-Spec 2602.16052, EVICT 2605.00342, SpecMoEOff 2508.21706.

**Read:** arXiv 2602.16052v1 full text; arXiv 2605.00342 (EVICT) full text, numbers via grep; arXiv 2508.21706 (SpecMoEOff) abstract+evaluation parts; SP-MoE 2510.10302, SpecMoE 2604.10152, MoESD 2505.19645: first page only (raw_text_read=True)

**Setup:** MoE-Spec: OLMoE-1B-7B (64 experts, k=8), Qwen3-30B-A3B (128, k=8), Mixtral-8x7B (8, k=2); EAGLE-3 drafts (OLMoE, Qwen3) and EAGLE-1 (Mixtral); NVIDIA A100 80GB (Mixtral on 2), single request, tree 63 tokens (sweep 3-511), T=0 and T=1, 80 samples per benchmark, max gen 1024; benchmarks GSM8K, MATH500, HumanEval, MBPP, CNN/DM. Lossy: ranks experts by summed routing probability over the draft tree and loads only top-B per layer (substitution policy). EVICT (Pan et al.): lossless prefix selection by expected-accepted/verify-cost, Qwen3-30B-A3B and two other MoEs, A100 SXM4 80GB, PyTorch and SGLang. SpecMoEOff (Wang et al.): Mixtral 8x7B with CPU-offloaded experts, A30 (250 GB CPU RAM) and RTX 4090D (190 GB), large-batch, CPU chunked-attention verification kernel.

**Numbers (what / value / baseline / ratio / where):**
- MoE-Spec vs EAGLE-3 speedup, Qwen3-30B-A3B GSM8K T=0: 2.4x | baseline: EAGLE-3 1.9x (AR 1.0x) | ratio: 1.26x | Table 1 p5
- MoE-Spec vs EAGLE, Mixtral-8x7B MBPP T=0: 2.3x | baseline: EAGLE 1.7x | ratio: 1.35x (paper: '35% relative') | Table 1 / Sec 4.2 p4-5
- MoE-Spec average relative speedup over EAGLE per model: Mixtral 27%, Qwen3-30B 16%, OLMoE 6% | baseline: EAGLE(-3) | ratio: 1.27 / 1.16 / 1.06 | Sec 4.2 p4
- Mean acceptance change: -1.4% average | baseline: EAGLE-3 | ratio: 0.986 | Sec 4.2
- Unique experts per layer at tree 255, OLMoE: EAGLE-3 54, MoE-Spec (B=32) 32 | baseline: 64 total | ratio: 54/64 = 84% | Fig 4c p6 (intro states '127-token tree activates 54 of 64'; Sec 4.3 says 54 at tree 255: internal inconsistency in the paper)
- Speedup at tree 255, OLMoE: MoE-Spec 2.1x vs EAGLE-3 1.85x (peak 1.95x at tree 31) | baseline: AR 1.0x | ratio: 1.14x | Fig 4b/4c p6
- Selector overhead: 2-3% | baseline: - | ratio: - | Sec 6 p8
- Quality vs speed, Qwen3-30B-A3B: ~95% of AR quality at 2.0x | baseline: AR | ratio: 0.95 | Sec 4.2 p5 / Fig 3
- Quality, Mixtral at max speedup: ~78% of AR | baseline: AR | ratio: 0.78 | Sec 4.2 p5
- EVICT: decoding speedup range and average over EAGLE-3: 1.08x-2.35x over AR; 1.21x avg over EAGLE-3 | baseline: AR / EAGLE-3 | ratio: - | Sec 4.2 p6 (Table 1)
- EVICT: reduction in active experts / verified tokens: -32.5% experts, -74.7% verified tokens, -26.6% verification latency | baseline: EAGLE-3 | ratio: 0.675 / 0.253 | Sec 4.3, Fig 5 p7
- SpecMoEOff: average speedup over best MoE offloading systems: 2.5x (2.1x avg over MoE-Lightning, 17.7x over DeepSpeed-ZeRO-Inference); SD gain 1.53x APPS, 1.45x CNN/DM vs its own no-SD ablation | baseline: MoE-Lightning / own w/o-SD | ratio: - | abstract, Sec 6 p11

**Catalog and first-pass numbers checked:**
- Deeper draft trees activate more unique experts, verify cost grows with draft size -> YES. MoE-Spec Fig 2/4; EVICT Fig 5
- EVICT 32.5% fewer unique activations -> YES. EVICT Sec 4.3: 'reduces the number of active experts by 32.5% on average' vs EAGLE-3 (Qwen3-30B-A3B etc., GPU)
- AcceptMoE figure (catalog text truncated at 'Acc') -> UNREAD / ABSENT. No paper called AcceptMoE found by search or in the downloaded PDFs; cannot be checked
- remedies cap experts per layer (MoE-Spec) -> YES. budget B per layer, lossy; EVICT instead truncates the draft prefix (lossless)
- first-pass: 10-30% higher throughput (MoE-Spec) -> YES. abstract: 10-30% over EAGLE-3
- first-pass: 127-token tree activates 54 of 64 experts -> YES (INTRO) BUT SEC 4.3 ATTRIBUTES 54 TO TREE 255. paper inconsistent
- first-pass: selector overhead 2-3% -> YES. Sec 6
- first-pass: EVICT/AcceptMoE/SpecMoEOff figures unverified -> NOW PARTLY RESOLVED. EVICT and SpecMoEOff read here; AcceptMoE still unverified

**Transfers to our regime:** partly. Direction transfers strongly (verify cost grows with the union of experts; this is our F.md pass-cost curve). Numbers do not: all are single-request or large-batch GPU runs with all experts in HBM (MoE-Spec, EVICT) or CPU-RAM-offloaded with large batches and Mixtral's 8 experts (SpecMoEOff), 1B-47B models, EAGLE-class trained drafts (not n-gram). MoE-Spec is LOSSY (drops tail experts; ~5% quality loss at 2x on Qwen3, 22% on Mixtral at max budget). EVICT is lossless and the more relevant remedy: for an n-gram draft the analogous rule is 'truncate the draft when the marginal expert-union cost exceeds the expected gain', which our measured pass-cost curve can feed. Cache size as share of experts (not modelled by any of them) breaks the transfer.

**Quality loss:** MoE-Spec: lossy, task metrics within one std of EAGLE at the chosen budget (Table 1), but ~95% (Qwen3) to ~78% (Mixtral) of AR quality at aggressive budgets; sampling at T=1 inherits EAGLE's q(x)=1 distribution shift (footnote 3). EVICT: lossless. SpecMoEOff: lossless verification.

**Ablation:** MoE-Spec: selection method (router vs oracle vs others), coverage policy, token policy (truncation vs substitution), tree size and budget sweeps (Sec 4.4-4.5). EVICT: adaptive prefix vs fixed. No ablation with an SSD tier or an LRU expert cache.

**Limitations stated:** MoE-Spec (Sec 6): budget hyperparameter per deployment; single-request A100 only; gains may shrink with batch; softmax routing only (DeepSeek-V3 sigmoid routing needs adaptation); no per-layer budgets. EVICT: SGLang eager mode shows slight regression from the extra runtime ops.

**Contradicts / supports our findings:** No. Consistent with F.md (0.40->1.79 s per verify pass at 8 GB). One refinement: MoE-Spec shows saturation at ~32 of 64 experts yields most of the speedup (a long tail of rarely-used experts), which suggests a cap on experts per verify pass is a cheaper lever than shortening drafts; we have not measured it on our traces.

**Unread parts:** Catalog URL (aggregator) not opened; AcceptMoE not found. SP-MoE / SpecMoE / MoESD only first pages read.

## P-0253

**Title/venue:** Faster In-Context Learning for LLMs via N-Gram Trie Speculative Decoding. Jinglin Chen, Qiwei Li, Zuchao Li, Baoyuan Qi, Guoming Liu, Haojun Ai, Hai Zhao, Ping Wang (Wuhan U, Xiaomi, SJTU). Proceedings of EMNLP 2025, pages 18040-18051.

**Read:** aclanthology.org/2025.emnlp-main.911.pdf, pdftotext of full PDF (69 kB text); the catalog URL (preview.aclanthology.org/setup/...) was replaced by the stable ACL URL (raw_text_read=True)

**Setup:** Vicuna-7B-v1.3, Llama-2-7B-Chat, Llama-3-8B-Instruct (+ Qwen2-7B for ablations); one NVIDIA RTX 4090, Python 3.9, PyTorch 2.5.1, CUDA 12.2, greedy; tasks Spec-Bench Summary (CNN/DM) and RAG (Natural Questions, 80 items, 5 Wikipedia passages), TriviaQA (own retrieval: bge-m3 + FAISS IVF-PQ + bge-reranker, top-5) and HAGRID. Baselines: vanilla, speculative sampling (Llama-68m / Vicuna-68m drafters), Medusa, SPACE, Hydra, Lookahead, REST. Method: trie of n-grams (length n 8-16, prefix Lp 2-5) built from the context, frequency-pruned min-heap of suffix chains, tree-attention verification of several drafts at once.

**Numbers (what / value / baseline / ratio / where):**
- Mean speedup, Vicuna-7B: 2.27x (Summary 1.75, RAG 3.48, TriviaQA 1.92, Hagrid 1.94); mean accept length RAG 5.19 | baseline: vanilla 1.00; Hydra 1.78x, SpS 1.61x, Medusa 1.49x, REST 1.27x, Lookahead 1.18x | ratio: 2.27/1.78 = 1.28x over best baseline | Table 1 p6
- Mean speedup, Llama-2-7B-Chat: 2.10x (Summary 1.28, RAG 3.62, TriviaQA 1.89, Hagrid 1.61) | baseline: Lookahead 1.44x, SpS 1.36x, REST 1.20x | ratio: 1.46x over Lookahead | Table 1
- Mean speedup, Llama-3-8B-Instruct: 1.56x (Summary 1.06, RAG 1.77, TriviaQA 1.75, Hagrid 1.68) | baseline: Lookahead 1.50x, REST 1.05x | ratio: 1.04x over Lookahead | Table 1
- Noisy vs clean retrieval (Qwen2-7B): RAG 1.22x noisy vs 1.87x clean; Hagrid/others in Table 4 | baseline: - | ratio: 0.65 | Table 4 p8
- num_draft sweep (Qwen2-7B): Summary 1.573x at 8 (1.480x at 128); RAG 1.906x at 8, 1.9195x at 32, 1.752x at 64 | baseline: - | ratio: - | Table 3 p8

**Catalog and first-pass numbers checked:**
- Context n-gram trie drafting on summarization, RAG, context QA -> YES. Spec-Bench Summary and RAG, TriviaQA, HAGRID
- first-pass: page gives no speedup figures -> N/A (FIRST PASS DIGEST-LIMITED). full text has them: 2.27x / 2.10x / 1.56x mean on the three models (Table 1)
- first-pass: Vicuna-7B, Llama2-7B-Chat, Llama3-8B -> YES. plus Qwen2-7B in ablations
- EMNLP 2025 pp.18040-18051 -> YES. page header

**Transfers to our regime:** no. Dense 7-8B models resident on an RTX 4090, batch 1, 80-item benchmarks with long retrieved contexts (high overlap by construction); the trie proposes MULTIPLE branch drafts verified with tree attention, which on a streamed MoE multiplies the expert union per verify pass (the MoE-Spec / Cascade effect). Overlap requirement same as P-0251; gain collapses with noisy retrieval (0.65x of clean) and on Llama-3 summarization (1.06x). Assumptions broken: GPU, dense weights, tree verification cost not modelled.

**Quality loss:** None claimed (speculative decoding, 'without compromising accuracy', abstract).

**Ablation:** Hyperparameters n and Lp (Table 2, token throughput), num_draft (Table 3), noisy vs clean retrieval (Table 4), trie build time (Table 6). No component ablation separating trie from tree attention.

**Limitations stated:** Sec 6: ties in frequency scores prune useful drafts under the fixed num_draft; strong dependence on quality of retrieved corpora (noisy retrieval hurts); n, Lp need tuning per model/dataset; timing noisy because GPU was shared (Appendix).

**Contradicts / supports our findings:** None.

# Per-family sections

## Family #20 (P-0244 to P-0247): provisional tokens / prefill-time reduction

### Further prior art found and read (this pass)
- Layered prefill (P-0247) is the closest published work to the layer-major prefill that follows from our B.md #9 expert-major idea; read in full. SeqMoE (arXiv 2609.12978) and MoE-Prefill (arXiv 2605.02960) surfaced by search: SeqMoE says chunked prefill causes repeated expert transfers under offloading and answers with offline frequency profiling plus prefetch; MoE-Prefill streams a few layers' expert shards from CPU pinned memory with a two-channel pipeline. Only search-result summaries seen, NOT read (UNREAD).
- Already in the first pass and not re-read here: ProMoE (2410.22134), Large LMs as Read/Write policy-makers for simultaneous generation (2501.00868), Voxtral Realtime resumable requests (2602.11298), CPU-GPU hybrid SLO paper (2606.10493). Remembered: LLMLingua (Jiang et al.), SnapKV, wait-k simultaneous translation (Ma et al. 2019), all from memory.
- No paper found on emitting provisional tokens from a partial prompt to hide prefill (searches: speculative prefill, chunked prefill MoE, prompt compression TTFT).

### What is new in OUR variant
Nothing in P-0244..P-0247 does what the idea does (emit tokens after the first 64-token block, reconcile later). SpecPrefill/Cross-Family (read in full) cut tokens before the big model and answer once; Sarathi-Serve (read in full) interleaves chunks and emits the first token only after the whole prompt; Layered prefill (read in full) changes loop order to avoid expert reloads and also emits after the whole prompt (its 'earlier tokens' in Fig 5 are queueing gains). Closest in spirit: the simultaneous-generation read/write policy paper (not read in full). The novelty of the early-emission mix stands as open, but F.md's finding (first block already costs 0.37-0.72 of the full prefill) and the fact that the published prune methods reach TTFT gains only where compute scales with tokens (SpecPrefill's 7.66x is 99% of its own FLOP bound 7.72x) say the payoff is small in our regime.

## Family #21 (P-0249 to P-0253): prompt-lookup / n-gram / suffix drafting

### Further prior art found and read (this pass)
- Cascade / "Utility-Driven Speculative Decoding for Mixture-of-Experts", Saxena, Tsai, Taneja, Jaleel, Qureshi (Georgia Tech, NVIDIA), arXiv 2506.20675v1 (17 Jun 2025), read in key parts. Static-K n-gram speculation on Mixtral 8x7B: verification time 2-3x higher, TPOT slowdown up to 1.5x (math 54% slowdown at K=3 while code +39%; mixed math+code only +13%), no K helps all tasks for any of five MoEs (Mixtral, DeepSeek-V1, Qwen-1.5, Phi-3.5, OLMoE); Cascade disables speculation when measured utility < 1 and limits slowdown to 5%, +7-14% over static K. GPU vLLM, FP8/FP16 models resident. This is the strongest independent prior evidence that n-gram drafting on MoE is a net loss on low-overlap tasks and that the right control is a utility-based on/off switch. It is also the paper P-0252's text cites as 'Saxena et al. (2025)'. Threatens novelty of a 'measure utility then back off' controller; supports our F.md loss at 8 GB.
- EVICT, arXiv 2605.00342 (read, see P-0252): lossless prefix selection by accepted/cost ratio; -32.5% active experts.
- SpecMoEOff, arXiv 2508.21706 (read in parts): speculative decoding to hide expert-offload latency on Mixtral with CPU RAM offload, large batches, 2.5x over MoE offloading SOTA. First speculative+offloading system; GPU+CPU-RAM, not SSD, not batch 1.
- SP-MoE arXiv 2510.10302 (abstract only): speculative decoding + speculative expert prefetching for offloaded MoE; SpecMoE arXiv 2604.10152 (abstract only): self-assisted speculation, up to 4.30x throughput on CPU-offloaded MoE at large batch; MoESD arXiv 2505.19645 (abstract only): SD benefits MoE more than dense at medium batch sizes (their finding contradicts the batch-1 picture; relevant if our regime ever batches). Not read beyond first page.
- EcoSpec 2607.12696, DraftExpert 2607.24434, Limits of Speculation 2609.22156: from the first pass, not re-read here.
- llama.cpp docs/speculative.md (read raw): ngram-cache, ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod exist; ngram-mod: 'MoEs require long drafts'.
- Remembered, not re-read: Lookahead decoding (Fu et al.), REST (He et al.), Token Recycling (Luo et al.), PIA.

### What is new in OUR variant
The mechanism (prompt lookup, suffix automaton, n-gram trie) is thoroughly published and shipped in llama.cpp. The regime variant (a streamed-expert CPU MoE with a tiny LRU, where a rejected draft costs a whole expert-union read) is addressed from the GPU side by Cascade (measured slowdown of n-gram on MoE), MoE-Spec and EVICT (cap experts / truncate drafts), and from the offloading side by SpecMoEOff and SP-MoE (large batch, CPU RAM). Nothing read measures n-gram drafting with experts streamed from NVMe at batch 1. Closest papers read in full: Cascade (2506.20675, key parts), MoE-Spec, EVICT.

## Findings that change what we should believe (summary)
1. P-0250: the catalog/first-pass dispute about 2.27x/2.49x is a VERSION issue: v1/v2 abstracts say 2.27x (SAM[T]) and 2.49x (SAM[E2]); v3 Table 1 says 2.27x and 2.58x. First pass wrongly said 2.27x was not found.
2. P-0246: the 2.8x/6.9x numbers (catalog, first pass) are v1; the published version says 3.7x (Yi-34B) and 5.6x (Falcon-180B).
3. P-0245: the paper is an ICLR 2026 conference paper per v3 header, not a workshop; the 55.98/57.34 row is Qwen3-8B target with Llama-3.2-1B draft.
4. P-0247: 'TTFT -70% (2.80 to 1.24 s)' is a conflation; Table 6 is -56%. The 39% is layered-vs-chunked (chunked loads are 1.64x layered); our 4.4-6.0x re-read figure is a different quantity (prefill-only, 64-token blocks).
5. P-0249: the slowdowns in the PR thread come from the first, disputed implementation, not from the merged code; they are weak evidence. The independent strong evidence of n-gram loss on MoE is Cascade (up to 1.5x slowdown on Mixtral).
6. P-0244: the 7.66x is 99% of the paper's own FLOP bound (7.72x): a pure compute-scaling result, and the 8B speculator would cost more FLOPs per token than our 125B MoE's active path (16 vs ~12 GFLOP). This reinforces F.md's bytes-saturate bound (1.5-2.8x) as the plausible ceiling for prompt pruning on our setup.
7. Nothing read contradicts our own measurements in F.md (prefill coverage, verify-pass cost, f_accepted, 8 GB drafting loss). Mild caution: our prose/code stand-in corpora give f=0.11-0.16; the P-0251 data imply 0.4-1.0 accepted drafts per step on ShareGPT/GSM8K-like outputs, so real chat overlap may be higher than our stand-ins (unmeasured, no saved model outputs).
