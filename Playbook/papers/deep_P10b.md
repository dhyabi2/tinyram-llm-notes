# deep_P10b: raw-text re-check of rows P-0222..P-0231 (9 rows; P-0226 is another batch's idea row)

Read 2026-10-29 from raw text (pdftotext -layout of arXiv/ACL PDFs, stripped forum HTML, raw README). Scratch: scratchpad/deep_P10b. Counts: raw text read 9 / unread 0 (P-0225 is our own local file; P-0224 and P-0231 are non-paper sources). Remembered facts are labelled "from memory"; none used for numbers.

Our regime for all 'transfers' verdicts: CPU-only, 8/16/32 GB RAM, expert weights streamed from NVMe, batch-1 decode, 125B-class 512-expert top-10 MoE.

## Corrections to the first pass (P10.md / P10.json)
1. P-0222 HookMoE: not 'prefill only'; paper also reports decode ITL speedup 1.09-1.13x (Sec 4.4). TTFT range is 1.24-1.46x. The 'four models' are named (Mixtral-8x7B, 8x22B, Qwen1.5-MoE-A2.7B, DeepSeek-V2-Lite). '2.5% degradation' is 2.50 absolute points on Mixtral-8x7B only (four-model mean 1.97). Domain tasks (GSM8K+Minerva) lose 6.4 points even with MATH-calibrated Hook; k reductions are 2->1, 4->1, 6->3 only. Marked 'abstract only' in the first pass; now full text.
2. P-0223 MC-MoE: accuracy and speed ARE in text tables (Table 4: 71.29 -> 66.94 at 2.54-bit with ODP, speed 1.71x; 2.05-bit 62.68, 1.80x; 1.57-bit 53.77, 1.89x). '>10% extra compression' is ~15% fewer activated parameters (12.6% at 2.54-bit) via top-2->1 for ~15% of tokens. Memory 16.24 GB static, 20.7 GB peak. Speedups are vs FP16 on 2x A100, PMQ alone gives 1.63x. Catalog framing 'top-k reduction plus low-bit' is only the dynamic 2->1 skip.
3. P-0224 Qwen3.5-122B post: '70+ vs 47-50 tok/s' is patched vs stock vLLM for the same top-4 model (70.8 vs 49.0, model card), with MTP depth-3 speculation; the post has NO top-8 vs top-4 speed number. Retraining covers only the shared expert (BF16) and MTP head; quality claim is vs other INT4 builds on a private benchmark, not BF16.
4. P-0228 Sub-MoE: the 1.38x row (120.9 tok/s) has average accuracy 0.45 vs 0.67 (first pass omitted). Retention numbers 96/86 are in the abstract (confirmed).
5. P-0230 RFID-MoE: catalog/first-pass note says 'wrong in part' -- confirmed, and the paper's rank floor is max(1, .) per group. Version read is v2 (11 Feb 2026), not v1. Table 2 shows anomalies: MoBE and RFID-MoE score above the original model at 20-40% on Qwen3-30B, D2-MoE rows copy the D2-MoE paper's numbers.
6. P-0231 REAL: the 89.08 in the first-pass headline is the strict GSM8K of rank-256/50%; the collapse of MMLU at 50% holds for all SVD variants on Qwen (and the mean-baseline variant collapses already at 25%); Qwen 25%/rank128 plain SVD keeps MMLU 80.12 (-3.6). No paper exists; repo created 2026-07-07.
7. P-0227 D2-MoE: first-pass numbers all confirmed; add that Table 4 has no uncompressed baseline and PPL at 60% is 6.35 (Table 4) vs 6.46 (Table 2). P-0229 MoBE: all confirmed; add Appendix B (effective rank above break-even).
8. P-0225: R16's '1.3x at 8 GB' equals the ideal-overlap bound; E.md later measured 1.61-2.4x serial.

## Results that should change what we believe about our own measurements
A. Top-k quality gate (family 16): our +45.6% OLMoE perplexity at top-4 is from a router that does not renormalise and has no shared expert. arXiv 2609.04575 shows on Qwen3.5-397B-A17B (512 experts, top-10, same shape class as ours) that 10->5 costs 2.10 MMLU points with standard renormalisation and 0.55 with a decoupled reference mass (GSM8K -4.2 / -0.8); on Qwen3.6-35B 8->4 costs 4.65 / 0.35. Dropping renormalisation entirely costs 27 MMLU points. So the 'accuracy-gated' verdict for the target model is not established by OLMoE; the quality side must be re-measured on a renormalising (and shared-expert) architecture, and k2 handling must be copied if llama.cpp's override is used. The speed side is unchanged (1.3x ceiling; published GPU decode gains 1.09-1.32x are consistent).
B. Low-rank premise (family 17): MoBE Appendix B measures the 95%-energy effective rank of real raw expert matrices (Kimi-K2 7168x2048) as above the break-even rank 1593 in most layers; REAL shows in-place rank-256 SVD of half the experts of Qwen3.6-35B destroys MMLU. Our unmeasured 'residual energy <=10% at r=64' premise is therefore unsupported by all read evidence (the delta-from-shared-base case, D2-MoE Fig 2, is only shown as a figure).
C. RFID-MoE does not give zero rank to never-routed experts, so our E.md remark that its zero-rank idea 'only applies to long-cold experts' attributes to the paper something it does not do.
D. No contradiction found for our ceiling lifts (1.35x top-4) or 1.58x low-rank closed form; published speedups (HookMoE decode 1.09-1.13x, DCI 1.27x at k=4, MoE-SVD 1.5x/1.8x at 60%) are all GPU.


---

## P-0222: HookMoE: A learnable performance compensation strategy of Mixture-of-Experts for LLM inference acceleration. Longkai Cheng*, Along He*, Mulin Li, Xueshuo Xie, Tao Li (Nankai Univ., Haihe Lab ITAI, Shenzhen Univ.). EMNLP 2025 main, pp 31594-31606, 4-9 Nov 2025 (ACL Anthology 2025.emnlp-main.1610; PDF text read, 899 lines)

- ID/version read: aclanthology.org/2025.emnlp-main.1610.pdf (no arXiv id found/used); raw text read: True
- Setup: Models (Table 1/3): Mixtral-8x7B (2/8, 46.7B total), Mixtral-8x22B (2/8, 140.6B), Qwen1.5-MoE-A2.7B (14.3B, 64 experts, 4 shared; Table 1 says 8/64 but Table 3 'Origin' is k=4 -- internal inconsistency), DeepSeek-V2-Lite (6 of 64 routed + 2 shared). Method: lower top-k (Mixtral 2->1, Qwen 4->1, DeepSeek 6->3) plus one trainable 2-linear 'Hook' module (0.14-0.70% of params) before one selected layer, trained on 512 calibration samples (C4 or MATH, 2048 tokens), 1000 'epochs' lr 1e-4, 4 h on 8 H800. Speed: HF transformers, bf16, 2x A800-80GB (NVLink 400 GB/s) or 2x H20-96GB, Mixtral-8x7B only; TTFT at 1k/2k/4k input tokens (1 output token), ITL with 256 new tokens, 100 repeats. Batch size not stated. No CPU, no offload, no llama.cpp/vLLM.
- Numbers (what | value | baseline | ratio | where):
  - Mixtral-8x7B avg of 8 zero-shot tasks, Table 3 p.5 | Hook k=1: 65.30 (+2.43 over plain k=1 62.87) | original k=2: 67.80 | -2.50 points absolute = -3.7% relative; active params 7.36B vs 12.90B = 0.57x | Table 3, p.5
  - Mixtral-8x22B avg | Hook k=1: 68.54 (+3.01); plain k=1 65.53 | k=2: 70.22 | -1.68 points (-2.4% rel) | Table 3
  - Qwen1.5-MoE avg | Hook k=1: 64.02 (+0.31); plain k=1 63.71 | k=4: 65.89 | -1.87 points | Table 3
  - DeepSeek-V2-Lite avg | Hook k=3: 58.50 (+0.32); plain k=3 58.18 | k=6: 60.31 | -1.81 points | Table 3
  - Four-model mean drop (my arithmetic) | (2.50+1.68+1.87+1.81)/4 = 1.97 points | originals | abstract's 'average 2.5%' equals the Mixtral-8x7B number only | Table 3
  - Domain-specific (GSM8K+Minerva avg), Mixtral-8x7B | Hook trained on MATH 41.91; on C4 40.16; plain top-1 41.43 | k=2: 48.34 | -6.4 points even with math calibration (Hook adds only +0.5 over plain top-1); C4-calibrated Hook is BELOW plain top-1 | Table 2, p.5
  - TTFT speedup (prefill) | 1.24x-1.46x (A800 and H20; 1k/2k/4k) | Mixtral k=2, HF transformers | as reported; read from Figure 3 (axis values garbled in text extraction) | Sec 4.4 text p.7, Fig 3
  - ITL speedup (decode) | 1.09x-1.13x | Mixtral k=2 | as reported | Sec 4.4 text p.7, Fig 3
  - Dynamic-k comparison Mixtral-8x7B | Hook+TLNLE Act 1.06, avg 65.52; MoED p=0.52 Act 1.07, 64.88; D2D tau 0.9 Act 1.09, 64.96 | NAEE Act 1.50, 67.32; origin 67.80 | Hook+TLNLE beats best dynamic-k at same Act by 0.56-0.64 points | Table 4, p.6
  - Layer-depth observation (TLNLE) | top-1 in layers 16-31 of Mixtral, top-2 in 0-15: -1.1% performance | all top-2 | -1.1% | Sec 3.2, Fig 1
- Catalog / first-pass claims:
  - over 50% fewer active experts -> partly: Mixtral/22B: 2->1 (50%; Act 1.06 means 47% remain); Qwen 4->1 (75%); DeepSeek 6->3 (50%). Only 2-8 expert-per-token baselines; never 10->4.
  - with a calibration step -> yes: 512 samples, trains 0.14-0.70% of params, ~4 h on 8 H800
  - 1.42x prefill speedup -> partly: TTFT range is 1.24-1.46x; 1.42 is the abstract value. Paper ALSO reports decode ITL 1.09-1.13x (first pass and catalog implied prefill only).
  - about 2.5% degradation -> partly: 2.50 absolute points (3.7% relative) on Mixtral-8x7B only; four-model mean 1.97 points; domain/math tasks lose 6.4 points
  - first-pass: 'four models, names not on page' -> no: Names are in Table 1/3: Mixtral-8x7B, Mixtral-8x22B, Qwen1.5-MoE-A2.7B, DeepSeek-V2-Lite
  - first-pass: prefill-only -> no: ITL 1.09-1.13x reported in Sec 4.4; the abstract only advertises prefill
- Transfers to our regime: no. Why: GPU, HF transformers eager, compute-bound regime; k only 2->1 (or 4->1, 6->3), never a 512-expert top-10 model; needs training of a Hook module; the speedup mechanism (less expert FLOPs on GPU, CPU launch overhead stated up to 50% of time) is absent in an NVMe-bound CPU decode where cost is bytes read per token (that part would transfer in principle: fewer experts = fewer bytes). Decode gain only 1.09-1.13x even on GPU. Math/domain quality loss 6.4 points.
- Quality loss: Table 3: -1.8 to -2.5 points avg on 8 zero-shot tasks (after Hook); Table 2: -6.4 points on GSM8K/Minerva (Mixtral, MATH-calibrated Hook); plain top-1 without Hook -4.9 (general) / -6.9 (domain).
- Ablation: Table 2 calibration set (C4 vs MATH); Table 4 dynamic-k routing with/without TLNLE; Fig 1 layer-range sweep. No ablation of speed vs k on a memory-bound system.
- Limitations (authors): Authors: compensation shrinks as k approaches original; models over 141B not evaluated (DeepSeek-V3, Grok-1, Qwen3). Speedup measured only in HF transformers; authors expect more with vLLM/SGLang.
- Contradicts / bears on our findings: No contradiction; consistent with our 1.3x ceiling lift (their 1.09-1.13x decode at k 2->1 on GPU). Their Table 4 shows top-1-in-upper-layers costs only 1.1%, i.e. per-layer k allocation (not uniform k) is a cheaper way to cut k than our uniform 10->4 gate test.

---

## P-0223: Mixture Compressor for Mixture-of-Experts LLMs Gains More (MC / MC-MoE). Wei Huang*, Yue Liao*, Jianhui Liu, Ruifei He, Haoru Tan, Shiming Zhang, Hongsheng Li, Si Liu, Xiaojuan Qi (HKU, CUHK, Beihang). ICLR 2025 (v2, 22 Feb 2025)

- ID/version read: arXiv 2410.06270v2 (PDF text, 1277 lines); raw text read: True
- Setup: Mixtral-8x7B (49B total, 13B active, 32 blocks, 8 experts, top-2) and Mixtral-8x22B (141B/39B, 56 blocks). PMQ: mixed-precision GPTQ, per-expert bit-width 1-3 bits chosen by LP using activation frequency, routing weight and quantisation loss; calibrated on 128x2048 C4; non-MoE layers 4-bit. ODP: online dynamic pruning, expert 2 skipped for a token if w1/w0 < median threshold, protecting ~2% most important tokens (attention + L1-norm score). Compression on 2-4 A100-80GB. Speed (Table 4, 13, 14): 16-bit baseline on 2x A100 (8x7B) / 4x A100 (8x22B); quantised models on 1x A100 or 1x RTX 3090, HQQ dequant kernels. Table 13 batches 1/8/16, input 512-4096.
- Numbers (what | value | baseline | ratio | where):
  - Mixtral-8x7B 2.54-bit PMQ only: LM-Eval avg / activated params / speedup | 67.50 / 4.53 GB / 1.63x | 16-bit: 71.29 / 26.31 GB / 1.00x (2x A100) | -3.79 points; activated bytes 0.17x; speed 1.63x (baseline uses 2 GPUs) | Table 4, p.9
  - 2.54-bit + ODP | 66.94 / 3.96 GB / 1.71x | PMQ only 67.50 / 4.53 / 1.63x | ODP adds -0.56 points, -12.6% activated bytes, +0.08x speed | Table 4
  - 2.05-bit + ODP | 62.68 / 3.23 GB / 1.80x | 16-bit 71.29 | -8.61 points | Table 4
  - 1.57-bit + ODP | 53.77 / 2.55 GB / 1.89x | 71.29 | -17.5 points | Table 4
  - Uniform 2-bit (no PMQ, no ODP) | 42.67 / 3.70 GB / 1.72x | 71.29 | -28.6 points; PMQ's mixed allocation is what keeps quality | Table 4
  - Mixtral-8x22B 2.54-bit + ODP | 71.21 / 10.96 GB / 1.82x | 76.33 / 76.49 GB / 1.00x (4x A100) | -5.12 points | Table 4
  - Decode latency per token, batch 1, 8x7B | MC 2.54-bit 1xA100: 0.015 s ([1,512]) ... 0.025 s ([1,4096]) | FP16 2xA100: 0.029 s ... 0.057 s | 1.93x at 512, 2.28x at 4096 (paper prints 48.3%/56.1% reduction) | Table 13, App A.9 p.19
  - Throughput tok/s | MC 2.54-bit: 38 (1xA100), 52 (1xRTX3090) | FP16 Mixtral 2xA100: 23; Llama2-13B 1xA100: 46 | 1.65x vs FP16 (A100) | Table 14, p.20
  - ODP dynamic pruning amount | ~15% of tokens drop from top-2 to top-1 (14.88% with protection, PPL 6.22) | manual thresholds 0.4-0.7: 12-22% pruned, PPL 6.29-6.89 | n/a | Table 12, App A.8
- Catalog / first-pass claims:
  - catalog: 'top-k reduction plus low-bit combination already published, with code' -> partly: Dynamic top-2->top-1 on ~15% of tokens (not a static 10->4 reduction) plus 1.57-2.54 bit mixed quant; code at github Aaronhuang-778/MC-MoE (stated in abstract).
  - first-pass: dynamic pruning gives >10% extra compression -> partly: ODP reduces activated parameters by about 15% (Table 4: 4.53->3.96 GB at 2.54-bit = 12.6%; 3.73->3.23 = 13.4%), with <1 point accuracy drop; it is activated-param reduction, not stored-size compression.
  - first-pass: 2.5-bit 8x7B ~16 GB static / 19 GB running -> partly: Paper: 16.24 GB (Table 4; text says 16.65) static, peak GPU memory 20.7 GB (Table 14).
  - first-pass: speedup and accuracy only in an unreadable image -> no: Both are in Tables 2, 4, 13, 14 as text: 1.63-1.89x vs FP16 2xA100; accuracy table above.
  - first-pass: Mixtral 8x7B/8x22B only -> yes: Only these two
- Transfers to our regime: no. Why: 8-expert top-2 models on GPUs; the 1.6-1.9x speedup is dominated by quantisation vs a 16-bit 2-GPU baseline (PMQ alone 1.63x), ODP adds 0.08x; ODP prunes at most 1 of 2 experts for 15% of tokens, nothing like 10->4 of 512 experts; our baseline is already ~4.3-bit so the 16-bit comparison inflates the gain; the paper's GPTQ+HQQ kernels are CUDA. The mixed-precision-per-expert idea (more bits for frequently used experts) does transfer as a design, but its measured effect is on perplexity, not on NVMe bytes.
- Quality loss: Table 4 LM-Eval avg (8 zero-shot): 8x7B 71.29 -> 67.50 (2.54-bit) -> 66.94 (+ODP) -> 62.68 (2.05-bit) -> 53.77 (1.57-bit); 8x22B 76.33 -> 71.21 (2.54+ODP). Uniform 2-bit 42.67.
- Ablation: Yes: PMQ vs uniform bits (Fig 5/6, Table 3), token protection (Fig 7/8), ODP threshold (Table 12), token importance metrics (Table 11). None isolates NVMe/offload cost.
- Limitations (authors): Not stated in a limitations section; compression experiments need 2-4 A100; training-free but calibration-dependent; only Mixtral.
- Contradicts / bears on our findings: No contradiction. Supports: at 2.05-bit quality drops 8.6 points, i.e. our '2-bit adds at most 0.06 over top-4 at 16/32 GB' speed result must be read together with a large quality cost (our OLMoE Q2_K +26% PPL is consistent in direction).

---

## P-0224: 'A faster (maybe better?) Qwen 3.5 122B', forum post by azampatti, NVIDIA Developer Forums, DGX Spark / GB10 category, 29 Aug 2026 7:33pm, with replies to 31 Aug; model card azampatti/Qwen3.5-122B-A7B-Int4 on Hugging Face (README read raw)

- ID/version read: forums.developer.nvidia.com/t/a-faster-maybe-better-qwen-3-5-122b/381813 (HTML stripped, 10.6k chars); huggingface.co/azampatti/Qwen3.5-122B-A7B-Int4 README (6.9k chars); raw text read: True
- Setup: Base Qwen3.5-122B-A10B: 48 layers, 256 routed experts, top-8, 1 shared expert, hidden 3072. Variant: router top-8 -> top-4 at inference (all 256 experts kept), shared expert retrained in BF16 by distillation against the unpruned base (rest frozen), auto-round INT4 g128 for everything else, MTP draft head retrained (KL distillation, depth 3). ~122B total, ~6.4B active/token incl. embeddings (~4.9B in layers), ~65 GB on disk. Hardware: single DGX Spark (GB10, SM121, 128 GB unified memory), patched vLLM 0.26.1rc1 (eugr/spark-vllm-docker derived), fp8 KV cache, single stream.
- Numbers (what | value | baseline | ratio | where):
  - Throughput, same model, same vLLM, patches off vs on | 70.8 tok/s patched | 49.0 tok/s stock | 1.445x -- this ratio is the effect of the hybrid-quant patch (BF16 shared expert off the fused path), NOT of top-4 | HF model card 'Read this first' table
  - bench.sh single-stream sweep, patched | Q&A 61.0, Code 69.1, JSON 64.0, Math 59.8, LongCode 70.8; mean 64.9 tok/s | none (no top-8 run of the same checkpoint) | no k=8 vs k=4 speed ratio is reported anywhere | HF model card 'Speed' table
  - Custom benchmark (10 rounds, trimmed mean) | TopK4-Int4 50.7% | Albond hybrid Int4FP8 51%; standard INT4 AutoRound 47% | -0.3 vs 51; +3.7 vs 47 (benchmark itself undisclosed) | forum post 1
  - tool-eval-bench | 89-91 /100 | 'on par with base FP8 or INT4' (no numbers given) | n/a | forum post 1
  - MTP head retraining | 'barely improve things by about ~2%' acceptance | n/a | n/a | forum post 1
- Catalog / first-pass claims:
  - same 8 to 4 trick on a 122B model -> yes: router top-8 -> top-4 of 256 experts on Qwen3.5-122B-A10B
  - with further training to recover quality -> partly: Only the shared expert (BF16) and the MTP draft head are retrained (distillation); routed experts and router untouched; no perplexity/benchmark table for BF16 base, quality claim 'matches base' is by a private custom benchmark compared with other INT4 builds (not BF16).
  - first-pass: 70+ vs 47-50 tok/s -> partly: Numbers exist (70.8 vs 49.0) but the comparison is patched vs stock vLLM for the SAME top-4 model; the post gives no top-8 vs top-4 speed. Speculative decoding (MTP depth 3) is included in 70.8.
  - first-pass: custom benchmark 50.7% vs 51%/47% -> yes: as quoted
  - first-pass: model has 'top-4 of 256' -> yes: plus 1 shared expert
- Transfers to our regime: partly. Why: Same model class (fine-grained MoE, 256 experts, top-8, 122B) as ours, and the only public case of halving k on a ~120B model with a healing step. But: GPU/unified-memory device with everything resident (no streaming), self-reported, no disclosed benchmark, no BF16 baseline, speed numbers confound k, patch, INT4 and MTP. The finding that matters for us is qualitative: top-4 of 8 is recoverable (per the author) via the shared expert, which is also the mechanism in the renormalisation paper below.
- Quality loss: Self-reported: 50.7% vs 51%/47% on a private 'very custom' benchmark, tool-call 89-91/100; no standard benchmark, no delta vs BF16.
- Ablation: None for k; author mentions an earlier faster method that could not be recovered (unspecified).
- Limitations (authors): Author: only tested on GB10; not peer reviewed; 'vibe-coded'; sample is 10 rounds trimmed mean.
- Contradicts / bears on our findings: Mild: our gate (OLMoE +45.6% PPL at top-4, OLMoE has no renormalisation and no shared expert) is a worse-case; see Training-Free Halving (2609.04575) below: for models with renormalised top-k and a shared expert, top-4 of 8 costs 0.35-4.65 MMLU points depending on the reference mass. Our top-k quality gate should be re-measured on a renormalising model.

---

## P-0225: Our own first screen R16 (Runs/R16 Screen of the top 22 brainstorm idea families.md), plus Playbook/pubtech/E.md family 16 results

- ID/version read: local files (R16 line 10 summary and E.md lines 190-222 read; not a paper); raw text read: True
- Setup: Internal: trace-driven simulator + closed-form ceiling model for 125B-class 512-expert top-10 MoE, 8/16/32 GB, NVMe streaming (E.md family 16).
- Numbers (what | value | baseline | ratio | where):
  - compute ceiling k=10/8/6/4/2 | 4.58/5.02/5.54/6.19/7.02 tok/s | k=10 4.58 | top-4 lift 1.35x | E.md family 16 table
  - R16 summary | top-k 10->6 or 4: ceiling 4.57 -> ~5.4-6 tok/s, 1.3x at 8 GB, accuracy-gated | 4.57 | 1.18-1.31x | R16 line 10
  - trace-driven serial gain 8 GB, 4-of-8 experts, 4.3 bit / 2.5 bit | 1.61x / 2.10x | full k | as listed | E.md
  - prior OLMoE PPL top-6 / top-4 / Q2_K / top-4+2bit | +8.8% / +45.6% / +26% / +83.5% | OLMoE full | gate 5% | E.md (earlier runs, not repeated)
- Catalog / first-pass claims:
  - ceiling 4.57 to 5.4-6 tok/s -> yes: R16 text matches; E.md gives 4.58 -> 5.54 (k=6) -> 6.19 (k=4)
  - 1.3x at 8 GB -> partly: R16 says 1.3x at 8 GB; E.md later measured 1.61-2.4x serial at 8 GB and 1.28-1.37x at ideal overlap; the R16 1.3x equals the overlap bound
  - accuracy-gated -> yes: OLMoE top-4 PPL +45.6% fails the 5% gate
- Transfers to our regime: n/a. Why: Internal row. Relevance of the new evidence: our quality number comes from OLMoE (norm_topk_prob not applied, no shared expert); the 512-expert top-10 Qwen3.5-397B result in 2609.04575 shows the loss from k reduction depends strongly on renormalisation.
- Quality loss: OLMoE PPL +8.8% (top-6), +45.6% (top-4).
- Ablation: E.md family 16 (ceiling, trace-driven serial/overlap).
- Limitations (authors): Decisive model test (PPL and decode speed with the k override on a real model) never run; OLMoE Q4_K_M not on disk.
- Contradicts / bears on our findings: Raises doubt on the quality side (not the speed side): published training-free evidence on exactly 512-expert top-10 (Qwen3.5-397B) says 10->5 costs 2.10 MMLU points with standard renormalisation and 0.55 with a decoupled reference mass, far below our OLMoE-derived 'top-4 = +45.6% PPL'. Also LExI/HookMoE show uniform k is the worst allocation (top layers need fewer experts).

---

## P-0227: Delta Decompression for MoE-based LLMs Compression (D2-MoE). Hao Gu*, Wei Li*, Lujun Li*, Qiyuan Zhu, Mark Lee, Shengjie Sun, Wei Xue, Yike Guo (HKUST, Birmingham, AISpeech). ICML 2025

- ID/version read: arXiv 2502.17298v1 (24 Feb 2025), PDF text, 932 lines; raw text read: True
- Setup: Models: Mixtral-8x7B, DeepSeekMoE-16B-Base (64 experts), Phi-3.5-MoE (16 experts), Qwen2-57B-A14B (64 experts). Method: Fisher-weighted merge of experts into one shared base Wb per layer, delta_i = W_i - Wb compressed by truncation-aware SVD, base pruned by semi-dynamic structured pruning (half static, half per input batch). Training-free, 512 WikiText-2 calibration samples, A100 GPUs. Quality: 3 PPL sets + 7 zero-shot tasks. Speed: Mixtral-8x7B, batch size 64 (Table 4), baselines NAEE, MoE-I2, LoSparse; no FP16 uncompressed row in Table 4.
- Numbers (what | value | baseline | ratio | where):
  - Mixtral-8x7B average accuracy (7 tasks) | 20%: 0.60; 40%: 0.57; 60%: 0.52 | original 0.63 (NAEE 0.58/0.48/0.36; MoE-I2 0.57/0.49/0.36) | -0.03 / -0.06 / -0.11 (95% / 90% / 83% retention); WikiText-2 PPL 3.98 -> 4.65 / 5.28 / 6.46 | Table 2, p.6
  - DeepSeekMoE-16B average accuracy | 0.54 / 0.49 / 0.41 | 0.56 original | 96% / 88% / 73% retention; PPL 6.38 -> 6.84/7.93/11.67 | Table 2
  - Phi-3.5-MoE 40% | 0.60 | 0.62 orig; NAEE 0.57 | 97% | Table 2
  - Qwen2-57B-A14B 40% | 0.58 | 0.59 orig | 98% | Table 2
  - Throughput, Mixtral-8x7B, batch 64, 60% / 70% / 80% compression | 277.72 / 300.33 / 313.29 tok/s (481/386/290 TFLOPs) | NAEE 271.89 / 272.66 / 278.53; MoE-I2 227.60/252.55/294.04; LoSparse 158.90/191.45/198.04 | vs NAEE +2.1% / +10.2% / +12.5%; no uncompressed baseline given | Table 4, p.7; Appendix A.3
  - Memory | 34.8 / 26.1 / 17.3 GB at 18.68B / 14.01B / 9.33B params | Mixtral 46.7B x 2 B = ~93 GB fp16 | compression is vs 16-bit weights (1.86 GB per B params) | Table 4
  - PPL at 60/70/80% (speed rows) | 6.35 / 8.15 / 12.95 | Table 2 gives 6.46 at 60% | internal inconsistency 6.35 vs 6.46 | Table 4 caption vs Table 2
  - Component ablations, 40% Mixtral | Fisher merge PPL 5.28 vs mean 7.66, frequency 6.42, TIES 12.45, RegMean 187 | n/a | Fisher merge is the key component | Table 5
- Catalog / first-pass claims:
  - 40 to 60% compression on Mixtral, Phi-3.5, DeepSeek, Qwen2 MoE -> yes: 20-60% on Mixtral/DeepSeek, 40% on Phi/Qwen2 in Table 2; up to 80% in Table 4 (speed)
  - over 13% better than other compressors -> yes: abstract: 'over 13% performance gains' vs baselines at 40-60% (e.g. Mixtral 60%: 0.52 vs 0.36 = +16 points)
  - first-pass: avg 0.63 -> 0.60/0.57/0.52 -> yes: as Table 2
  - first-pass: 313.29 tok/s at 80% vs NAEE 278.53 -> yes: Table 4
  - first-pass: memory 34.8/26.1/17.3 GB at 60/70/80% -> yes: Table 4
  - first-pass: batch 64, A100 -> yes: Table 4 caption; 'A100 GPUs' Sec 4
  - first-pass: speed gain over other compressors only 2-12% -> yes: +2.1% (60%), +12.5% (80%) vs NAEE; vs LoSparse far more
- Transfers to our regime: no. Why: GPU batch 64 (compute-bound, the opposite of our batch-1 byte-bound regime); the speed table has no uncompressed baseline; compression ratios are against 16-bit weights while ours starts from ~4.3-bit Q4_K, so a 40-60% reduction of fp16 bytes is a ~2.3-3.1 bit-equivalent already reached by plain quantisation; quality loss is already -0.06 (40%) / -0.11 (60%) points avg; the base is a layer-wide Fisher merge, so every token reads the shared base (not per-expert delta only) which is compatible with 'read base once per layer' but the paper does not build that kernel. Decomposition needs backward passes (Fisher) so a 125B model would be costly.
- Quality loss: Table 2: Mixtral avg 0.63 -> 0.60 / 0.57 / 0.52 at 20/40/60%; DeepSeekMoE 0.56 -> 0.54/0.49/0.41; WikiText PPL 3.98 -> 4.65/5.28/6.46.
- Ablation: Yes: merge methods (Table 5), compressor choices for base and delta (Table 6: truncation-aware SVD on delta 5.28 vs vanilla SVD 6.22, activation-aware SVD 5.91, pruning 5.74), calibration size/source (Tables 8-9), delta trimming (Fig 4), ShareV/MergeU (Table 11). Nothing about a streamed-from-disk or CPU path.
- Limitations (authors): Authors: 'involves decomposition and pruning steps with some complexity'. No other limitation stated.
- Contradicts / bears on our findings: Consistent with our closed form (about 50% size at rank ~64; ~1.5x). Their Fig 2 claims delta weights have much higher singular-value energy retention than raw experts, i.e. residual energy of the delta is the quantity we left unmeasured; paper shows it in a figure only (no 10%-at-r=64 number). MoBE (below) argues the effective rank of real experts (95% energy) exceeds break-even rank, which is evidence against our 10%-residual-at-r=64 premise.

---

## P-0228: Sub-MoE: Efficient Mixture-of-Expert LLMs Compression via Subspace Expert Merging. Lujun Li, Qiyuan Zhu, Jiacheng Wang, Wei Li, Hao Gu, Sirui Han, Yike Guo (HKUST, XJTU, Birmingham). arXiv 29 Jun 2025 (also AAAI per search result)

- ID/version read: arXiv 2506.23266v1 (PDF text, 738 lines); raw text read: True
- Setup: Mixtral-8x7B, Qwen1.5-MoE-A2.7B-Chat (60 experts), Qwen3-30B-A3B (128 experts; text lists Qwen3-235B-A22B but tables use 30B), DeepSeek-MoE-16B. Method: K-means clustering of experts by output cosine, joint SVD on concatenated experts in a group giving a shared U, frequency-weighted merge of V, optional intra-expert compression (dagger variant). Calibration 128x2048 WikiText-2. Eight H800 GPUs; Table 2 runtime is tokens/sec on 8x H800 (batch and sequence length not stated).
- Numbers (what | value | baseline | ratio | where):
  - Mixtral 8x7B -> 6 experts, avg 8 zero-shot | Sub-MoE 0.64 | original 0.67; HC-SMoE 0.61; MC-SMoE 0.44 | 0.955 retention (25% expert reduction) | Table 1, p.7
  - Mixtral -> 4 experts | Sub-MoE 0.58 | 0.67; HC-SMoE 0.51 | 0.866 retention (50% reduction); WikiText PPL 3.98 -> 6.97 | Table 1
  - Qwen3-30B-A3B 128->96 / 128->64 experts | 0.60 / 0.57 | 0.69 | 0.87 / 0.83; WikiText PPL 8.64 -> 13.59 / 21.05 | Table 1
  - Qwen1.5-MoE 60->45 / 60->30 | 0.58 / 0.46 | 0.60 | 0.97 / 0.77 | Table 1
  - DeepSeek-MoE-16B 64->48 / 64->32 | 0.55 / 0.49 | 0.57 | 0.96 / 0.86 | Table 1
  - Runtime, Mixtral 8x7B original | 87.7 tok/s | n/a | 1.00 | Table 2, p.8
  - 6x7B + intra compression 10/20/30% | 93.1 / 104.7 / 120.9 tok/s; avg acc 0.58 / 0.52 / 0.45 | 87.7 tok/s; acc 0.67 | 1.06x / 1.19x / 1.38x; accuracy ratio 0.87 / 0.78 / 0.67 | Table 2
  - 4x7B + intra 10/20/30% | 95.3 / 108.2 / 122.7 tok/s; acc 0.53 / 0.48 / 0.43 | 87.7; 0.67 | 1.09x / 1.23x / 1.40x; acc 0.79/0.72/0.64 | Table 2
  - Memory | 8->6 experts -24% memory, 6% accuracy drop; 8->4 -48% memory, 13% accuracy decline | 8 experts | as authors | Sec 4.4 text, Fig 4 right
- Catalog / first-pass claims:
  - Keeps 96% / 86% of Mixtral-8x7B performance at 25% / 50% expert reduction -> yes: abstract says 96|86; Table 1: 0.64/0.67 = 95.5%, 0.58/0.67 = 86.6%
  - 'so a quality cost shows up by 50%' -> yes: -13% at 50%; Qwen3-30B -17% at 50%; Qwen1.5-MoE -23% at 50%
  - first-pass: 87.7 -> 120.9 tok/s = 1.38x with 6 experts, 30% intra -> yes: but the accuracy at that point is 0.45 vs 0.67 (first pass omitted); text says '1.1-1.3x'
  - first-pass: Qwen3-30B 87%/83% -> yes: 0.60/0.69 and 0.57/0.69
  - first-pass: retention computed by fetcher -> yes: Retention for 25/50% is in the abstract; others derived from Table 1 by me too
- Transfers to our regime: no. Why: It reduces the NUMBER of experts (merging) and stores shared U per cluster; GPU 8x H800, no memory-bound/offload scenario; 50% expert reduction already loses 13-23% accuracy; merged experts change which expert a token needs, so it does not give a rank-r residual per original expert as in our variant. Interesting side: shared-U-per-group with expert-specific V is a lossy relative of our shared-basis + residual.
- Quality loss: Table 1/2: see numbers; PPL Mixtral 3.98 -> 5.16 (6 exp), 6.97 (4 exp); with intra compression 30% on 6x7B: WikiText 11.22, avg 0.45.
- Ablation: Table 3: clustering (K-means on output vs others), subspace merging (U vs V sharing, frequency weighting). Fig 4: with fine-tuning (+FT) gains 4-6%.
- Limitations (authors): Authors: needs calibration data; data-free left to future.
- Contradicts / bears on our findings: No. It reinforces that high compression of expert weights costs quality quickly.

---

## P-0229: MoBE: Mixture-of-Basis-Experts for Compressing MoE-based LLMs. Xiaodong Chen*, Mingming Ha, Zhenzhong Lan, Jing Zhang, Jianguo Li (Inclusion AI, Renmin Univ., Westlake). arXiv 7 Aug 2025 (alphaxiv result says ICLR 2026)

- ID/version read: arXiv 2508.05257v1 (PDF text, 972 lines); raw text read: True
- Setup: Models: Ling-Lite-Chat, DeepSeek-V2-Lite-Chat, Qwen3-30B-A3B-2507, Qwen3-235B-A22B-2507, DeepSeek-V3-0324 (671B), Kimi-K2-Instruct (1T, 384 experts). Each expert up/gate matrix W_i = A_i B_i with B_i = sum_j alpha_ij B_j over m shared basis matrices (activation f applied), learned by minimising reconstruction error (Adam lr 0.07, H100s, data-free); down matrix untouched. m=4 (small), 32 (Qwen3), 64 (V3), 128 (Kimi). MoBE-dagger reduces k to offset extra activated params (8->6). 15 benchmarks. Baselines D2-MoE (needs calibration; infeasible on big models) and MoLAE. No latency measured.
- Numbers (what | value | baseline | ratio | where):
  - Qwen3-30B-A3B-2507 avg of 15 benchmarks, 24% params removed | MoBE 75.8; MoBE-dagger 75.7 | original 78.6; D2-MoE 66.4; MoLAE 67.5 | 0.964 retention (-2.8); D2-MoE 0.845 | Table 3, p.9
  - Qwen3-235B-A22B-2507 24% | MoBE 80.9; dagger 79.7 | orig 81.5; MoLAE 73.2 | 0.993 | Table 3
  - DeepSeek-V3-0324 30% | MoBE 78.0; dagger 77.9 | orig 79.3; MoLAE 73.1 | 0.984 | Table 3
  - Kimi-K2 24% | MoBE 81.1; dagger 80.8 | orig 82.4; MoLAE 76.6 | 0.984 | Table 3
  - Ling-Lite 16% / DeepSeek-V2-Lite 15% | 58.6 / 36.9 | 59.7 / 38.0; D2-MoE 53.1 / 34.7 | 0.982 / 0.971 | Table 3
  - Average degradation over all models | ~1.4% (MoBE), ~0.5% (MoBE-dagger) | originals | authors' statement | Sec 4.3
  - Effective rank (95% energy) of real expert matrices | Kimi-K2 7168x2048: average effective rank > 1593 in most layers | break-even rank r_t = 7168*2048/(7168+2048) = 1593 | effective rank exceeds break-even => pure rank decomposition cannot compress without loss | Appendix B, Fig 9-11
  - Compression ratios achieved | 15-16% (small models), 24% (Qwen3), 30% (V3) | parameters | far below the 40-60% of D2-MoE/MoE-SVD | Table 3 'Ratio' column
- Catalog / first-pass claims:
  - Re-parameterises each expert's up/gate matrix as a combination of basis matrices shared across a layer -> yes: and keeps an expert-specific A_i; down matrix unchanged
  - authors argue SVD baselines such as D2-MoE lose substance -> yes: reasoning: effective rank above break-even (Appendix B), reconstruction error higher (Figs 12-14)
  - first-pass: 15-30% compression at 96-99% retention -> yes: Table 3 as above
  - first-pass: Qwen3-30B 78.6 -> 75.8 at 24%; D2-MoE 66.4 -> yes: Table 3
  - first-pass: no latency reported; custom kernel needed -> yes: Limitations: multiple fused-MoE calls mimic the factorisation, inefficient; mega-kernel needed
  - first-pass: H100 -> yes: 'All experiments are run on H100 GPUs'
- Transfers to our regime: no. Why: No speed data at all; it shrinks stored parameters 15-30% only; shared basis is the mechanism closest to ours, but the achieved ratio and the extra activated params (needs k reduction to compensate, MoBE-dagger) matter: each token reads basis + A_i; with fused expert kernels not available on CPU. Compression is vs bf16.
- Quality loss: Table 3: avg of 15 tasks drop 0.7-2.8 points (24-30% params); larger drops on AIME/LCB/MultiPL-E (e.g. Qwen3-30B MultiPL-E 70.6 -> 61.9, LCB 41.5 -> 35.6, both Qwen3-30B).
- Ablation: Activation function (Sec 4, ReLU has order-of-magnitude higher MSE), basis count m, reconstruction MSE comparisons (Figs 12-14). None for speed.
- Limitations (authors): Authors: slight accuracy drop remains; KD between full models left to future work; needs a mega-kernel.
- Contradicts / bears on our findings: Appendix B is the one direct empirical contradiction to our premise: measured effective rank of real raw expert matrices is above the break-even rank, so a low-rank residual of rank 64 will not carry 90% energy unless the shared part removes most of it (the delta-after-shared-base question remains unmeasured by us; D2-MoE Fig 2 and MoBE both indicate it is partly structured but lossy).

---

## P-0230: Effective MoE-based LLM Compression by Exploiting Heterogeneous Inter-Group Experts Routing Frequency and Information Density (RFID-MoE). Zhendong Mi, Yixiao Chen, Pu Zhao, Xiaodong Yu, Hao Wang, Yanzhi Wang, Shaoyi Huang (Stevens, Northeastern). Preprint dated 13 Feb 2026

- ID/version read: arXiv 2602.09316v2 (11 Feb 2026) PDF text, 862 lines (first pass cited v1, 10 Feb); raw text read: True
- Setup: Models: Qwen3-30B-A3B-2507 (main), DeepSeekMoE-16B-Base, DeepSeek-V2-Lite-Chat, Ling-mini-2.0, Qwen2-57B-A14B, Qwen3-235B-A22B-2507 (Fig 3 only). Builds on MoBE/basis sharing: experts sorted by routing frequency, grouped 4 per group, shared basis per group by SVD, group rank K_g = max(1, K_total * C_g / sum C), C_g = xi*E_g(effective rank) + (1-xi)*F_g(frequency), xi 0.7-0.8; plus a residual-reconstruction trained vector eta (3% of params) with sparse orthogonal projection. Routing statistics from 1024 WikiText-2 samples; 8x A100. Zero-shot tasks (6) and PPL (WikiText-2, PTB, C4). No speed measurement.
- Numbers (what | value | baseline | ratio | where):
  - Qwen3-30B-A3B-2507 avg accuracy at 60% compression | RFID-MoE 0.64 | original 0.66; MoBE 0.61; RS-MoE 0.55; NAEE 0.54; D2-MoE 0.50 | 0.97 retention (MoBE 0.92) | Table 2, p.6
  - PTB PPL at 60% | RFID-MoE 16.92 | MoBE 24.93; original 12.41; D2-MoE 38.84 | 1.36x original PPL; -8.01 vs MoBE | Table 2
  - HellaSwag at 60% | 0.66 | MoBE 0.58; original 0.69 | +8 points | Table 2
  - Qwen3-30B at 20% / 40% | 0.72 / 0.71 | original 0.66 | above the original (the MoBE row is 0.71/0.69, also above original; ARC-e 0.85 vs 0.78 original -- suspicious/inconsistent baselines) | Table 2
  - DeepSeekMoE-16B 60% | 0.60 | orig 0.62; MoBE 0.58; D2-MoE 0.43 (copied from D2-MoE paper numbers 11.67/27.73/27.63) | 0.97 | Table 2
  - Qwen3-235B PTB PPL | 10.67 | MoBE 11.13; original 10.11 | +0.56 over original | Fig 3, p.8
  - Rank floor | K_g = max(1, ...) | catalog: never-activated experts get zero rank | the paper describes zero-rank-for-never-routed as the flawed frequency-only strategy it rejects | Sec 3.3 eq 13; Intro p.2; Sec 3.2
  - Routing skew (Qwen3-30B layer 0) | Group 16: freq 5432, effective rank 557; Group 30: freq 158, effective rank 1049 | n/a | low-frequency groups can have higher information density | Table 1, p.4
- Catalog / first-pass claims:
  - Rank is allocated per expert group -> yes: groups of 4 experts sorted by frequency
  - some experts never activated get zero rank -> no: That is the failure mode of frequency-only allocation which the paper criticises; their rule floors every group at rank 1 and weights by effective rank (xi 0.7-0.8).
  - first-pass: Qwen3-30B at 60%: 0.64 vs MoBE 0.61 vs original 0.66; PTB 16.92 vs 24.93 -> yes: Table 2
  - first-pass: arXiv v1 10 Feb -> partly: v2 dated 11 Feb 2026 was read; preprint stamp 13 Feb
  - first-pass: A100 -> yes: 8x A100
- Transfers to our regime: no. Why: Perplexity/accuracy only, no latency or memory-speed data; basis-sharing method needing training (eta, bases); compression ratios target parameter count vs bf16; frequency statistics are gathered on a calibration set. Useful for us only as evidence that routing-skew-aware rank allocation helps quality, and that low-use experts can be information-dense (a caution against dropping cold experts' rank).
- Quality loss: Table 2: Qwen3-30B 0.66 -> 0.64 at 60% (3%); DeepSeekMoE-16B 0.62 -> 0.60; Qwen3-235B PTB +0.56 PPL. Baseline numbers in Table 2 show anomalies (MoBE/RFID above original at 20-40%).
- Ablation: Table 3 calibration source for frequency (WikiText-2 vs C4), Table 4 xi values (0.5/0.7/0.8 best near 0.7-0.8). No ablation of speed.
- Limitations (authors): No limitations section found; Impact Statement only.
- Contradicts / bears on our findings: Our E.md note 'zero-rank idea of RFID-MoE only applies to experts cold over a long run' rests on the catalog's wrong description; the paper does not allocate zero rank. Routing skew in Table 1 (5432 vs 158 tokens per group) supports our measured skew (hottest half = 98% of accesses).

---

## P-0231: REAL: low-rank compression of the weakest experts (GitHub anzr299/REAL, repo description 'MoE expert compression: SVD vs pruning vs merging on Qwen3.6-35B, Qwen3-30B, Gemma-4-26B (GSM8K + MMLU)'); no paper named; created 2026-07-07, last push 2026-07-09, no license

- ID/version read: raw README.md from raw.githubusercontent.com + GitHub API repo JSON (8.5k chars); code and results/ JSON not read; raw text read: True
- Setup: Qwen3.6-35B-A3B (40 layers, 256 experts, top-8, I=512; hybrid SSM + attention) and Gemma-4-26B-A4B-it (30 layers, 128 experts, top-8, I=704, parallel dense MLP). 'Weakest' experts picked by REAP-style saliency (router-weighted output L2 norm). Plain truncated SVD or activation-aware ALS refinement of gate/up/down in place (same shape; storage saving not realised, 'out of scope'). Calibration 8/68/24 c4/math/code (~846k tokens). GSM8K 5-shot chat greedy 1024 tokens; MMLU 0-shot no-chat (Qwen), 5-shot chat or forced-BOS 0-shot (Gemma). Hardware not stated; REAP and REAM rows are external numbers.
- Numbers (what | value | baseline | ratio | where):
  - Qwen baseline | GSM8K 87.87/89.08 (strict/flex); MMLU 83.73 | n/a | 1.00 | README Qwen table
  - Qwen SVD plain, 25% experts, rank 128 | GSM8K 87.49/89.16; MMLU 80.12 | 87.87/89.08; 83.73 | MMLU -3.6 points | README
  - Qwen SVD plain, 50% experts, rank 256 | GSM8K 89.08/90.22; MMLU 22.96 | 87.87/89.08; 83.73 | GSM8K +1.2 but MMLU chance-level (-60.8). First pass's '89.08' was the strict value for this row; 89.08 is also the baseline's flexible number | README
  - Qwen SVD plain 50% rank 128 / 64 | GSM8K 83.70/84.38 and 68.08/68.92; MMLU 22.95 both | 87.87/89.08 | GSM8K -4.2 / -19.8 | README
  - Qwen activation-aware + mean-baseline, 25%, rank 128 | GSM8K 87.95/89.61 but MMLU 23.02 | 83.73 | MMLU collapse even at 25% | README
  - Gemma baseline | GSM8K 86.96/88.32; MMLU chat 83.83; MMLU 0-shot 66.66 | n/a | 1.00 | README Gemma table
  - Gemma SVD plain 25% rank 176 | GSM8K 85.14/86.58; MMLU chat 80.99 | 86.96/88.32; 83.83 | MMLU -2.8 | README
  - Gemma SVD plain 50% rank 88 | GSM8K 84.08/85.37; MMLU chat 64.14 | 86.96/88.32; 83.83 | MMLU -19.7; GSM8K -2.9 | README
  - Gemma SVD plain 50% rank 128 | GSM8K 85.29/86.28; MMLU 0-shot 58.69 | 86.96/88.32; 66.66 | MMLU -8.0 | README
  - External baselines (not implemented) | REAP 50% Qwen: GSM8K 21.30, MMLU 61.67; REAM 50%: 79.23 / 51.64 | n/a | at 50% SVD/prune/merge all lose heavily on at least one benchmark | README
- Catalog / first-pass claims:
  - Python pipeline with activation-aware truncated SVD -> yes: plain SVD, activation-aware ALS and mean-baseline variants
  - not a llama.cpp change -> yes: Python/PyTorch in-place reconstruction
  - shows accuracy costs at 50% expert fraction -> yes: Qwen MMLU ~23 (chance) for all SVD variants at 50%; Gemma MMLU chat -19.7 at rank 88; GSM8K mostly preserved
  - first-pass: Qwen baseline GSM8K 87.87, SVD rank 256 89.08 MMLU 22.96 -> partly: 87.87 is the strict score; rank-256/50% plain SVD is 89.08 strict (90.22 flex); the 25%-fraction rows keep MMLU near 80 so the collapse is specific to 50% (and the mean-baseline variant).
  - first-pass: Gemma rank 88 GSM8K 84.08, MMLU 64.14 chat (baseline 83.83) -> yes: strict GSM8K; flex 85.37
- Transfers to our regime: no. Why: In-place reconstruction (no storage saving, explicitly out of scope); only the 'weakest' fraction of experts is compressed (25-50% of experts), not a shared basis plus residual for all; no speed; different models; MMLU collapse at rank 256 of I=512 shows a half-rank approximation of half the experts is unusable on Qwen. Useful as an accuracy ceiling for low-rank-per-expert on a 256-expert top-8 model: ranks near 50% of the intermediate dimension are needed even for 25% of experts.
- Quality loss: See numbers; Qwen MMLU -3.6 (25%, r=128) to chance (50%); Gemma MMLU chat -2.8 (25%, r=176) to -19.7 (50%, r=88); GSM8K small changes.
- Ablation: Plain vs activation-aware vs mean-baseline SVD, rank, expert fraction (README tables). No storage/speed ablation.
- Limitations (authors): README: memory saving needs factors stored separately and a changed forward pass (out of scope); results marked 'eval' still running; MMLU protocol depends on BOS handling for Gemma.
- Contradicts / bears on our findings: Supports the screen's warning that 50%+ compression of expert weights costs quality and that rank must be a large fraction of min dimension (Qwen I=512: rank 128 at 25% fraction loses 3.6 MMLU; rank 256 at 50% fraction loses everything).

---

## Family #16 (top-k reduction): prior art and what is new in our variant

Read at same depth (raw PDF text): Training-Free Halving of Activated Experts in Fine-Grained MoE (Xing Chen, Hengshuai Yao, arXiv 2609.04575v1, 4 Sep 2026); Distribution-Consistent Inference for Dynamic Sparse MoE (Kim, Soro, Hwang, KAIST, arXiv 2609.09241v1, 8 Sep 2026); LExI (Chitty-Venkata et al., Argonne, arXiv 2509.02753v1, 2 Sep 2025, setup/abstract/figure-2 text read, tables not extracted); NAEE (Lu et al., arXiv 2402.14800v2: abstract, intro read; speed tables not extracted by pdftotext -- UNVERIFIED numbers). Also (searched, not opened in full): Elbow routing 2608.04401 (P-0221, other batch), ASET (ACL Findings 2026), OEA 2511.02237, SERE 2602.07616, ZEDA 2605.18643.

### 2609.04575 (important: changes how to read our quality gate)
- Models: Qwen3.6-35B-A3B (40 layers, 256 experts, k=8) and Qwen3.5-397B-A17B (60 layers, 512 routed experts, k=10) -- the second is the same expert count and top-k as our target class. Method: activate top k1, but renormalise by the router mass of top k2 (one integer, no training, no extra compute). Hardware: not a speed paper (cost accounting only: halving k1 8->4 cuts routed-expert compute 50%, total expert term 44%, ~17% end-to-end vs ~3.0B active params).
- Key numbers: Table 4 PPL WikiText Qwen3.6-35B native 6.643; k1=6 with standard renorm (k2=6) +4.08%; k2=8 -1.61%; k2=256 (no renorm) +356%. Table 5 MMLU 5-shot n=2000: k1=4,k2=4 77.00% (-4.65 pts, p=1.7e-9); k1=4,k2=16 81.30% (-0.35, p=0.66) vs native 81.65%. Qwen3.5-397B: native 89.45%; k1=5,k2=5 87.35% (-2.10, p=1e-4); k1=5,k2=10 88.90% (-0.55, p=0.24); k1=8,k2=8 89.60 (+0.15). Table 6 GSM8K n=500: 397B k1=5,k2=5 -4.2 pts; k2=10 -0.8 pts. Perplexity vs MMLU choose different k2.
- Meaning for us: published training-free evidence on a 512-expert top-10 model says halving k (10->5) costs 0.55 MMLU points with the right reference mass and 2.10 with standard renormalisation. Our OLMoE-derived +45.6% PPL at top-4 is for a model whose router does not renormalise and has no shared expert, so it is not evidence about the 125B-class target. 10->4 (our row) is beyond what they tested (smallest k1/k = 0.5); their k2=E ablation shows catastrophic loss if the renormalisation convention is wrong, which is a pitfall when using llama.cpp's expert_used_count override on a renormalising architecture.
- Authors' limitations: both models from one series; English/Python corpora; no long-form generation or long context; paired-test non-significance is not proof of equivalence.

### 2609.09241 (DCI/LDA)
Qwen3-30B-A3B, vLLM, single RTX PRO 6000 Blackwell 96 GB (Table 4): k=8 910.71 tok/s TPOT 33.22 ms (ShareGPT), k=6 999.78, k=4 1156.26 (1.27x), k=2 1477.55 (1.62x); NuminaMath k=8 1035.71 -> k=4 1368.72 (1.32x). General-knowledge average (Table 1/2 row): 74.44% at k=8, 66.49% at k=4 (-7.95), 72.17% with their correction (-2.27). Throughput numbers are batched serving, not batch-1. Training-free correction is an inference-time representation realignment. Supports the 1.3x ceiling lift and shows a quality fix exists; GPU compute-bound, so not our regime.

### LExI (data-free per-layer k)
H100, batch size 16, vLLM; states that expert pruning improves memory but not throughput, whereas lowering the active-expert count does; per-layer allocation beats uniform pruning at equal throughput (Qwen1.5-MoE: same throughput as 50% pruning with +10% accuracy). Directly supports non-uniform k per layer (as HookMoE's TLNLE). Tables not extracted; numbers beyond the abstract are UNVERIFIED.

### What is new in our variant
Not new: reducing k at inference (NAEE 2024, dynamic top-p, LExI, Elbow, HookMoE, Training-Free Halving, DCI, MC-MoE ODP, the NVIDIA-forum 122B build). What none of the sources do: (1) evaluate k reduction where the saving is bytes streamed from NVMe at batch 1 with a cache sized as a small share of experts (all report GPU/compute or resident-memory speedups); (2) pair it with a footprint-aware cache. The paper that comes closest is 2609.04575 for the quality side and ProMoE/HOBBIT-style offloading papers for the system side (not re-read here). A joint top-k + cache + NVMe measurement would be new, but only the simulation exists on our side.

---

## Family #17 (shared basis + low-rank residual per expert): prior art and what is new in our variant

Read at same depth (raw PDF text): MoE-SVD (Li et al., ICML 2025, PMLR v267 li25az; no arXiv listing found), D2-MoE, Sub-MoE, MoBE, RFID-MoE (rows above). Searched, only abstract/snippet: MoLAE (Liu et al., Huawei, arXiv 2503.23100: latent-expert factorisation, shared latent projection plus small expert-specific matrices; appears as baseline in MoBE Table 3 with 12-30% compression), MoBiE (2604.06798, snippet), MoE-I2, MC-SMoE, HC-SMoE (baselines inside the read tables).

### MoE-SVD (raw text read)
One V matrix shared across experts per layer and top-k trimmed U matrices; Mixtral-8x7B and Phi-3.5-MoE, throughput on a single H800 GPU. Table 2: Mixtral original 87.7 tok/s; 20% 104.7 (1.2x), avg acc 0.58 (0.60 with LoRA FT) vs 0.63; 40% 109.8 (1.3x), 0.50; 50% 123.9 (1.4x), 0.43; 60% 156.1 (1.8x), 0.37 (PPL 13.52). Phi-3.5-MoE original 98.2; 60%: 148.7 (1.5x), 0.49 vs 0.62. Text: up to 1.52x (Phi) / 1.53x (Mixtral) at 60% (Fig 4); weight memory 40.4% of original at 60%. So the abstract's '60% compression, 1.5x faster' hides -0.26 (Mixtral) / -0.13 (Phi) average accuracy. Baselines ASVD/SVD-LLM collapse (PPL 1e3-1e4) at 50-60%. Limitation stated: SVD adds communication overhead under tensor parallelism.

### Joint reading across the family
- Every method reports compression against 16-bit weights; none against a 4-bit baseline; none measures CPU or SSD decode; the only speeds are GPU (H800, A100, 8x H800) at batch 64 or unspecified, 1.2-1.8x for 20-60% size reduction, mostly compute effects.
- Quality: 20% costs 0.02-0.05 avg accuracy; 40% 0.05-0.13; 60% 0.11-0.26 on Mixtral-class models (D2-MoE 0.52 vs 0.63; MoE-SVD 0.37 vs 0.63); best-in-class MoBE/RFID-MoE keep ~96-99% but only at 15-30% (MoBE) or via trained bases (RFID-MoE, 60% claim, own baselines inconsistent).
- Real expert matrices: MoBE Appendix B measures effective rank (95% energy) above the break-even rank for Kimi-K2; so a pure low-rank factorisation of raw expert weights cannot reach large compression losslessly. D2-MoE Fig 2 argues deltas from a Fisher-merged base have faster singular-value decay (figure only, no number).
- REAL README: in-place SVD of the weakest 25% of Qwen3.6-35B experts at rank 128 (of 512) costs 3.6 MMLU points; 50% of experts at any tested rank destroys MMLU (~23).

### What is new in our variant
Not new: shared base (+ trained/Fisher/SVD basis) with per-expert low-rank residual (D2-MoE, MoE-SVD, MoBE, MoLAE, RFID-MoE, Sub-MoE). Possibly new: the objective and regime -- minimise bytes streamed per token from NVMe at batch 1 against a ~4.3-bit baseline, with the shared base read once per layer and cached while only residual factors are streamed per expert. No paper found measures this; none reports a kernel that reads a shared basis once per layer on CPU; none compares against a quantised baseline. Closest papers: D2-MoE (structure: shared base + SVD delta) and MoBE (shared basis + per-expert coefficient matrix). Also worth noting that the published evidence, read in full, makes the premise (residual energy of 90% at r=64) look doubtful rather than supported, and that fp16 low-rank factors vs 4.3-bit weights break even at rank ~150 (our arithmetic, E.md).
