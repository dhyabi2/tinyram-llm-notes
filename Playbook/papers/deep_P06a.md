# Deep re-check, batch P06a (raw full text)

Rows: P-0121, P-0122, P-0123, P-0125, P-0127, P-0128, P-0129, P-0130. Raw text read: 8 of 8, unread: 0. Raw files in the session scratchpad deep_P06a.


## P-0121: mbolt 0.1.0: Profile-guided layout optimization for GGUF MoE models (PyPI mbolt, github doramirdor/mbolt; author Dor Amir; no paper)

- Version read: PyPI JSON (version 0.1.0, uploaded 2026-07-13); GitHub README.md raw; results/phase0-gate.md, phase1-report.md, phase2-report.md raw; mlx-lm issue #1438 (32 comments) via GitHub API; repo API (8 stars, created 2026-07-13, last push 2026-07-22)
- Setup: Phase 2 / README table: Qwen3-Next-80B-A3B-Instruct UD-IQ3_XXS, 33.1 GB, 48 layers x 512 experts, top-10 (same geometry class as ours); Apple M5 Pro MacBook Pro 64 GB (phase0 report), Apple SSD AP1024Z 1 TB NVMe; CPU experts (-ot ffn_exps=CPU); 24 GB mlocked 'squeeze' (phase2 section 3; sensitivity row 32 GB squeeze, which the report calls 'tighter memory' although 32>24: the two squeeze labels are inconsistent in the text); 128 tokens, 3 reps, medians; baseline = stock GGUF + stock llama.cpp mmap fault path. Phase 0 replay used Qwen3-30B-A3B IQ3_XXS (128 experts, top-8, slices 588-816 KiB), 200 prompts, trained on first 80% (60,618 tokens), tested on held-out 15,155 tokens; physical replay, QD1.
- Numbers (what | value | baseline | ratio | where):
  - e2e tok/s stock file stock llama.cpp | 5.80 | - | 1.00 | README 'The numbers'; phase2 sec.3
  - e2e stock file + explicit-read prefetcher | 7.60 | 5.80 | 1.31 | README
  - e2e reorder (chain+pipeline) + prefetcher | 8.00 | 5.80 stock / 7.60 stock+pf | 1.38 vs stock; 1.053 vs prefetcher alone (own calc) | README
  - e2e interleave (up|gate|down contiguous) + prefetcher | 9.00 | 5.80 stock / 7.60 stock+pf / 8.00 reorder+pf | 1.55 vs stock; 1.184 vs prefetcher alone; 1.125 vs reorder+pf (own calc) | README
  - e2e interleave with fault streaming (no prefetcher) | 5.70 | 5.80 | 0.98 | phase2 sec.3: layout alone does nothing under mmap
  - 32 GB squeeze: stock / stock+pf / interleave / interleave+pf | 6.00 / 9.10 / 6.10 / 9.80 | 6.00 | 1.52 / 1.02 / 1.63; layout increment over prefetcher 9.80/9.10 = 1.077 | phase2 sec.3 sensitivity (squeeze32.log)
  - storage-level reads/token, cold, held-out replay on physical files | 1418 -> ~370; 301.7 -> 135.4 ms/token | stock file | 2.23x (sim predicted 2.29x) | phase2 sec.3 table
  - prefetcher counters | 28.6k -> 8.5k reads for ~13 GB; avg read 450 KB -> 1.5 MB | stock | 3.3x fewer reads | phase2 sec.3
  - Phase 0 replay (30B, warm-32/128 LRU slots) chain+pipeline | 1.23x warm, 1.27x cold; chain alone 1.14x; clique 1.11x; heat 1.06x; interleave 1.59x/1.78x | stock layout | - | phase0-gate.md table; CV 11-14%
  - Phase 0 fit I/O model t = reads*L + bytes/B | L=73 us, B=11.3 GB/s, R2=0.956; modeled chain+pipeline 1.10x vs measured 1.23x on this Mac; PCIe4 NVMe 1.07x, PCIe3 1.05x, SATA/USB 1.03x; interleave 1.42/1.28/1.12x | - | - | phase0-gate.md
  - cross-domain: layout trained on dolly applied to code | 1.22x (in-domain 1.46x, mixed 1.61x) | stock | - | README (phase1 data, not re-derived by me)
  - independent MLX cross-check (third parties, mlx-lm#1438), coalesce lever A2 (9 slices to 1 read) Qwen3.6-35B-A3B-8bit | I/O 2.34x; decode +17.6% at 0.3 resident, +14.3% at 0.45 | MLX offload reader with shared fds and 8-way pool | - | comment by mabaeyens 2026-07-20
  - same, gpt-oss-120b MXFP4 (~4 MB slices, 56.7 GiB table, 32 GB RAM) | A2 I/O 1.29x, A1 1.08x, e2e +9.7% (read share 39.3%) | - | - | same thread 2026-07-20 13:37
  - same, Qwen3-235B-A22B 4-bit, fraction 0.2, M5 Pro 48 GB (A1 only: 9 preads to 3) | 1.677 -> 2.218 tok/s decode (+32.3%), TTFT 13.34 -> 9.83 s (-26.3%), preads/token 2394 -> 798 | stock reads | 1.32x decode | comment by lBroth 2026-07-20 16:55; author states per-read gain only 1.339x by formula, the rest is read share near 1
- Catalog / first-pass numbers:
  - [yes] 1.55-1.63x faster memory-squeezed MoE streaming in llama.cpp: 9.00/5.80=1.552 (24 GB squeeze) and 9.80/6.00=1.633 (32 GB squeeze). But baseline is stock mmap; the author's own explicit-read prefetcher alone gives 1.31x / 1.52x, so the layout share is 1.18x / 1.08x
  - [yes] 2.23x at the storage level: 1418 -> ~370 reads/token, 301.7 -> 135.4 ms; this is the INTERLEAVE layout (up|gate|down contiguous), not co-activation reorder
  - [yes] rewrites GGUF byte-exact (+0.003% size): phase2: '+0.003 % file size' for interleave; weights byte-exact claimed, KLD 0.00096, top-1 98.9% (README) - output is not token-identical under any permutation per the author
  - [yes] repo github: github.com/doramirdor/mbolt, 8 stars, created 2026-07-13
- Transfers to our regime: partly. Same expert geometry (512 experts, top-10, CPU experts, NVMe streaming, batch 1) so it is the closest published regime. But (a) only 33 GB model squeezed to 24-32 GB RAM, we have 69 GB vs 8-32 GB; (b) most of the end-to-end gain is the explicit-read prefetcher (1.31x/1.52x) and the interleave, not the co-activation reorder (1.05x over prefetcher, 1.08x at 32 GB); (c) layout is invisible to mmap faults; (d) self-reported, one machine, N=3, 8 stars; (e) the author states profile-free coalescing carries ~90% of the win and the clique reorder adds ~12% of read ops at decode, and that plain LRU beats any pinned set and cross-layer prefetch is a null.
- Quality loss: Weights byte-exact; routing equivalence 100.000% through the permutation at equal inputs; KLD 0.00096 and top-1 agreement 98.9% (5x below CPU vs Metal backend delta) per README; author notes token-identity is impossible under any expert permutation (floating-point sum order). No benchmark scores.
- Ablation: Yes and it isolates what we need: stock+pf 7.60, reorder+pf 8.00, interleave+pf 9.00, interleave without pf 5.70 (README, phase2). Phase 0 replay ablation: heat 1.06, pipeline 1.07, clique 1.11, chain 1.14, clique+pipeline 1.23, chain+pipeline 1.23, interleave 1.59 (warm 32/128).
- Limitations: Authors state: gains scale with I/O-boundness (model >= RAM); layout only pays with explicit reads (mmap parity measured); big ~4 MB slices cap per-read gain at ~1.3x; moderate ratios with big slices weak (~+10%); LRU-caching engines see +14-18% not the no-retention numbers; trace should resemble deployment (matched tracing worth ~20-30%); single machine; baseline varied between sessions (5.8 vs 10.0 across days, within-session ratios only); interleave is CPU-experts only.
- Contradicts our findings: No. It confirms them: our leave-one-out reorder gain 1.045x serial / 1.073x overlapped at 8 GB, 1.003x at 16 GB and 1.000x at 32 GB (pubtech/C.md) matches mbolt's 1.053x over the prefetcher and 1.077x at 32 GB squeeze; our held-out run length 1.35 on 128-expert traces; mbolt's own simulator over-predicted (modeled 1.10x, replay 1.23x, real e2e 1.05x). What changes: the large lever is the contiguous per-expert interleave (our family #19 repack), 2.23x I/O and 1.18x e2e over the prefetcher; do not credit it to co-activation reorder.

## P-0122: LLM in a flash: Efficient Large Language Model Inference with Limited Memory (Keivan Alizadeh, Iman Mirzadeh, Dmitry Belenko, S. Karen Khatamifard, Minsik Cho, Carlo C Del Mundo, Mohammad Rastegari, Mehrdad Farajtabar; Apple; ACL 2024)

- Version read: arXiv:2312.11514v3 [cs.CL] 30 Jul 2024, PDF via pdftotext (139,904 chars)
- Setup: OPT 6.7B (ReLU), sparsified Falcon 7B, Phi-2, Persimmon 8B, Llama 2 7B (FATReLU); dense FFN neurons, no experts. Hardware: Apple M1 Max 1 TB SSD, Apple M2 Ultra 2 TB SSD (CPU float32 or Metal float16), Linux box with RTX 4090 24 GB (bf16). About half of the model allowed in DRAM. Flash throughput: 6.10 GB/s dense sequential, 1.25 GB/s sparse reads, 2.25 GB/s with bundling; 32 reader threads; reads without OS caching (purge). Data: C4 validation, 128-token prompt, 256 generated tokens. Baselines: naive (reload on demand, best theoretical I/O numbers used, 'real number might be higher') and hybrid (half in DRAM, half loaded). Software: HF Transformers + PyTorch + custom binary extension.
- Numbers (what | value | baseline | ratio | where):
  - Table 2 OPT 6.7B on M1 Max I/O latency: naive | 13.4 GB, 6.10 GB/s, 2196 ms | - | - | Table 2
  - Table 2 hybrid | 6.7 GB, 1090 ms | 2196 | 2.01 | Table 2
  - Table 2 + predictor | 0.9 GB, 1.25 GB/s, 738 ms | 1090 | 1.48 | Table 2
  - Table 2 + windowing | 0.2 GB, 164 ms | 738 | 4.50 | Table 2
  - Table 2 + row-column bundling | 0.2 GB, 2.25 GB/s, 87 ms | 164 | 1.885 (throughput 1.8x); vs naive 25.2x | Table 2
  - Table 3 OPT 6.7B CPU total latency | 3182 -> 669 ms | naive | 4.76 | Table 3
  - Table 3 OPT 6.7B Metal M1 / Metal M2 / GPU | 2389->565 / 2270->305 / 2218->84 ms | naive | 4.23 / 7.44 / 26.4 | Table 3
  - Table 3 Falcon 7B CPU | 3095 naive, 1947 hybrid, 706 all | naive / hybrid | 4.38 / 2.76 | Table 3
  - Table 3 Persimmon 8B CPU | 3806 / 2495 / 1041 | naive / hybrid | 3.66 / 2.40 | Table 3
  - Table 3 Phi-2 CPU | 1287 / 711 / 546 | naive / hybrid | 2.36 / 1.30 | Table 3
  - Table 3 Llama 2 7B CPU | 3095 / 1903 / 994 | naive / hybrid | 3.11 / 1.91 | Table 3
  - Appendix D co-activation bundling | negative result, no numbers | - | - | Sec. 3.2 and App. D
- Catalog / first-pass numbers:
  - [yes] Row-column bundling to enlarge read chunks works: Table 2: 164 -> 87 ms, 1.25 -> 2.25 GB/s; it concatenates an up-projection column with the matching down-projection row (2*d_model*bytes chunk)
  - [yes] bundling by co-activation was a negative result: App. D: bundling a neuron with its closest friend loaded very active neurons repeatedly; hot neurons are the closest friend of many. No quantification in the paper
  - [yes] (first pass) Table 3 OPT 6.7B CPU 3182 -> 669 ms: also Metal M1 2389->565, GPU 2218->84; abstract text says 'up to 4x and 20x', tables compute to 4.76x and 26.4x
  - [yes] (first pass) bundling takes I/O from 164 to 87 ms: it is a cumulative step after windowing; the baseline of that step is already the sparse-read pipeline
- Transfers to our regime: no. Dense ReLU-sparse FFN neurons (predictor decides which neurons), not top-k routed experts; DRAM holds half the model plus a sliding window of recently active neurons, while we have 8-32 GB against 69 GB; baselines are best-case theory, not llama.cpp; Apple SSD with 32 parallel readers. The only transferable pieces are the chunk-size/throughput curve (1.25 -> 2.25 GB/s when chunk doubles), which matches our 0.9 MB -> 4 MiB curve idea, and the negative co-activation-bundling result (qualitative).
- Quality loss: Predictor-induced: Table 4 e.g. OPT 6.7B ArcEasy 66.1 -> 66.2; Falcon relufied 72.52 -> 70.20 to 72.35 depending on predictor rank; Phi-2 relufied 80.60 -> 78.90-79.96; Llama 2 MMLU 41.8 dense, 38.96 sparsified, 38.63 with predictors (App. C).
- Ablation: Yes (Table 2, cumulative: hybrid, predictor, windowing, bundling) but bundling only in its row-column form; co-activation bundling has no numbers.
- Limitations: Authors: only half of model in DRAM tested; relies on ReLU sparsity (relufied models for Falcon, Phi-2, Llama2); baselines use theoretical best I/O ('real number might be higher'); needs predictors; they note OS buffer cache could improve figures; Metal needs custom extension for dynamic shapes.
- Contradicts our findings: No. Consistent with our held-out co-activation reorder being weak (pubtech/C.md family #20). Their row-column bundling is the analogue of our gate/up/down repack (family #19), not of reorder.

## P-0123: MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache (Leyang Xue, Yao Fu, Zhan Lu, Chuanhao Sun, Luo Mai, Mahesh Marina; Univ. of Edinburgh)

- Version read: arXiv:2401.14361v3 [cs.LG] 12 Mar 2025, PDF via pdftotext (84,165 chars)
- Setup: Models: Switch-128x0.2B (30-100 GB), NLLB-MoE 128x0.4B (220 GB), DeepSeek-V2-Lite 64x2.4B (31 GB), Mixtral-8x7B (120 GB), Snowflake Arctic 128x4B (900 GB). Hardware: one NVIDIA RTX A5000 24 GB, PCIe 4.0 (text 32 GB/s, Table 1 caption 24 GB/s); host DRAM 32 GB (DeepSeek), 64 GB Switch, 256 GB NLLB, 128 GB Mixtral, 1 TB Arctic - every model fully resident in host DRAM. Batch 1, prompt 512, decode 32; 290 tasks from BIGBench/FLAN/MMLU. Baselines: DeepSpeed-Inference, llama.cpp via Ollama (CPU computes offloaded params), Mixtral-Offloading, BrainStorm (re-implemented by authors), vLLM. Request-level Expert Activation Matrix Collection (EAMC) for cache replacement and prefetch.
- Numbers (what | value | baseline | ratio | where):
  - Table 1 DeepSeek-V2-Lite TPOT avg: MoE-Infinity | 155 ms (tail 169; GPU idle 51) | - | - | Table 1 p.1
  - Table 1 vLLM | 485 ms | 155 | 3.13 | Table 1
  - Table 1 DeepSpeed | 737 ms | 155 | 4.75 | Table 1
  - Table 1 Mixtral-Offloading | 1250 ms | 155 | 8.06 | Table 1
  - Table 1 Ollama (llama.cpp) | 2590 ms | 155 | 16.7 | Table 1
  - Fig. 7 end-to-end TPOT | Switch 155 ms, NLLB 531 ms, DeepSeek 155 ms, Mixtral 836 ms (worst case, 8 large experts); about 1.4x better than BrainStorm/Mixtral-Offloading on Mixtral | baselines in figure (values only in plot) | 1.4 on Mixtral | Sec. 5.1 text
  - Headline range | abstract 3.1-16.7x; conclusion 2.7-13.7x | - | - | Abstract / Sec. 6 (the two disagree inside the paper)
  - Workload change recovery | about 50 requests within a dataset, 30 across datasets | - | - | Table 3
- Catalog / first-pass numbers:
  - [no] 2-20x over llama.cpp and others: Abstract 3.1-16.7x, conclusion 2.7-13.7x; max 16.7x is vs Ollama on DeepSeek-V2-Lite
  - [yes] Activation-aware prefetch/cache from request-level traces: EAM = L x E count matrix per request, matched by cosine distance to a collection of past EAMs
  - [yes] no disk layout: no SSD at all: all experts in host DRAM, GPU cache, PCIe
  - [yes] (first pass) DeepSeek-V2-Lite Table 1 155 ms vs 2590 ms Ollama: ratio 16.7
  - [yes] (first pass) 'host DRAM 32 GB to 1 TB': 32 GB for DeepSeek up to 1 TB Arctic
- Transfers to our regime: no. GPU cache over PCIe from host DRAM; no NVMe, no CPU compute path except the Ollama baseline; cache sized as GPU memory; DeepSeek-V2-Lite has 64 experts (top-6) so request-level reuse is easier than 512 experts top-10. Useful only as cache-policy prior art. Key finding relevant to us: reuse skew exists inside a request but disappears across many requests, so a static layout/pin trained across requests loses (consistent with our session-pinned and per-sequence cache families).
- Quality loss: None (exact caching/prefetch).
- Ablation: EAMC capacity sweep (Fig. 9, 1-120) and prefetch/cache micro-benchmarks (Sec. 5.2); no layout ablation.
- Limitations: Authors: Mixtral (few large experts, high activation ratio) is the worst case (836 ms); long-context shrinks the GPU expert buffer; all KV in GPU memory; EAMC needs adaptation after workload shifts.
- Contradicts our findings: No. Its premise (request-level reuse, global skew vanishes) matches our per-sequence and pinned-set results. Note Ollama 2590 ms is a poor llama.cpp number because 32 GB DRAM and CPU compute with PCIe-staged experts; do not compare to our CPU-only llama.cpp.

## P-0125: MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts (v1, Wenfeng Wang, Jiacheng Liu, Xiaofeng Hou, Xinfeng Xia, Peng Tang, Mingxuan Zhang, Chao Li, Minyi Guo; SJTU/HKUST); renamed in v2/v3 Specter: Fast MoE Inference via Quantization-Enabled Multi-Step Speculative Offloading (ASPLOS 2027)

- Version read: arXiv:2511.14102 v1 (18 Nov 2025, 104k chars), v2 (20 Oct 2026) and v3 (22 Oct 2026, 117k chars), all read via pdftotext; abs page: 'Accepted to ASPLOS 2027'
- Setup: v1/v3: Phi-3.5-MoE (41.9B total/6.6B active, 8 experts, top-2, 78 GB FP16), Qwen1.5-MoE-A2.7B (14.3B/2.7B, 60 experts top-4 + 1 shared, 29 GB), DeepSeek-V2-Lite (15.7B/2.4B, 64 experts top-6, 34 GB). One A100 40 GB, 24-core Xeon Silver 4310, 256 GB RAM, PCIe 4.0 x16 (16 GB/s per direction). Draft = GPTQ INT4 copy of the target's own experts (sharing weights/KV); target FP16. Datasets C4, WikiText-2, HumanEval, GSM8K, GPQA. Batch 1 (v1 and main v3) plus B=1..32 in v3. Baselines v1: HF Transformers, Mixtral-Offloading SC/SM; v3 adds vLLM, llama.cpp, SpecMoEOff (primary). GPU budgets high/low 24/18 (DS), 20/16 (Qwen), 40/32 (Phi) GB. Software: HF Transformers + CUDA streams.
- Numbers (what | value | baseline | ratio | where):
  - v1 abstract headline Phi-MoE | at most 2.34x over state-of-the-art offloading framework | unspecified | 2.34 (not derivable from text tables; figure only) | v1 abstract, Sec.1 contributions
  - v1 Fig.13 text, Phi-MoE high memory | 163.1 ms vs Mixtral-Offloading-SC 536.7, SM 351.8, Transformers 960.5 | MO-SC / MO-SM / HF | 3.29 / 2.16 / 5.89 (own calc) | v1 Sec.4.2
  - v1 Qwen2-MoE | 74.1 ms | MO-SC 2.5x, Transformers 3.5x (as stated) | - | v1 Sec.4.2
  - v3 abstract headline | average 1.50x, up to 1.88x TPOT speedup over SpecMoEOff, single request | SpecMoEOff | 1.88 = Qwen1.5-MoE high-mem 66.4/35.3 (own calc) | v3 abstract, Sec.4.2.1
  - v3 Phi-3.5 high-mem TPOT | 163.3 ms | SpecMoEOff 292.1, MO-SM 295.1, vLLM 297.5, Transformers 960.6 | 1.79 / 1.81 / 1.82 / 5.88 (own calc) | v3 Sec.4.2.1
  - v3 DeepSeekV2-Lite high-mem | 72.2 ms | Transformers 344.9 | 4.78 | v3 Sec.4.2.1
  - v3 Table 3 batch 32 DeepSeekV2-Lite avg TPOT | 119.2 ms | SpecMoEOff 718.0, MO-SM 1262.9, Transformers 1493.6 | 6.02 / 10.6 / 12.5 (own calc) | v3 Table 3
  - v3 cache hit rate (DeepSeekV2-Lite) | 99.85 / 98.62 / 96.25% at 16/24/32 GB | LRU 29.2 / 50.94 / 76.56% | - | v3 Fig.17, v1 Table 3
  - v3 Table 5b ablation | 69.2 full; 83.5 w/o PIO; 94.4 w/o PIO+SIC; 130.4 w/o PIO+SIC+SEE ms/token | full | 1.21 / 1.36 / 1.88 slowdown | v3 Table 5b
  - v3 Table 6 oracle prefetch error injection | 63.0 ms at 100% accurate; 84.2 ms at 41% accurate | 63.0 | 1.34 | v3 Table 6
  - v3 Table 5a H2D hidden | 55.5% / 60.5% / 22.4% (DS / Qwen / Phi) | - | - | v3 Table 5a
  - v1 Table 4 ablation DeepSeekV2-Lite | 13.02 full; 12.37 w/o async prefetch; 8.88 w/o fused kernel; 8.29 neither (units tok/s) | full | 0.95 / 0.68 / 0.64 | v1 Table 4
- Catalog / first-pass numbers:
  - [yes (v1 only)] up to 2.34x over best offloading framework on Phi-MoE: 2.34x appears in v1 abstract and intro; it is absent from v2/v3 (full text grep; only a reference title matches). v3 headline is 1.50x average / 1.88x max over SpecMoEOff
  - [yes] Small draft predicts experts to prefetch: INT4 GPTQ draft with shared weights produces multi-step expert lookahead; recall 0.93-0.96 (v3 Table 6b)
  - [yes] (prefetch gain, not fewer bytes): authors say draft overlaps PCIe with compute; verification of k tokens increases distinct experts per layer
  - [yes] title stale: paper renamed Specter in v2 (Oct 2026)
  - [yes] (first pass) v3 up to 6.02x at batch 32 on DeepSeekV2-Lite: vs SpecMoEOff; Table 3 718.0/119.2
  - [yes] (first pass) 'Xeon, 256 GB RAM': Xeon Silver 4310, 256 GB
  - [yes] (first pass) 2.34x 'not found in the first 100k characters of v3': confirmed absent in full v3 text; first pass's 39k unread chars are now read
- Transfers to our regime: no. A100 GPU with PCIe from host DRAM; benefit is overlapping 16 GB/s PCIe transfers with draft compute; needs a GPU-resident quantized draft copy of the same model plus FP16 target; 8-64 experts; CPU-only, 8-32 GB RAM, NVMe streaming break all assumptions. Bytes read are not reduced, and our simulation (pubtech/C.md #21) shows verify-union bytes ratio >= 1 for K>=2.
- Quality loss: None claimed: target verification keeps output distribution exact (v3 Sec.5 says 'standard target verification preserves the output distribution'); draft acceptance >90% (v1 Sec.4.5).
- Ablation: Yes: v1 Table 3 hit-rates for LRU/LRU(scaled)/single-prefetch/speculative; v3 Table 5b cumulative component removal (PIO, SIC, SEE), v1 Table 4 (async prefetch, fused kernel). It cleanly isolates prefetch from cache size (LRU-S at same slots).
- Limitations: v3: Phi-3.5 hides only 22.4% of H2D; gains shrink when LRU-S has big cache (32 GB); batch growth increases expert working set (Fig.16); depends on high draft-target alignment. v1: cannot compare to vLLM/llama.cpp directly (built on Transformers); no small same-architecture MoE draft exists so draft ablation 'cannot be conducted'.
- Contradicts our findings: No. Matches pubtech/C.md #21: draft-based gain is prefetch overlap, not fewer bytes. Table 6 shows oracle prefetch is only 1.34x better than 41%-accurate prefetch, i.e. on a GPU-PCIe system prediction quality has limited upside; in a disk-bound CPU system with prefetch window of one layer compute (~ms) vs read time the overlap budget is far smaller.

## P-0127: Efficient Mixture-of-Experts LLM Inference with Apple Silicon NPUs (NPUMoE; Afsara Benazir, Felix Xiaozhu Lin; University of Virginia)

- Version read: arXiv:2604.18788v1 [cs.LG] 20 Apr 2026 (catalog says 22 Apr), PDF via pdftotext (150,525 chars)
- Setup: Models (Table 1): Phi-3.5-MoE-Instruct 42.6B total/6.6B active, 32 layers, 16 experts top-2 (83.75 GB, ~150 MB per expert); Phi-tiny-MoE 3.8B/1.1B (7.51 GB); Qwen3-30B-A3B 128 experts top-8, 48 layers (28.64 GB). Hardware: Apple M2 Max (64 GB, 12-core CPU) and M2 Ultra (192 GB, 24-core CPU, 32-core ANE); the text does not say which model ran on which chip (an 83.75 GB model cannot fit the 64 GB machine). Baselines all Core ML based: Core ML CPU-only, Core ML naive (default scheduling), ANEMLL (attention on NPU, experts on CPU). Workloads HellaSwag, BoolQ, RULER; long-context prefill dominated; average decode length 8 tokens. Chunked prefill, FP16 compute (INT8 dequantised), whole model resident in unified memory; energy with Zeus.
- Numbers (what | value | baseline | ratio | where):
  - Prefill TTFT reduction overall | 1.32x-5.55x across baselines and settings | Core ML CPU / naive / ANEMLL | - | Abstract, Sec.5.3, Fig.14
  - Prompt 1024, chunk 256 / 512 | 2.07x and 1.32x vs Core ML CPU; 5.55x and 3.44x vs naive; 1.95x and 1.33x vs ANEMLL | - | - | Sec.5.3
  - Prompt 4096, chunk 512 / 1024 | 1.58x/1.72x vs CPU, 3.85x/3.70x vs naive, 1.50x/1.56x vs ANEMLL | - | - | Sec.5.3
  - Energy per token | 1.81x-7.37x better (CPU 1.81-3.19x, naive 4.71-7.37x, ANEMLL 1.86-3.43x) | - | - | Sec.5.4 Fig.15
  - End-to-end (prefill + ~8 decode tokens) | up to 3.86x vs naive, 1.26x vs Core ML CPU, 1.19x vs ANEMLL | - | - | Sec.5.3
  - Attention latency penalty | +27.6% on NPU/CPU backend | - | - | Sec.5.3
  - Ablation Fig.17 | Ours-all 2.89x faster than Ours-base (NPU one graph per expert, worst), 1.09x faster than Ours-TG; Ours-TG best energy (998.4), 3.74x less than Ours-base | Ours-base | 2.89 | Sec.5.6
  - Calibration overlap | 86% (top-4) / 94% (top-8) mean overlap on unseen inputs in prefill; PhiMoE 65.2% prefill / 77.0% decode across datasets | random | - | Sec.5.6 Fig.18; Phi has only 16 experts so top-8 is half the pool
  - Table 3 accuracy PhiMoE with token dropping | token drop 10.63-21.77%, padded 31.7-37.5%; accuracy BoolQ 86.00 -> 86.00, Hellaswag 88.57 -> 87.62 (-1.07), RULER niah_single_2 95.5 -> 95.0, multikey 96 -> 96 | FP16 | - | Table 3
  - Padding | 35.35% of computed tokens zero-padded; 1.25-1.36x CPU cycles | - | - | Sec.5.6
  - GPU comparison | GPU 1.26x-5.4x faster than NPU but 1.87x-4.48x more energy, 5.65x-9.41x power | NPUMoE | - | Sec.5.6 / App. A
- Catalog / first-pass numbers:
  - [yes] NPU for MoE prefill on unified memory: Core ML graphs, static capacity tiers, grouped expert execution, load-aware compute residency
  - [yes] NPU prefill without its optimizations performs worst: Ours-base worst; 2.89x gap
  - [yes] 86% (top-4) / 94% (top-8) expert-prediction overlap: but as offline-calibration overlap in prefill, and the model is not named in that sentence; Phi has 16 experts
  - [yes] (first pass) energy 1.81x-7.37x: -
  - [yes] (first pass) 2.07x vs Core ML CPU at prompt 1024 chunk 256: -
  - [yes] (first pass) 'whole model resident in RAM': load and cache once at compile time; graph eviction only if unified memory overloaded; but 83.75 GB vs 64 GB machine unresolved
  - [no] (first pass) arXiv date 22 Apr 2026: paper stamp is 20 Apr 2026 (catalog's 22 Apr probably listing date)
- Transfers to our regime: no. Everything is in unified memory so prefill is dispatch/compute-bound, opposite of NVMe-bound; the baselines are Core ML (not tuned llama.cpp); Apple ANE hardware; Phi has 16 experts. It drops tokens above static capacity (lossy, 10-22% of tokens). Only supports break-even logic: an accelerator helps prefill once I/O is hidden (prompt > ~600-1100 tokens in our arithmetic).
- Quality loss: Yes, small: capacity-limited token dropping, accuracy delta 0 to -1.07% on 4 tasks (Table 3), PhiMoE only.
- Ablation: Yes (Fig.17, Table 4/5): padding, grouping group size G in {4,8}, expert capacity {32,64,128}; no storage component.
- Limitations: Authors: optimisation targets long-context prefill-dominated workloads; decode follows the same pipeline, memory bound and does not amortise CPU-NPU coordination; GPU is faster; attention on NPU costs +27.6%.
- Contradicts our findings: No direct contradiction. It supports our arithmetic that accelerator wins appear only when prefill is compute-bound. Note its prefill expert imbalance figure (Fig.5) shows skew in prefill; our 125B prefill coverage is still unmeasured.

## P-0128: ProMoE: Fast MoE-based LLM Serving using Proactive Caching (Xiaoniu Song, Zihang Zhong, Rong Chen, Haibo Chen; SJTU IPADS / Zhejiang Univ.)

- Version read: arXiv:2410.22134v3 [cs.DC] 1 Sep 2025, PDF via pdftotext (128,974 chars)
- Setup: Models (Table 1): Deepseek-moe 16.4B/2.8B 28 layers 64 experts top-6 (FP16); Deepseek-v2-lite 15.7B/2.7B 64 top-6 (FP16); Qwen1.5-moe 14.3B/2.7B 60 top-4 (FP16); Qwen2-moe 57.4B/14.2B 64 top-8 (INT4); Mixtral-8x7B 46.7B/12.9B 8 top-2 (INT4). Hardware: RTX 4090 24 GB, i9-14900K, 128 GB DRAM, PCIe 4.0 32 GB/s; cache rate varied to emulate smaller GPUs. ShareGPT (and Alpaca). Batch 1 default, 1-4 tested. Codebases: HF transformers (TO baseline, improved) and llama.cpp (LO baseline layer offload, CPU computes offloaded layers); added baselines UM, static cache, LRU cache. Learned predictor with stride prefetching, chunked prefetch, early preemption, reordered inference.
- Numbers (what | value | baseline | ratio | where):
  - Abstract average speedup vs existing offloading | prefill 2.20x (up to 3.21x); decode 2.07x (up to 5.02x) | TO/UM/LO in transformers and llama.cpp | transformers 2.15/2.47 and llama.cpp 2.25/1.66 average to 2.20/2.07 (own calc, matches) | Abstract, Sec.7.2
  - vs hand-crafted caches (static, LRU) average | prefill 1.78x (up to 2.48x); decode 1.34x (up to 1.79x) | static/LRU | - | Sec.1
  - transformers codebase vs static / LRU | prefill 1.42x / 2.21x; decode 1.47x / 1.31x | - | - | Sec.7.2 Fig.10
  - llama.cpp codebase vs static / LRU | prefill 1.36x (1.75) / 2.12x (2.22); decode 1.49x (1.79) / 1.09x (1.17) | - | - | Sec.7.2 Fig.11
  - llama.cpp vs UM/LO etc. | prefill 2.25x (up to 3.21x); decode 1.66x (up to 2.07x) | - | - | Sec.7.2
  - Ablation decode | prefetch + chunked + early preempt + reordered: 1.35x over LRU base (transformers) | LRU base | 1.35 | Fig.12 text
  - Ablation prefill transformers | early preemption 1.27x, reordered inference 2.39x vs base; plain prefetching 'minimal improvement' | base | - | Sec.7.3
  - Cache-rate sweep llama.cpp QW-2 | vs LRU 1.53x prefill / 1.14x decode; vs static 1.10x / 1.27x; LO wins decode at low cache rate | - | - | Sec.7.4 Fig.16
  - Batch size, llama.cpp QW-2 | ProMoE over LRU decode 1.16x at bs1, 1.34x at larger bs | LRU | - | Sec.7.5
  - Critical-path loading time | 69.68% -> 30.96% (QW-2 prefill); decode 25.61%/29.20% vs LRU 45.52%/50.89% | - | - | Sec.7.4
- Catalog / first-pass numbers:
  - [yes] 2.20x average prefill speedup (up to 3.21x): vs offloading baselines; vs static/LRU caches it is 1.78x
  - [yes] over offloading baselines (GPU, PCIe): RTX 4090 over PCIe from DRAM
  - [yes] (first pass) 2.07x decode: up to 5.02x
  - [yes] (first pass) 1.78x / 1.34x vs caches: up to 2.48x / 1.79x
  - [yes] (first pass) 'llama.cpp CPU offload wins for Mixtral decode': authors: LO surpasses ProMoE in decode for Mixtral because 25% of experts are active; also LO wins at low cache rates
- Transfers to our regime: no. Weights live in host DRAM and stream to a 24 GB GPU; our experts live on NVMe and are computed on CPU; predictor (learned from intermediate hidden states, stride prefetch) needs per-layer lookahead and compute to overlap, but at 512-expert top-10 the same prediction accuracy would have to be much higher; llama.cpp codebase gain over LRU decode is only 1.09x. 64-expert models.
- Quality loss: None claimed (prefetching only; lossless).
- Ablation: Yes (Fig.12/13): prefetch, chunking, early preemption, reordered inference; shows prefill benefit comes from early preemption and expert reorder, not prefetch, because almost all experts are touched.
- Limitations: Authors: reduced benefit in llama.cpp (less Python overhead); LO better for Mixtral and low cache rate decode; evaluated at batch size 1-4; models up to 57B INT4.
- Contradicts our findings: No. Its prefill claim 'nearly all experts accessed' is for 64-expert models with long prompts; our decode-trace coverage of 55-66% after 200 tokens (pubtech/C.md #2-0) is a different quantity. ProMoE's own LRU thrashing observation in prefill (IDs in order) is an argument for expert-major ordering already in our idea.

## P-0129: DuoServe-MoE: Dual-Phase Expert Prefetch and Caching for LLM Inference QoS Assurance (Yuning Zhang, Grant Pinkert, Nan Yang, Yanli Li, Dong Yuan; University of Sydney)

- Version read: arXiv:2509.07379 v2 (9 Apr 2026, 100,866 chars) read in full for results; v1 (71,894 chars) read for the headline numbers
- Setup: Models: Mixtral-8x7B and 8x22B (4-bit AWQ, 8 experts top-2), Qwen3-30B-A3B (FP8, 128 experts top-8), DeepSeekMoE-16B (66 experts per paper text, 8 activated, full weights). Hardware: (1) AMD Threadripper PRO 3945WX 12-core, 128 GB RAM, RTX A5000 24 GB, Ubuntu 20.04; (2) Threadripper PRO 5955WX 16-core, 128 GB, RTX A6000 48 GB; PCIe 4.0. Datasets SQuAD and Orca-Math. Baselines: On-Demand Fetch (HF Accelerate), Layer-wise Full Prefetch (MoESys style), MoE-Infinity (MIF; OOM on 22B). Built on vLLM, CUDA two-stream prefill pipeline, MLP layer predictor trained on traces for decode (0.6 ms, ~300 MB GPU, 2.5% of dataset, collection+training <= 9 h).
- Numbers (what | value | baseline | ratio | where):
  - v2 headline | TTFT 1.78x-5.34x, end-to-end 1.42x-7.55x | ODF and LFP (and MIF where available) | - | v2 abstract and Sec.VI-B
  - v1 headline | TTFT 1.78x-3.8x, end-to-end 1.42x-7.54x | same | - | v1 Sec.1 and Sec.VI; the TTFT upper bound changed between versions
  - v2 dataset averages | end-to-end 1.72x over ODF on Orca, 1.67x on SQuAD; on 30B TTFT over ODF about 1.95-1.99x | - | - | Sec.VI-B
  - Table III predictor accuracy, Top-k (all routed experts right) | DuoServe 54.16-66.85%; MIF 36.13-44.33% | MIF | ~1.4-1.6x | Table III
  - Table III 'at least half' routed experts right | DuoServe 90.31-99.1%; MIF 76.83-84.22% | MIF | - | Table III
  - Table II peak GPU memory (LFP/ODF/MIF/DuoServe/GPU-only) | Mixtral-8x7B 4.24/3.63/15.24/3.91/25.14 GB; Qwen3-30B 1.80/1.24/7.67/1.54/30.0 GB; DeepSeekMoE-16B 2.95/1.98/11.35/2.28/32.8 GB | - | - | Table II
  - Batch size | throughput gain shrinks as batch grows (union of experts densifies) | - | - | Fig.7 text
- Catalog / first-pass numbers:
  - [yes] Two-stream pipeline overlapping expert prefetch with compute during prefill: prefill computes one expert while prefetching another; all experts of each layer are fetched
  - [partly] Prefill activates experts densely, decode sparsely: the paper asserts it; the supporting text example is Mixtral (8 experts, top-2) 'can activate all experts in a layer during prefill'; I found no measured coverage for the 128-expert Qwen3 or DeepSeekMoE in the text
  - [yes (v1)] (first pass) TTFT 1.78x-3.8x, E2E 1.42-7.54x: current v2 says TTFT 1.78-5.34x, E2E 1.42-7.55x
  - [partly] (first pass) predictor top-2 hit 54-67% (Table 3): it is Table III 'Top-k = all routed experts correct' (for Qwen3/DeepSeek that is top-8, 55-61%), table numbering III; range 54.16-66.85% correct
  - [yes] (first pass) A5000 / A6000: plus 128 GB RAM, PCIe 4.0
- Transfers to our regime: no. GPU with PCIe from host RAM; Mixtral top-2 of 8 makes 'dense prefill' trivially true; for decode, the predictor needs per-layer GPU inference and trace training (<=9 h) and only prefetches, so CPU-only NVMe streaming loses the GPU compute overlap; no disk.
- Quality loss: None claimed (prefetch/cache only, quantization is AWQ/FP8 from the model setup).
- Ablation: Limited: compares to ODF, LFP, MIF; no phase-by-phase ablation of prefill vs decode components found in text; predictor accuracy compared in Table III.
- Limitations: Authors: models with larger expert pools and sparser routing are harder to learn; throughput gain shrinks at large batch; MIF OOMs on 22B; predictor training needs offline traces.
- Contradicts our findings: No, but it does not support 'prefill touches nearly every expert' for 128+ expert models; our decode-trace coverage 55-66% after 200 tokens (proxy) leaves that open. Absolute baseline latencies are in figures (not text-extractable).

## P-0130: ssd-llm (github quantumnic/ssd-llm; Rust; 'Run 70B+ LLMs on Apple Silicon by using SSD as extended memory'; MIT license; 58 stars; created 2026-02-17, last push 2026-10-24)

- Version read: README.md raw, BENCHMARKS.md raw, CHANGELOG.md raw, src/inference/transformer.rs and moe.rs raw, repo tree and API JSON (no paper)
- Setup: Dense GGUF models (LLaMA 7B/13B/70B Q4_0 targets) with mmap + madvise, LRU layer cache, prefetcher, Metal GPU kernels; MoE support v1.10 (Mixtral-style, top-K gating, on-demand loading). BENCHMARKS.md: MacBook Apple M4 base, 16 GB, macOS 26.6, ssd-llm v1.39.0 vs llama.cpp b10180 (Metal), one model only: Qwen2-0.5B-Instruct Q4_0 (330 MB, 24 layers) that fits in RAM.
- Numbers (what | value | baseline | ratio | where):
  - llama.cpp pp512 / tg128 (Qwen2-0.5B, M4) | 2628 t/s / 156.6 t/s | - | - | BENCHMARKS.md
  - ssd-llm estimated prefill / decode (same model) | ~222 t/s / ~9.2 t/s ('Est.') | llama.cpp | 0.084 / 0.059 (own calc) | BENCHMARKS.md
  - ssd-llm cold layer load / sequential prefetch stream | 4274 MB/s / 3058 MB/s | - | - | BENCHMARKS.md
  - README benchmark table 70B Q4 8 GB budget | ~8 ms/layer layer load, Est. tok/s 'TBD' | - | - | README 'Benchmarks' (header says v0.5)
  - Layer-major batch prefill | implemented: transformer::batch_prefill / batch_prefill_lora; CHANGELOG v0.6.0 2026-02-18 'loads each transformer layer once, processes ALL prompt tokens through it' | per-token layer loads | no measurement | CHANGELOG, transformer.rs lines 124-175, 672-706
- Catalog / first-pass numbers:
  - [yes (dense)] Layer-major batch prefill, each layer's weights loaded once for all prompt tokens: implemented in code for dense layers; no prefill benchmark exists
  - [partly] (first pass) no measured tok/s ('TBD'): README table TBD; BENCHMARKS.md has a tiny-model fits-in-RAM run where ssd-llm estimates ~222 t/s prefill vs llama.cpp 2628 t/s
  - [no] (first pass) 'unclear if implemented': implemented (CHANGELOG v0.6.0 + code); MoE: moe.rs contains batch_expert_selection (union of experts for a batch) but transformer.rs does not call it in the files I read, so MoE prefill is per-token experts inside the layer-major loop (from grep of the two files; not run)
  - [yes] (first pass) 58 stars: 58 on 2026-10-29
- Transfers to our regime: no. No MoE-at-scale evidence at all; layer-major prefill for dense models is a standard idea (llama.cpp ubatch already loops tokens per layer, from memory). For MoE the useful variant is expert-major, which the repo's MoE code does not implement.
- Quality loss: none stated
- Ablation: none
- Limitations: None stated in README beyond the TBD; numbers are 'Est.'.
- Contradicts our findings: No. Provides no evidence for or against our I/O-bound prefill arithmetic.

# Family sections

## Family screen 1 #20 (co-firing experts adjacent on disk): P-0121, P-0122, P-0123
Further prior art (read at the depth stated):
- mlx-lm issue #1438 thread (github.com/ml-explore/mlx-lm/issues/1438), 32 comments, read the mbolt-related ones: independent MLX streamer author measured coalescing levers on his own reader. A1 (3 reads to 1) 1.61x I/O, +11.0%; A2 (9 to 1) 2.34x, +17.6% at 0.3 resident (Qwen3.6-35B 8-bit); gpt-oss-120b 1.29x/+9.7% (bandwidth bound ~4 MB slices); Qwen3-235B 4-bit at 0.2 resident A1 +32.3% decode (1.677 -> 2.218 tok/s). Same thread: plain LRU beats any pinned hot set (cross-topic pin-38 -6 to -7 points hit rate), cross-layer Jaccard 0.017 (no prefetch signal), three independent prefetch nulls. All consistent with our pubtech findings. Strengthens "layout pays iff latency bound; the gain is coalescing, not clique order".
- FlashMoE (arXiv 2601.17063v2, 5 Oct 2026): SSD expert offload, ML-based cache replacement (recency + frequency to a small net), up to 51% higher hit rate than LRU/LFU, 2.5x total prefill speedup over llama.cpp (4.1x over Fiddler/DAOP), desktop Ryzen + RTX 5070, 16 GB DDR5, PCIe5 NVMe 7.4 GB/s, OLMoE-1B-7B and Qwen3-30B-A3B. Read abstract, setup and eval excerpts only (partial). Cache policy prior art, no layout. Threat to cache-policy novelty, not to layout.
- Edge0 (arXiv 2609.18063v2, 17 Sep 2026): streaming 35B MoE (256 experts) from SSD on Mac mini M4 Pro 24 GB / MacBook M2 16 GB; trained next-layer 'prerouter' (routing replacement, lossy, recovered with LoRA + int4), 20.4 tok/s in 2.9 GiB vs resident 3.9 tok/s; notes that stacked layout means nine direct reads per expert (gate/up/down x scales). Read abstract, setup, tables partially. Closest published disk-streaming system on Apple hardware; uses trained prediction, not layout.
- EStream (arXiv 2609.06551, 6 Sep 2026): offline expert permutation and grouping, each group's gate/up/down in one contiguous UFS extent, vectored reads; mobile NPU prefill, 2.25-27.57x TTFT. Read abstract + layout section only (partial). Shows the contiguous-per-expert/group layout idea on flash, for prefill; threatens novelty of 'group experts contiguously on flash'.
- FloE (2505.05950): intra-expert contiguity via pairing gate columns with down rows (per search snippet; PDF downloaded, not read in full). SeqMoE (2609.12978), SSD energy analysis (2508.06978): snippets only. PowerInfer-2 neuron-cluster bundling in flash, colibri streamer: from memory / named in mbolt docs, not opened.
What is new in OUR variant: nothing new in the idea. mbolt (Jul 2026) is the same idea on a 512-expert top-10 model, and LLM in a flash is the neuron-level ancestor (including the negative co-activation result). Our contribution is a measurement: leave-one-out, after an LRU cache at 8/16/32 GB, miss run length only 1.01-1.30 and tok/s gain 1.045x/1.003x/1.000x; consistent with mbolt's 1.05x over its prefetcher and the MLX author's 'profile-free coalescing carries ~90%'. Closest papers, read in full: mbolt docs/phase reports, LLM in a flash.

## Family screen 1 #21 (draft K tokens, verify in one batched pass): P-0125
Further prior art:
- SP-MoE (arXiv 2510.10302v2, Nov 2025): SD-aware expert offloading, draft/target correspondence for prefetch, cutoff-layer policy, batched I/O; 1.07x-3.5x TPOT over SOTA; RTX 4090-class, PCIe 4.0 (read abstract and setup only). Same family as Specter, GPU.
- SpecMoEOff (Specter's primary baseline, microbatch speculative offloading): not located by search, cited only in Specter v3; not read.
- DraftExpert (2607.24434), SPICE (2608.21240), AcceptMoE (2608.02989, 2.06x on RTX 5090, expert eligibility conditioned on cache residency): search-result summaries only, not opened. AcceptMoE's cache-aware acceptance is the only one that touches the bytes rather than the overlap.
- Edge0 (above) predicts next-layer routing with a trained head instead of a draft.
What is new in OUR variant: published work (MoE-SpeQ/Specter, SP-MoE) is GPU/PCIe prefetch overlap; our test asks whether verify-union reads fewer bytes per accepted token on a CPU/NVMe system. Result: bytes ratio 0.97-1.0 at K=1 and >1 beyond (pubtech/C.md); Specter Table 6 independently shows prefetch accuracy matters only 1.34x even on GPU. No paper reads NVMe-to-CPU with speculation. Novelty: open as a negative measurement, known as a technique.

## Family screen 2 #0 (NPU/accelerator for prefill; layer- or expert-major prefill): P-0127, P-0128, P-0129, P-0130
Further prior art:
- EStream (above): MoE prefill on mobile NPU with experts paged from UFS flash through a fixed arena; states 'one request touches most experts, more than a phone can hold' and 2.25-27.57x pure-prefill TTFT speedup. Closest to an NPU + flash prefill design (partial read). Its premise about expert coverage in prefill is stated for mobile-sized MoEs; we have no 125B measurement.
- FlashMoE prefill 2.5x vs llama.cpp (above) includes expert loading at prefill.
- ProMoE ablation: prefetching alone gives 'minimal improvement' in prefill because nearly all experts are accessed; gains come from reorder/early preemption (read).
- llama.cpp ubatch processing and FluxMoE: from memory / catalog, not re-read.
What is new in OUR variant: published evidence for 'prefill touches nearly all experts' is Mixtral-class (8 experts) or 64-expert models with long prompts; evidence for NPU prefill is unified-memory Apple Core ML with whole model in RAM. A disk-bound analysis (I/O 22-44 s vs compute 19 s for 500 tokens, break-even 575-1150 tokens) has no published counterpart I found. Novelty: variant/open, unmeasured on 125B prefill routing.

# Summary of corrections to the first pass
1. P-0129 DuoServe: arXiv v2 (9 Apr 2026) raises TTFT upper bound from 3.8x to 5.34x and E2E 7.54x to 7.55x; the 54-67% predictor figure is Table III 'Top-k' (all routed experts right, top-8 for Qwen3/DeepSeek), not top-2; 'at least half' hit is 90-99% (MIF 77-84%).
2. P-0130 ssd-llm: layer-major batch prefill IS implemented (v0.6.0, transformer::batch_prefill) for dense layers; it is not benchmarked; MoE union selection exists but is not wired in the files read; BENCHMARKS.md has only a 330 MB fits-in-RAM model where ssd-llm's estimated prefill is 12x slower than llama.cpp.
3. P-0121 mbolt: first pass left out that at the 32 GB squeeze the prefetcher alone is 1.52x and the layout adds only 1.077x; interleave-without-prefetcher is 0.98x; reorder-over-prefetcher is 1.053x; the 1.55x/1.63x are versus stock mmap. The MLX cross-check numbers are now verified in the thread but measure expert-slice coalescing on a different reader, not clique reorder. Squeeze labels in the author's text are inconsistent (24 vs 32 GB 'tighter').
4. P-0125: confirmed 2.34x is in v1 only (absent from v2/v3 full text); former 39k unread chars now read; v1 text numbers give 3.29x vs MO-SC and 2.16x vs MO-SM on Phi-MoE, so 2.34x is not derivable from text.
5. P-0127 NPUMoE: arXiv stamp is 20 Apr 2026 (not 22); the 86/94% overlap is on a 16-expert model (top-8 is half the pool); lossy token dropping 10-22% with accuracy delta 0 to -1.07% was not mentioned; which model ran on the 64 GB M2 Max is not stated.
6. P-0122: abstract says up to 4x/20x; Table 3 computes to 4.76x CPU and 26.4x GPU for OPT 6.7B.
7. P-0123: confirmed; additionally Table 1 caption PCIe 24 GB/s vs text 32 GB/s; Ollama baseline is a poor llama.cpp proxy.

# What changes what we should believe about our own measurements
- Nothing contradicts them. Our 1.045x reorder gain after cache is independently echoed by mbolt (1.053x over its prefetcher, 1.077x at 32 GB) and by mbolt's own simulator overshoot (replay 1.23x vs e2e 1.05x).
- The large lever in that family is per-expert contiguity (interleave) plus explicit reads (2.23x I/O, 1.18x e2e over prefetcher at 33 GB/24 GB RAM); third-party MLX numbers (+9.7% to +32%) depend on slice size and resident fraction, and the +32% case is at 0.2 resident where cold reads are nearly the whole token, i.e. our deeper regime (69 GB vs 8-32 GB) is closest to that corner. That is a reason to re-check our family #19 (repack/coalesce) separately from #20 (reorder): #19 may deserve a higher prior than #20.
