# Deep re-check, batch P01a (raw full text, no digests)

Scratch: <scratch> All 13 rows read from raw PDF text via curl + pdftotext (no digests). Unread parts are flagged per row (figure bars that pdftotext scrambles).

Distinct papers read: MoE-Spec (P-0002/P-0011), EcoSpec (P-0003), DraftExpert (P-0004), SpecMoEOff (P-0005/P-0014), MoE-SpeQ->Specter and SP-MoE (P-0006/P-0013), S2-MoE (P-0007), SpecExec (P-0008), Cascade (P-0010), AcceptMoE (P-0012), EStream (P-0016).


## P-0002: MoE-Spec: Expert Budgeting for Efficient Speculative Decoding. Bradley McDanel (Franklin & Marshall; Meta Reality Labs), Steven Li, Sruthikesh Surineni, Harshit Khaitan (Meta Reality Labs). Preprint, 19 Feb 2026 (arXiv header: 17 Feb 2026).

- ID read: arXiv 2602.16052v1 (PDF, pdftotext raw, 964 lines; read fully except appendix figures)
- Raw text read: True
- Setup: OLMoE-1B-7B (7B/1B active, 64 experts, top-8), Qwen3-30B-A3B (128 experts, top-8), Mixtral-8x7B (8 experts, top-2). Drafts: EAGLE-3 for OLMoE/Qwen3, EAGLE-1 for Mixtral. NVIDIA A100 80GB (Mixtral on 2 GPUs), FP16, PyTorch 2.1/CUDA 12.1, ALL experts GPU-resident (App. A Table 3). Single request, draft tree 63 tokens (sweep 3-511), 80 samples/benchmark, max 1024 new tokens, T=0 and T=1 (5 seeds). Benchmarks GSM8K, MATH500, HumanEval, MBPP, CNN/DM. Baselines AR, EAGLE (named per model). Budget B chosen to maximise speed while quality stays within 1 std of EAGLE (budgets 8-56 OLMoE, 16-64 Qwen3, 2-7 Mixtral).

- Headline numbers (what | value | baseline | ratio | where):
  - Mixtral MBPP T=0 speedup over AR | MoE-Spec 2.3x | EAGLE 1.7x | 1.35 (computed 2.3/1.7) | Table 1, p.5
  - Qwen3-30B GSM8K T=0 speedup over AR | MoE-Spec 2.4x | EAGLE-3 1.9x | 1.26 | Table 1, p.5
  - Qwen3-30B CNN/DM T=0 | 1.9x | EAGLE-3 1.5x | 1.27 | Table 1
  - OLMoE MATH500 T=0 | 2.5x | EAGLE-3 2.1x | 1.19 | Table 1
  - Mean relative speedup gain over EAGLE, per model | Mixtral 27%, Qwen3 16%, OLMoE 6% | EAGLE/EAGLE-3 | 1.27 / 1.16 / 1.06 (as stated) | Sec 4.2 text, p.4
  - Mean acceptance length change | -1.4% average | EAGLE | 0.986 | Sec 4.2
  - Unique experts per layer at tree 31 | MoE-Spec 29 vs EAGLE-3 39 (OLMoE, B=32) | EAGLE-3 | 0.74 | Sec 4.3, Fig 4a
  - OLMoE speedup at tree 255, B=32 | 2.1x | EAGLE-3 1.85x (peak 1.95x at tree 15-31) | 1.14 | Sec 4.3/Fig 4b-c
  - Quality at max speedup, T=1, average of 5 benchmarks (relative to AR) | Mixtral ~78%, Qwen3 ~95% at 2.0x, OLMoE steepest decline | AR=100% | n/a | Sec 4.2, Fig 3
  - Routing mass in top-32 of 64 experts, tree 63 (OLMoE) | 93% (excluded experts get 7%) |  |  | Intro, Fig 2c, Sec 4.3
  - OLMoE 127-token tree activates | 54 of 64 experts per layer |  | 0.84 of experts | Intro p.1
- Catalog / first-pass numbers checked:
  - [yes] Verification loads the union of experts over the draft tree -> Eq. 3 and Sec 3.1: target loads union of top-k over the tree
  - [yes] It caps experts per layer by aggregated router probability -> Router-based ranking s_i = sum_t g_i(h_t), top-B shortlist (Eq. 5); static and oracle rankings also studied
  - [yes] Reported 10-30% throughput gain -> Abstract and Sec 4.2: per-model mean relative gains 6% / 16% / 27%; the 10-30% is vs EAGLE-3 (EAGLE-1 on Mixtral), not vs AR; on GPU with all experts resident
  - [yes] First-pass: Table 1 Mixtral MBPP T=0 2.3x vs 1.7x; Qwen3 GSM8K 2.4x vs 1.9x -> Table 1 as above
  - [yes] First-pass: acceptance length drops about 1.4% -> Sec 4.2
  - [yes] First-pass: 63-token trees, A100 80GB (2 for Mixtral), FP16 -> Table 2/3 App. A
- Transfers to our regime: no. Why: GPU-resident experts: the saving is HBM bandwidth over the verification union, not an NVMe read. Draft is a trained EAGLE head. Lossy truncation/substitution (we require exact routing). Trees of 63 tokens on 64-128-expert models; our chains are k=8 on 512 experts top-10. Their tail-dropping idea (93% of mass in top half of experts) is a lossy route-restriction lever which does match the family of 'restrict the union' but without any cache term. Authors note batched serving would see smaller gains.
- Quality loss: Lossy by construction (experts outside the shortlist are dropped or substituted). Table 1 picks the budget within 1 std of EAGLE quality, so table accuracies are close (e.g. Qwen3 HumanEval T=0 78 vs 82 AR, 82 EAGLE-3; Mixtral HumanEval 27 vs 20 AR, noise-level). Tighter budgets: Fig 3 quality relative to AR falls to ~78% (Mixtral) at maximum speedup; Static ranking gives near-zero GSM8K/MATH500 accuracy at B<=16 (Fig 5a); Router ranking tracks an oracle; Oracle reaches near-AR quality at B=24 (37.5% of experts).
- Ablation: Yes, Sec 4.4: ranking method (static/router/oracle) and coverage policy (truncation vs substitution perform comparably); tree-size sweep; reconstruction error (Fig 6: oracle matches router with 25% fewer experts). No ablation against any cache/offload.
- Limitations (authors): Authors (Sec 6): budget hyperparameter must be chosen per deployment; router-based ranking adds 2-3% overhead and may not pay for very small trees; single-request A100 only, batching may give smaller gains; softmax routing only (DeepSeek-V3 sigmoid may need adaptation); no per-layer budget; no co-activation model.
- Contradicts/supports our findings: No. Fig 4a shows the union approaches all experts for big trees (EAGLE-3 54 of 64 at tree 255), consistent with our measured union 29.1 of 64 (45%) at k=8 on the 128-expert trace. Their finding that EAGLE speedup peaks then declines with tree size (1.95x -> 1.85x) matches our 'union cost eats acceptance gain'. Nothing about an LRU cache or disk.


## P-0003: Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for Mixture-of-Experts (EcoSpec). Jincheng Xie*, Runheng Liu*, Heyan Huang, Yawen Ling, Hanbin Dai, Yu Zheng, Wen Hu (Tsinghua, BIT, JD AI Infra, SWJTU, Xidian). Preprint for ICML 2026 format, 15 Jul 2026.

- ID read: arXiv 2607.12696v1 (PDF raw, 1263 lines; read fully incl. appendices A-D)
- Raw text read: True
- Setup: DeepSeek-V3.1 (671B/37B active, top-8, MTP baseline), Qwen3-235B-A22B (235B/22B, top-8, EAGLE-3), GPT-OSS-120B (120B/5.1B, top-4, EAGLE-3). 8x NVIDIA H200, HuggingFace Transformers research prototype (not an optimised server), batch 1 main (batch 1/2/4 in App. A, 8 in Table 8), verification budget gamma=4 tokens, draft 3 steps top-2. Seven benchmarks (GSM8K, HumanEval, AIME25, Math500, AMC22-24, MTBench, MMStar), T=0 and T=1. Predictor: Qwen3-0.6B (Qwen3, GPT-OSS) / DeepSeek-R1-Distill-Qwen-1.5B (DeepSeek) fine-tuned on offline routing traces of the same seven datasets, random 80/20 split, ~4 ms/step. No offloading at all; costs are HBM reads.

- Headline numbers (what | value | baseline | ratio | where):
  - Qwen3-235B avg speedup over AR, T=0 | EcoSpec 1.36x | EAGLE-3 1.22x | 1.115 (1.36/1.22) | Table 1 p.7
  - GPT-OSS-120B avg, T=0 | 1.31x | EAGLE-3 1.14x | 1.149 | Table 1
  - DeepSeek-V3.1 avg, T=0 | 1.15x | MTP 1.10x | 1.045 | Table 1
  - Qwen3-235B MTBench T=0 (max cell) | 1.62x | EAGLE-3 1.46x | 1.11 | Table 1
  - Max cells quoted in intro | GPT-OSS 1.50x (MMStar T=1), DeepSeek 1.47x (AMC T=1) | EAGLE-3 1.31x / MTP 1.44x | 1.15 / 1.02 | Table 1, Sec 1
  - Unique experts per layer per verify step, Qwen3 T=0 | 20.5 | EAGLE-3 23.7 (AR 8) | 0.865 | Table 1/3
  - Estimated HBM read per verify step (Qwen3/GPT-OSS/DeepSeek) | 88.1 / 5.5 / 96.8 GB | 99.3 / 6.0 / 97.3 GB | 0.887 / 0.917 / 0.995 (11.3% / 8.0% / 1.0% less) | Table 2 (estimates from footprints, not measured)
  - Batch 4 average speedup | Qwen3 1.09x, GPT-OSS 1.06x | EAGLE-3 1.00x / 1.00x | 1.09 / 1.06 | App A, Tables 6-7
  - Batch 8 throughput vs AR at batch 8 | 0.69x (Qwen3), 0.65x (GPT-OSS), same for both methods | AR |  | Table 8
  - Verification time vs AR step, batch 1: rho = T_AR/T_verify | Qwen3 EcoSpec 0.67 (EAGLE-3 0.59); GPT-OSS 0.90 (0.72) |  | verify costs 1.5x and 1.1x an AR step | Table 8
  - Ablation global-cost vs marginal-cost (Qwen3 GSM8K) | 1.39x | 1.28x | 1.09 | Table 4
- Catalog / first-pass numbers checked:
  - [yes] Charges the marginal expert-loading cost in draft selection -> Score S(t)=P(t)/(dCost+eps), dCost = new predicted experts not already in buffer B (Eq. 8-9)
  - [partly] Up to 1.62x speedup -> 1.62x is the single best cell (Qwen3-235B MTBench, T=0, vs AR); the baseline speculative method on that cell already gives 1.46x; averages 1.15-1.36x; gain over the SD baseline is +0.05 to +0.17x on average
  - [yes] First-pass: DeepSeek 1.10->1.15, Qwen3 1.22->1.36, GPT-OSS 1.14->1.31 -> Table 1
  - [yes] First-pass: batch 8 below AR (0.69x/0.65x) -> Table 8 (identical for both methods; it is the AR-normalised throughput at batch 8)
  - [partly] First-pass: predictor trained and tested on same seven benchmarks -> Traces collected on the seven datasets, random 80/20 sample split (App B.1); predictor accuracy is reported on the held-out 20%, but it is in-distribution; whether the 80 percent training prompts overlap the end-to-end test prompts is not stated
  - [yes] First-pass: memory-traffic reductions are estimates -> Table 2 'Estimated HBM read traffic'
- Transfers to our regime: no. Why: HBM-bound 8xH200 with experts resident; no offloading, no SSD, no CPU. Needs a trained per-model predictor and a trained draft (EAGLE-3/MTP). Gains come from picking 4 verified nodes with overlapping experts; with gamma=4 the union is only 20.5 experts/layer. In our regime the extra cost term would be a disk read, which the paper never models; the draft-selection idea (choose tree nodes whose predicted experts are already resident) is transferable in principle, the numbers are not. Batch >1 shows the gain shrinks and goes negative (0.65-0.69x at batch 8).
- Quality loss: None claimed (verification rule unchanged, lossless draft-tree selection); acceptance length drops slightly (2.41 -> 2.32 Qwen3).
- Ablation: Table 4: static global-cost vs dynamic marginal-cost scoring; Table 9: predictor accuracy 50%/60%/converged (1.17/1.29/1.39x GSM8K Qwen3); Table 10: oracle vs predicted expert sets are identical (20.4 vs 20.5 experts). No cache/offload ablation.
- Limitations (authors): No dedicated limitations section. Prototype on HF Transformers; DeepSeek-V3.1 gains small because routing is balanced (App D); speedups fall with batch.
- Contradicts/supports our findings: Internal inconsistency: Table 3 latencies (AR 0.490 s/token, EAGLE-3 0.840 s/step with 2.41 accepted tokens) imply 0.49*2.41/0.84 = 1.41x for EAGLE-3 and 1.53x for EcoSpec on Qwen3, but the same table and Table 1 report 1.22x and 1.36x; GPT-OSS: 1.03x/1.17x implied vs 1.14x/1.31x reported; so the AR time or the speedup basis is not the one stated. Does not contradict our measurements; Table 8 (verification of 4 tokens costs 1.1-1.7x an AR step even on H200) supports 'verify is not free on MoE'.


## P-0004: DraftExpert: Expansion-Aware Self-Speculative Decoding for End-Device MoE Inference. Dengke Han. arXiv preprint, 27 Jul 2026.

- ID read: arXiv 2607.24434v1 (PDF raw, 673 lines; read fully)
- Raw text read: True
- Setup: DeepSeek-V2-Lite (15.7B total, 2.45B active, 27 layers, 64 routed + 2 shared, top-6; routed pool 14.4B = 26.8 GiB in BF16) and Moonlight-16B-A3B (same shape). CG: RTX 4090 24GB, BF16 PyTorch 2.11, routed experts in host RAM copied on demand over PCIe. MN: Xiaomi MIX Flip 2, Snapdragon 8 Elite, Hexagon HTP v81 NPU, llama.cpp Q4_0, experts in Flash staged into shared DRAM (storage bandwidth not stated). Single user, decode-stage TPS after prefill, prompts chat/GSM8K/MBPP/summarisation/instruction following. Baselines: AR with expert offload, best training-free shared+top-r exact self-draft (r swept 1-3). Method needs one trained draft expert per layer (self-distillation from frozen target; training cost/data volume not stated).

- Headline numbers (what | value | baseline | ratio | where):
  - Decode TPS DS-V2-Lite CG | 2.99 (DraftExpert) | AR 2.19; top-r 1.19 | 1.365 vs AR; top-r is 0.54x of AR | Table 2 p.4
  - DS-V2-Lite MN (phone Flash->NPU) | 15.47 | AR 10.18; top-r 4.80 | 1.52; top-r 0.47x of AR | Table 2
  - Moonlight CG | 2.50 | AR 1.94; top-r 1.01 | 1.29; top-r 0.52x | Table 2
  - Moonlight MN | 13.69 | AR 8.50; top-r 3.82 | 1.61; top-r 0.45x | Table 2
  - Average over the 4 cells | 1.45x | AR offload | (1.365+1.52+1.29+1.61)/4 = 1.445 | Table 2
  - Resident-memory AR reference, DS Q4_0 on phone | 14.36 TPS | Flash-staged AR 10.18 | DraftExpert 15.47 is 1.08x above all-resident AR | Setup table p.4
  - Draft acceptance / unique routed experts touched at K=4 (DS) | 85.0% acc, 88 unique experts | top-1 22.5% acc, 84; top-2 31.4%, 183; top-3 45.9%, 244 |  | Table 3
  - Verification cost normalised to 1 AR token (DS): fixed K -> conf+expansion -> +prefetch | CG 2.35x -> 1.48x -> 1.25x; MN 3.10x -> 1.70x -> 1.35x | fixed-K verify | fixed-K to final: 0.53 (CG), 0.44 (MN); first step alone -37% / -45% | Table 4
  - Prefetch hit / waste | 88% / 12% (DS), 86% / 14% (ML) |  |  | Table 4
  - Verification union growth | 59% of all routed layer-experts touched at K=16; MoE verify cost 2.4x (K=?) to 5.8x vs dense ~1.0x | one-token decode |  | Fig 4-5 (read from figure text)
  - Expert-load share of decode time | 87% (CG), 49% (MN) |  |  | Fig 1b
- Catalog / first-pass numbers checked:
  - [yes] States that verifying a multi-token block activates the union of experts, so verification is not one target step -> Abstract, Condition 2, Fig 4-5
  - [yes] Uses it for dynamic draft truncation and prefetching -> Confidence-expansion truncation (Eq. 12-14) and draft-router prefetch (Alg 1)
  - [yes] First-pass: 1.36x/1.52x/1.29x/1.61x, avg 1.45x -> Table 2
  - [yes] First-pass: acceptance 83-87%, prefetch hit 86-88% -> Table 3: 83.0-85.0% acceptance at K=4; Table 5 acc 84.5-87.1%, hit 86-88%
  - [partly] First-pass: verification cost ~40% lower than fixed-length -> Table 4: 37% (CG) and 45% (MN) lower from truncation alone, 47% and 56% with prefetch
  - [yes] First-pass: naive shared+top-r drafting slower than AR (1.19 vs 2.19) -> Table 2, all four cells 0.45-0.54x of AR
- Transfers to our regime: partly. Why: Direction supports us (union growth, naive speculation loses to AR under offload). Not our regime: model is 16B (not 125B/512 experts), top-6 of 64; the offloaded tier is host RAM over PCIe (CG) or phone flash with an NPU (MN); needs a trained per-layer draft expert resident on the accelerator (extra memory + training); baseline AR cache policy is not stated, so gain vs an LRU best path is unknown. Batch 1 and exactness do match.
- Quality loss: None (exact: every emitted token verified by the frozen target).
- Ablation: Yes: draft cost (Table 3), verify-side truncation and prefetch (Table 4), loss terms (Table 5: residual lifts acceptance 47.7->79.5%, router loss lifts prefetch hit 73->88%). Cache vs layout isolated: no.
- Limitations (authors): No limitation section; no code link; training cost not reported; only two 16B models.
- Contradicts/supports our findings: No. Fig 4/5 (verification cost grows with K, 59% of layer-experts at K=16) agree with our union growth. Their win requires a learned draft + prefetch, i.e. not a bytes-only effect, consistent with our 'byte side alone refuted'.


## P-0005: Accelerating Mixture-of-Experts Inference by Hiding Offloading Latency with Speculative Decoding (SpecMoEOff). Zhibin Wang, Zhonghui Zhang, Yuhang Zhou, Zibo Wang, Mo Zhou, Peng Jiang, Weilin Cai, Chengying Huan, Rong Gu, Sheng Zhong, Chen Tian (Nanjing Univ., HKUST-GZ).

- ID read: arXiv 2508.21706v2 (31 Oct 2025; first pass read v1 29 Aug 2025). PDF raw. Bar values of Fig 8/9 are not recoverable from pdftotext (scrambled), so per-config throughputs are UNREAD; only text-stated numbers used.
- Raw text read: True
- Setup: Mixtral-8x7B FP16 (87 GB: experts 84 GB), EAGLE draft (<2 GB), 8 experts top-2. Env 1: A30 (933 GB/s HBM, CPU-GPU 25 GB/s) + Intel Gold 6426Y (2.43 TFLOPS, 357 GB/s) with 250 GB RAM; Env 2: RTX 4090D (1008 GB/s, 23 GB/s) + Gold 5418Y (1.45 TFLOPS, 197 GB/s) with 190 GB RAM. Implemented on SGLang (20k+ LoC), KV cache and attention on CPU. Throughput-oriented offline serving: global batches 824 (A30) / 515 (4090D) in the roofline analysis (APPS, 566 input + 1024 output tokens). Datasets APPS (mean input 566) and CNN/DailyMail (1006), outputs 128-1024. Baselines DeepSpeed-ZeRO-Inference 0.17.4, MoE-Lightning, and SpecMoEOff-w/o-sd.

- Headline numbers (what | value | baseline | ratio | where):
  - Throughput vs DeepSpeed-ZeRO-Inference | 11.3-27.5x, avg 17.7x | DeepSpeed-ZeRO |  | Sec 6.2 text
  - Throughput vs MoE-Lightning | 1.6-2.9x, avg 2.1x | MoE-Lightning |  | Sec 6.2 text
  - Headline in abstract / intro / conclusion | 'up to 2.5x decode throughput'; Intro bullet: '2.5x average speedup'; Conclusion '2.5x' | MoE-Lightning | inconsistent with Sec 6.2 avg 2.1x | Abstract, Sec 1, Sec 7
  - Gain from SD alone (SpecMoEOff vs w/o-sd) | 1.53x APPS, 1.45x CNN/DM | SpecMoEOff-w/o-sd (already beats MoE-Lightning) |  | Sec 6.2
  - GPU utilisation of batch-1 offloaders (their roofline) | 0.76% (Mixtral decode, A30); MoE-Lightning 3.13% |  |  | Sec 1
  - Dynamic draft length gain | 2% (input 512, output 128) | fixed best draft length | 1.02 | Sec 6.3
  - Estimator error | iteration 9.7%; components 4-27% |  |  | Table 3
- Catalog / first-pass numbers checked:
  - [yes] Speculative decoding to raise GPU utilization under MoE offloading -> Core claim
  - [partly] One loaded expert serves a few inputs, so transfer is the bottleneck -> This is the paper's description of batch-1 offloaders (0.76% GPU utilisation). The method fixes it by processing hundreds of sequences per batch (824/515) with draft tokens enlarging per-expert work further; it is a large-batch throughput system, not a batch-1 latency system
  - [yes] First-pass: up to 2.5x vs MoE-Lightning (abstract); 1.6-2.9x avg 2.1x -> Sec 6.2; but the paper's own intro/conclusion claim 2.5x as average
  - [yes] First-pass: 11.3-27.5x vs DeepSpeed -> Sec 6.2
  - [yes] First-pass: SD adds 1.53x / 1.45x -> Sec 6.2
  - [yes] First-pass: A30 + Xeon 6426Y 250GB, 4090D + Xeon 5418Y 190GB, batches 515-824 -> Table 1 and Sec 3.3 (batches are the roofline-analysis global batches)
- Transfers to our regime: no. Why: Offline throughput regime: hundreds of sequences per batch, 190-250 GB host RAM for KV cache, GPU compute, CPU attention. At batch 1 extra draft tokens come from the same sequence and add experts instead of reusing loaded ones. Also part of the headline is memory management (w/o-sd already beats MoE-Lightning), and SD adds only 1.45-1.53x.
- Quality loss: None (standard lossless speculative decoding).
- Ablation: Draft length sweep (Fig 10), input/output length (Fig 11), CPU/GPU memory (Fig 12), iteration breakdown (Fig 13); SpecMoEOff-w/o-sd isolates SD (1.45-1.53x). No cache-vs-layout-vs-prefetch isolation.
- Limitations (authors): No limitation section. CPU chunked attention becomes the bottleneck as draft tokens increase (Fig 13); evaluated only on Mixtral-8x7B.
- Contradicts/supports our findings: No. Consistent with our view that SD helps only when per-expert work is amplified by large batch.


## P-0006: Two papers in one row. (a) MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts (v1, 18 Nov 2025; Wenfeng Wang, Jiacheng Liu, Xiaofeng Hou, Xinfeng Xia, Peng Tang, Mingxuan Zhang, Chao Li, Minyi Guo; SJTU/HKUST). The same arXiv id was RETITLED 'Specter: Fast MoE Inference via Quantization-Enabled Multi-Step Speculative Offloading' in v2 (20 Oct 2026) and v3 (22 Oct 2026). (b) SP-MoE: Speculative Decoding and Prefetching for Accelerating MoE-based Model Inference (Liangkun Chen, Zijian Wen, Tian Wu, Xiaoxi Zhang, Chuan Wu; Sun Yat-sen Univ., HKU), v1 11 Oct 2025, v2 6 Nov 2025.

- ID read: arXiv 2511.14102v1 and v3 (raw PDF, v2 also downloaded); arXiv 2510.10302v2 (raw PDF). Some figure bars not extractable; SP-MoE Table 3 column order is ambiguous in the extraction (only the DeepSeek averages are stated in text and used).
- Raw text read: True
- Setup: MoE-SpeQ/Specter: Phi-3.5-MoE (41.9B, 16 experts top-2, 78 GB), Qwen1.5-MoE (14.3B, 60+shared, top-4, 29 GB), DeepSeek-V2-Lite (15.7B, 64 top-6, 34 GB); FP16 target, GPTQ INT4 draft of the same model (shares non-expert weights and KV cache); one A100 40GB, 24-core Xeon Silver 4310, 256 GB RAM, PCIe 4.0 x16 (31.5 GB/s); HF Transformers; baselines Transformers, Mixtral-Offloading (SC = same cache, SM = same memory), vLLM/llama.cpp (v3), SpecMoEOff (v3). Single request. SP-MoE: Mixtral-8x7B+Mistral draft, Phi-3.5-MoE+Phi-mini-MoE, DeepSeek-Lite+DeepSeek-Lite-AWQ; Env1 RTX 3090 24GB + Xeon Gold 6348 512 GB, Env2 RTX 4090 24GB + Gold 6338N 1 TB, Env3 A100 40GB + Platinum 8358P 1 TB, PCIe 4.0 x16; batch 1, output length fixed at 100 tokens, greedy, ONE draft token per iteration by default (draft length swept 2-16 in Fig 13); datasets HumanEval, BigBench, WikiText-103, MMLU-Pro; all baselines (Mixtral-Offloading, MoE-Infinity, AdapMoE) are run WITH SD.

- Headline numbers (what | value | baseline | ratio | where):
  - MoE-SpeQ v1 abstract | 'at most 2.34x speedup over the state-of-the-art offloading framework' (Phi-MoE) |  |  | v1 abstract
  - MoE-SpeQ v1 Sec 4.2 Phi-MoE avg TPOT (high-memory) | 163.1 ms | Mixtral-Offloading-SC 536.7 ms; -SM 351.8 ms; Transformers 960.5 ms | 3.29x vs SC; 2.16x vs SM; 5.89x vs Transformers | v1 Sec 4.2
  - MoE-SpeQ v1 Qwen / DeepSeek | 74.1 ms/token Qwen | 2.5x over SC, 3.5x over Transformers; DeepSeek 4.8x over Transformers | as stated | v1 Sec 4.2
  - MoE-SpeQ v1 per-step expert coverage in cache (Table 3), DS-V2-Lite | speculative prefetch 99.85 / 98.62 / 96.25% at 16/24/32 GB cache | LRU 29.2 / 50.94 / 76.56%; LRU(scaled slots) 61.13 / 76.56 / 95.77% |  | v1 Table 3 (metric = fraction of required experts already cached at each decoding step AFTER prefetch, not demand hit rate)
  - MoE-SpeQ v1 ablation DS-V2-Lite | full 13.02; w/o async prefetch 12.37; w/o fused draft kernel 8.88; w/o both 8.29 |  | 0.95 / 0.68 / 0.64 of full | v1 Table 4
  - MoE-SpeQ acceptance | over 90% (INT4 draft of same model; up to 99% on DS-V2-Lite in v3) | EAGLE ~80% |  | v1 Sec 4.5, v3 Sec 3.2.2
  - Specter v3 headline | avg 1.50x (up to 1.88x) TPOT speedup | SpecMoEOff |  | v3 abstract, Sec 4.2.1
  - Specter v3 Phi-3.5-MoE avg TPOT | 163.3 ms | SpecMoEOff 292.1; MO-SM 295.1; vLLM 297.5; Transformers 960.6 | 1.79x / 1.81x / 1.82x / 5.88x | v3 Sec 4.2.1
  - Specter v3 K=1 to 16 expert demand DS-V2-Lite | 6.0 -> 31.7 experts per layer (5.3x, nearly half of 64) |  |  | v3 Sec 2.3, Fig 4
  - SP-MoE avg TPOT reduction | 34% (29-43%) vs Mixtral-Offloading; 19% (14-25%) vs MoE-Infinity; 12% (8-17%) vs AdapMoE | each baseline with SD | 1.52x / 1.23x / 1.14x | Sec 5.1, Fig 9
  - SP-MoE Fig 9 caption | avg 1.35x, peak 1.75x (HumanEval, 3090, vs Mixtral-Offloading), min 1.12x vs AdapMoE |  |  | Fig 9
  - SP-MoE Fig 10 caption / abstract | avg 2.3x, peak 3.5x (DeepSeek-Lite, A100, vs Mixtral-Offloading), min 1.1x (Phi, 3090, vs AdapMoE); abstract/conclusion say min 1.07x (DeepSeek, A100, vs AdapMoE) |  | min is stated inconsistently (1.1x vs 1.07x, different cells) | Fig 10, Sec 8
  - SP-MoE expert hit rate, DeepSeek model | SP-MoE avg 40.06% (36.85-42.20) | AdapMoE 21.85, Mixtral-Offloading 17.67, MoE-Infinity 16.73 | 1.83x vs AdapMoE | Sec 5.2, Table 3
  - SP-MoE ablation vs SD+offload baseline (draft length 1) | vanilla prefetch 1.68x/1.39x/1.96x; + batched I/O 1.80x/1.59x/1.96x (Mixtral/Phi/DeepSeek) | baseline TPOT 1.40 / 0.48 / 0.32 s |  | Sec 5.6, Fig 12
- Catalog / first-pass numbers checked:
  - [yes] SP-MoE prefetches experts for multi-token verification -> Drafting-stage prefetch with draft attention output into target gate; cutoff-layer policy
  - [yes] MoE-SpeQ uses a quantized draft to prefetch experts -> INT4 GPTQ copy of the target
  - [partly] Up to 2.34x over an offloading framework on Phi-MoE -> True of v1's abstract only. v1 body gives 3.29x (vs SC) / 2.16x (vs SM) for the same cell; the paper was rewritten (v2/v3, retitled Specter) with new headline 1.50x avg/1.88x max vs SpecMoEOff
  - [yes] First-pass: SP-MoE abstract 1.07-3.5x; avg 34/19/12%; captions 1.35x and 2.3x -> as listed; note baselines run with SD too
  - [partly] First-pass: prefetch hit rate >96% (MoE-SpeQ Table 3) -> It is per-step cache coverage after prefetch, not a hit rate for demand loads; true demand hit rates in SP-MoE are 17-56% (Table 3)
  - [yes] First-pass: MoE-SpeQ last 8k chars not read -> now read; ablation shows async prefetch contributes only 5%, the fused INT4 draft kernel 32%
- Transfers to our regime: no. Why: Both papers: experts in host RAM over PCIe 4.0 to a 24-40 GB GPU, not SSD to CPU. Draft is a second model (Mistral-7B, quantised copy) needing GPU memory. SP-MoE baselines are re-implemented offloaders with SD; speedups are vs weak caches (hit rates 15-20%), SP-MoE greedy chain draft only. MoE-SpeQ is lossless but its cache is 16-32 GB for 15-42B models (~50-100% of experts), unlike our 4%-12% cache share. The prefetch lever is the part relevant to us; its isolated ablation in MoE-SpeQ is small (5%).
- Quality loss: None for both (lossless draft-then-verify, greedy/standard rejection).
- Ablation: MoE-SpeQ Table 4 (async prefetch, fused kernel) and Table 3 (caching strategies). SP-MoE Fig 11-14: GPU memory sweep, ablation of vanilla prefetch / worker prefetch / batched I/O, draft length, cutoff layer (U-shaped for Mixtral/Phi). No SD-off vs SD-on isolation for SP-MoE.
- Limitations (authors): SP-MoE (Sec 7): batch size 1 only (large batches uncorrelated expert activation left to future work), sequential greedy drafting only, no tree or sampling. MoE-SpeQ v1: cannot ablate a dense draft as no matched small MoEs exist; compared only with Transformers-based baselines (llama.cpp/vLLM infeasible in v1).
- Contradicts/supports our findings: No direct contradiction. SP-MoE's own text: with longer drafts the number of required experts grows and 'reaching full-expert activation increases loading overhead', performance gaps narrow (Fig 13) - same mechanism as our union growth.


## P-0007: S2-MoE: Enabling Efficient Self-Speculative Decoding for Mixture-of-Experts on Edge Devices. Haochen Huang, Shengxuan Qiu, Meng Li (Peking University). arXiv v1 15 Aug 2026, v2 19 Aug 2026. Code: github.com/angerybob/S2-MoE (not opened).

- ID read: arXiv 2608.15018v2 (PDF raw, 1023 lines; read fully; appendix tables 5-8 are interleaved in the extraction and were disentangled with the layout version).
- Raw text read: True
- Setup: OLMoE-1B-7B (6.7B total, 64 experts top-8), DeepSeek-V2-Lite-Chat (15.3B, 64 top-6), Qwen3-30B-A3B (30.5B, 128 top-8), GPT-OSS-120B (116.8B, 128 experts top-4). Hardware: Jetson Orin NX 16GB, AGX Orin 32GB and 64GB (CPU/GPU share DRAM; params beyond DRAM go to SSD, SSD model and bandwidth NOT stated), and RTX 4090 with constrained GPU memory (cold experts in CPU RAM; the constrained size is not stated). Implemented in llama.cpp with a new expert-level offload (non-expert params resident; frequently used experts kept resident from offline profiling; no LRU/prefetch baseline because none exists for llama.cpp). Weight quantisation format and batch size not stated (edge, 1-2 described in text). Baselines: AR, expert-sparsity self-draft (ES), layer-sparsity self-draft (LS), EAGLE-3 (llama.cpp PR #18039), Cascade (re-implemented on EAGLE-3). Draft length 8; width 2 for baselines, adaptive for S2-MoE. Training-free; components: routing-aware adaptive expansion (utility = confidence*T_AR / (marginal new experts*T_exp + draft cost)), reuse-aware expert gating (additive logit bias toward experts shared across draft tokens: LOSSY), context-aligned draft sharing the target KV cache. Tasks: Spec-Bench categories MT, RG, SU, TR, QA, MA (math), HE.

- Headline numbers (what | value | baseline | ratio | where):
  - Headline | up to 5.3x, ~2.0x average over AR; Jetson 1.3-5.3x, RTX 4090 1.2-2.9x | AR with the same offload |  | Abstract, Sec 6.2.1
  - Orin NX 16GB OLMoE HumanEval | 12.08 tok/s, accepted length 7.00 | AR 2.30 tok/s | 5.25x | Table 5
  - Orin NX 16GB DeepSeek HumanEval | 2.07 tok/s, acc 5.42 | AR 0.55 tok/s | 3.76x | Table 5
  - Orin NX 16GB Qwen3-30B HumanEval | 1.22 tok/s | AR 0.47 tok/s | 2.60x | Table 5
  - Orin NX 16GB GPT-OSS-120B HumanEval (116.8B on a 16GB device) | 1.65 tok/s, acc 2.95 | AR 0.95 tok/s; EAGLE-3 1.14x; Cascade 1.12x | 1.74x vs AR | Table 5
  - AGX Orin 64GB (all fit nearly): DeepSeek HE | 26.62 tok/s | AR 16.17 | 1.65x | Table 7
  - RTX 4090 constrained: OLMoE HE; GPT-OSS HE | 9.93 tok/s; 2.34 tok/s | AR 3.63; 1.68 | 2.74x; 1.39x | Table 8
  - Naive baselines under tight memory | ES 0.37-0.74x of AR (e.g. Orin NX OLMoE ES 1.40 vs 2.30), LS 0.17-0.42x on OLMoE | AR | <1 | Table 5
  - EAGLE-3 / Cascade typical | EAGLE-3 1.0-1.9x, Cascade 1.0-1.9x (e.g. Orin NX OLMoE HE 1.29x / 1.17x) | AR |  | Table 5-8
  - Quality (reuse-aware gating) | PPL ratio 1.012-1.013 (OLMoE, DeepSeek, Qwen3), top-1 agreement 89.1-93.0%, mean KL 0.026-0.065 (OLMoE 0.048, DeepSeek 0.065, Qwen3 0.026); task averages OLMoE 39.38 -> 39.61, DeepSeek 58.68 -> 60.74 (GSM8K/HumanEval/ARC-E/ARC-C); Qwen3 AIME24 68.89 -> 70.00, AIME25 54.44 -> 55.56 | Original unmodified model |  | Table 2
  - Cost model fit | R2 0.943-0.999, MAPE 0.8-4.5% |  |  | Fig 6
  - Context alignment acceptance gain | +23% to +79% (OLMoE GSM8K 33.01 -> 59.09) | independent-context draft | 1.79 | Table 3
- Catalog / first-pass numbers checked:
  - [yes] Expert-level offloading on Jetson Orin and RTX 4090 with llama.cpp -> Sec 5; SSD on Jetson, CPU RAM on 4090
  - [yes (was unread)] (catalog) I did not read its numbers -> Now read: up to 5.3x / ~2.0x avg vs AR; no hard-disk bandwidth given
  - [yes] First-pass: up to 5.3x, ~2.0x avg; Jetson 1.3-5.3x; 4090 1.2-2.9x -> Abstract and Sec 6.2.1
  - [yes] First-pass: Orin NX 16GB OLMoE HumanEval 5.25x; DeepSeek HumanEval 3.76x (Tables 5-8) -> Table 5
  - [yes] First-pass: PPL ratio ~1.012-1.013, top-1 agreement >89% -> Table 2
  - [yes] First-pass: baseline is AR with the same offload (not an LRU-cache-tuned path) -> Sec 5: no existing llama.cpp expert-offload implementation was available; baseline AR keeps profiled hot experts resident
  - [partly] First-pass: largest GPT-OSS-120B fits in 64GB -> The 120B model is also run on the 16GB Orin NX and 32GB AGX (AR 0.95 tok/s at 16GB), so overflow to SSD at ~13% of weights resident is exercised; weight format unstated
- Transfers to our regime: partly. Why: Closest published setting to ours: llama.cpp, tight memory, SSD tier (Jetson unified memory, so there is no separate RAM tier), batch 1-2, training-free, even a 117B 128-expert model on 16GB. But: (1) the lever that makes it work is lossy (reuse-aware gating bias, PPL +1.2%, top-1 agreement 89-93%); (2) baseline AR has no LRU/prefetch tuning, so gains vs our best path (LRU + explicit reads) are unmeasured; (3) Jetson SSD/UMA bandwidth is unstated and differs from a laptop CPU; (4) speculative acceptance lengths 3-7 per 8-token draft from an expert-sparse self-draft, much higher than the acceptance our byte tests assumed; (5) 128 experts/layer top-4 (GPT-OSS) vs our 512/top-10.
- Quality loss: Reuse-aware gating is a bounded routing approximation (authors: 'not a strictly lossless transformation'). WikiText-2 PPL ratio 1.012-1.013, top-1 agreement 89.1-93.0%, mean KL 0.026-0.065, RMS logit shift 4.9-7.0 (as printed); task accuracy within noise (OLMoE avg 39.38 -> 39.61, DeepSeek 58.68 -> 60.74, Qwen3 AIME24 68.89 -> 70.00). GPT-OSS PPL/KL not reported.
- Ablation: Fig 8 progressive speedup (self-spec + context alignment + adaptive expansion + reuse-aware gating); Fig 9 expansion strategies; Tables 3-4 context alignment and expert reuse; Fig 10 sensitivity to draft top-k and reuse cap. The cache-vs-layout-vs-prefetch decomposition is absent (no cache baseline).
- Limitations (authors): No limitation section. Authors state gating is not strictly lossless; evaluation only Jetson + one 4090; no comparison with any caching/prefetch offloader (none in llama.cpp).
- Contradicts/supports our findings: Does not contradict the lossless byte arithmetic, but it is a counter-example to 'speculation cannot pay on an SSD-bound llama.cpp path': 1.3-5.3x on SSD-bound Jetson. Reason it can differ from ours: (a) lossy gating cuts the union, (b) baseline has no cache, (c) acceptance lengths 5-7 (OLMoE) imply a >85% per-token acceptance for the draft, above our modelled 0.5-0.9 and tied to a specific self-draft. Also AR rates are tiny (0.47-0.95 tok/s for 30B-120B at 16GB), consistent with our 8GB/NVMe order of magnitude.


## P-0008: SpecExec: Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices. Ruslan Svirschevski*, Avner May*, Zhuoming Chen*, Beidi Chen, Zhihao Jia, Max Ryabinin (Yandex/HSE, Together AI, CMU, Meta). NeurIPS 2024.

- ID read: arXiv 2406.02532v3 (30 Nov 2024; PDF raw, flattened; text and tables 1-3 read, appendices not read in full)
- Raw text read: True
- Setup: Dense Llama-2-70B (chat and base, fp16 or GPTQ 4-bit) with Llama-2-7B draft; Llama-3-8B/70B; Mistral-7B / Mixtral-8x7B (GPTQ) as the only MoE rows. A100 80GB used with NO layers kept in VRAM to emulate RTX 4090/L40 (12-22 GB VRAM), RAM offloading, PCIe 4.0 (3090/2080Ti on PCIe 3.0), 140 GB DDR5 assumed for fp16 70B, batch size 1, 32+ tokens per prompt, datasets OpenAssistant, MTBench, C4, WikiText-2. Baselines: sequential offloaded decoding, SpecInfer, Sequoia (literature number).

- Headline numbers (what | value | baseline | ratio | where):
  - Llama-2-70B fp16, OAsst, t=0.6, budget 2048 | SpecExec 3.12 tok/s, gen rate 20.6 | SpecInfer 1.34 tok/s | 18.7x vs sequential (8.0x for SpecInfer); 2.33x SpecExec over SpecInfer | Table 1
  - Llama-2-70B GPTQ 4-bit, budget 128 / 256 | 6.02 / 6.17 tok/s | sequential offloaded | 8.9x / 9.1x | Table 1
  - Mistral-7B / Mixtral-8x7B (GPTQ), OAsst, budget 256 | 3.58 tok/s, gen rate 12.38 | sequential offloaded | 3.5x | Table 1 (the only MoE row)
  - Mixtral base, C4/WikiText-2 (Table 2) | 3.72-4.54 tok/s; 3.23 tok/s WikiText |  | 5.5x, 6.7x; 3.2x | Table 2
  - Llama-3-8B/70B | 2.62 tok/s (15.6x); 2.79-2.94 tok/s (16.6-17.5x) |  |  | Table 1
  - Consumer GPUs, Llama-2-70B GPTQ | RTX 4090 5.66 tok/s (8.3x); RTX 4060 3.28 (4.6x); RTX 3090 3.68 (10.6x); RTX 2080Ti 1.86 (6.1x, Sheared-Llama-1.3B draft) | sequential offloaded |  | Table 3
  - Optimal draft tree size | 1024-2048 (SpecExec), 128-512 (SpecInfer) on A100 |  |  | Fig 5, text
- Catalog / first-pass numbers checked:
  - [yes] Tree speculation with RAM offloading: 4-6 tok/s for Llama-2-70B at 4-bit -> Abstract: '4-6 tokens per second with 4-bit' (6.0-6.2 on A100 Table 1; 3.3-5.7 on consumer GPUs Table 3) and 2-3 tok/s with 16-bit
  - [partly] Dense model, not MoE -> Main results are dense; Mixtral rows exist (3.2-6.7x), far below the dense 9-18x, which is itself evidence of the MoE union penalty
  - [yes] First-pass numbers (3.12 / 18.7x, 1.34 / 8.0x, 6.0 / 8.9x, 4090 5.66 / 8.3x) -> Tables 1 and 3
- Transfers to our regime: partly. Why: Dense upper bound. For the MoE rows the speedup versus sequential offloaded decoding is only 3.2-6.7x with 128-1024 token trees and a 7B draft; RAM (not SSD) offload; CPU-GPU PCIe; the baseline moves the whole model each token (no cache), unlike an MoE with a cache. Draft model must fit alongside, and trees of 1000+ tokens would touch nearly every expert, hence nothing like it is available on a 512-expert MoE.
- Quality loss: None (target verification, lossless sampling and greedy).
- Ablation: Draft budget sweep (Fig 5), comparison with SpecInfer/Sequoia. No MoE-specific ablation.
- Limitations (authors): Authors discuss that gains depend on hardware and on draft/target pairing; appendix G shows non-offloaded results. No MoE caching analysis.
- Contradicts/supports our findings: No. The Mixtral 3.5x versus Llama-70B 15-18x gap in the same table is independent evidence that the union grows with the tree; consistent with our s1 #0 refutation of byte savings.


## P-0010: Utility-Driven Speculative Decoding for Mixture-of-Experts (Cascade). Anish Saxena, Po-An Tsai, Hritvik Taneja, Aamer Jaleel, Moinuddin Qureshi (Georgia Tech, NVIDIA Research).

- ID read: arXiv 2506.20675v1 (17 Jun 2025; PDF raw, flattened; read through sections 1-4, 7, 8, results; some figure data not extractable)
- Raw text read: True
- Setup: Five MoEs: Mixtral-8x7B FP8 (13B/47B active, 8 experts top-2), Phi-3.5-MoE FP8 (6.6B/42B, 16 top-2), OLMoE FP8 (1B/7B, 64 top-8), DeepSeek-V1 FP16 (2.8B/16.4B, 66 experts), Qwen-1.5 FP16 (2.7B/14B, 64 experts). One RTX 6000 Ada 48GB, 24-core i9-14900K, 64 GB RAM, vLLM (0.6.6 appears in the text), single-batch serving, n-gram speculation (main) with static K up to 3; EAGLE only on Mixtral (static K=1 best). Tasks: GSM8K math, HumanEval code, MT-Bench extraction, plus 4 mixed workloads. NO offloading or caching in the evaluation (only in related work).

- Headline numbers (what | value | baseline | ratio | where):
  - Mixtral n-gram speculation at K=7: ETR gain vs TPOT speedup | ETR 1.6-3.2x but TPOT speedup e.g. 1.4x for code (3.2x ETR) | dense LLaMA-3-8B 1.4-1.8x TPOT at K=7 |  | Sec 2.5, Fig 4
  - Verification data movement increase | 2-3x; Mixtral with 3 drafts up to 6 of 8 experts = 3x | K=0 |  | Abstract, Sec 1
  - Worst-case slowdown of static K | 26% / 38% / 54% for K=1/2/3 (5-model, 7-task); up to 1.5x on Mixtral | no speculation |  | Sec 6, Fig 13
  - Cascade worst-case slowdown | 5% (14% without adaptive back-off) |  |  | Abstract, Sec 5
  - Cascade gain over best static K (average) | 7-14% (abstract); Fig 13 caption says 7-15% | static K |  | Abstract, Fig 13
  - Cascade vs no speculation | up to 1.6x; +20% on average when speculation is beneficial | K=0 |  | Fig 13, Sec 1
  - OLMoE | Cascade 29% vs static K=2 31%, K=3 33% speedups |  | Cascade is 3% below static K=3 | Sec 6
  - DeepSeek-V1 | marginal 1% gain |  |  | Sec 6
- Catalog / first-pass numbers checked:
  - [yes] Verifying K drafts activates many more experts: Mixtral with 3 drafts loads up to 3x the data -> Sec 1
  - [yes] Speculation can slow decode by 1.5x -> Sec 1, Fig 1c (static K on Mixtral)
  - [yes] Cascade limits slowdown to 5% -> Abstract
  - [yes] First-pass: 'worst losses 26/38/54% at K=1/2/3' -> Sec 6
  - [partly] First-pass: vLLM 0.6.6; i9-14900K 64GB -> Hardware confirmed; the vLLM version string 0.6.6 appears but I did not confirm it names the evaluation framework (another 'v0.8.2' string also appears)
  - [yes] First-pass: 'up to 1.6x over no speculation' -> Fig 13 caption
- Transfers to our regime: no. Why: No offloading in the evaluation; experts resident in GPU memory (Mixtral FP8 ~47GB in 48GB). Cost is HBM, not disk. Drafter is n-gram (no training) so it is the one family member that is cheap; but gains are small (7-14%), and K is capped at 3 with a throttle (turn speculation off) rather than a speedup. The throttle idea (measure utility = ETR gain / verification cost; disable when <1) is directly applicable in principle: disk-bound verification cost is far larger, so the utility threshold would almost always say off.
- Quality loss: None (lossless).
- Ablation: Fig 14-15: dynamic spec disable, adaptive back-off, hill climbing each added; EAGLE comparison (Fig 17).
- Limitations (authors): Authors note OLMoE slightly below static K=3 due to test-phase overhead; focused on single-batch GPU serving.
- Contradicts/supports our findings: No. 'Up to 1.5x slowdown, 3x data movement for K=3' is the same direction as our drafted-cost results (ratios 1.29-3.6).


## P-0011: MoE-Spec: expert budgeting for speculative decoding. SAME PAPER as P-0002 (alphaxiv summary link in catalog; read arXiv PDF 2602.16052v1 instead).

- ID read: arXiv 2602.16052v1 (raw; see P-0002). The alphaxiv .md page was not opened.
- Raw text read: True
- Setup: OLMoE-1B-7B (7B/1B active, 64 experts, top-8), Qwen3-30B-A3B (128 experts, top-8), Mixtral-8x7B (8 experts, top-2). Drafts: EAGLE-3 for OLMoE/Qwen3, EAGLE-1 for Mixtral. NVIDIA A100 80GB (Mixtral on 2 GPUs), FP16, PyTorch 2.1/CUDA 12.1, ALL experts GPU-resident (App. A Table 3). Single request, draft tree 63 tokens (sweep 3-511), 80 samples/benchmark, max 1024 new tokens, T=0 and T=1 (5 seeds). Benchmarks GSM8K, MATH500, HumanEval, MBPP, CNN/DM. Baselines AR, EAGLE (named per model). Budget B chosen to maximise speed while quality stays within 1 std of EAGLE (budgets 8-56 OLMoE, 16-64 Qwen3, 2-7 Mixtral).

- Headline numbers (what | value | baseline | ratio | where):
  - Mixtral MBPP T=0 speedup over AR | MoE-Spec 2.3x | EAGLE 1.7x | 1.35 (computed 2.3/1.7) | Table 1, p.5
  - Qwen3-30B GSM8K T=0 speedup over AR | MoE-Spec 2.4x | EAGLE-3 1.9x | 1.26 | Table 1, p.5
  - Routing mass in top-32 of 64 experts, tree 63 (OLMoE) | 93% (excluded experts get 7%) |  |  | Intro, Fig 2c, Sec 4.3
  - OLMoE 127-token tree activates | 54 of 64 experts per layer |  | 0.84 of experts | Intro p.1
- Catalog / first-pass numbers checked:
  - [yes] Verification loads the union of draft-token experts -> Eq. 3
  - [yes] A training-free cap of B experts per layer controls the cost -> Abstract; but it is lossy (substitution/truncation)
  - [yes] First-pass: OLMoE 127-token tree activates 54 of 64 experts; top 32 of 64 hold 93% of routing probability -> Intro p.1 (the 93% figure is for tree size 63, Fig 2c)
  - [yes] First-pass: duplicate of P-0002 -> same arXiv id
- Transfers to our regime: no. Why: See P-0002.
- Quality loss: Lossy by construction (experts outside the shortlist are dropped or substituted). Table 1 picks the budget within 1 std of EAGLE quality, so table accuracies are close (e.g. Qwen3 HumanEval T=0 78 vs 82 AR, 82 EAGLE-3; Mixtral HumanEval 27 vs 20 AR, noise-level). Tighter budgets: Fig 3 quality relative to AR falls to ~78% (Mixtral) at maximum speedup; Static ranking gives near-zero GSM8K/MATH500 accuracy at B<=16 (Fig 5a); Router ranking tracks an oracle; Oracle reaches near-AR quality at B=24 (37.5% of experts).
- Ablation: Yes, Sec 4.4: ranking method (static/router/oracle) and coverage policy (truncation vs substitution perform comparably); tree-size sweep; reconstruction error (Fig 6: oracle matches router with 25% fewer experts). No ablation against any cache/offload.
- Limitations (authors): Authors (Sec 6): budget hyperparameter must be chosen per deployment; router-based ranking adds 2-3% overhead and may not pay for very small trees; single-request A100 only, batching may give smaller gains; softmax routing only (DeepSeek-V3 sigmoid may need adaptation); no per-layer budget; no co-activation model.
- Contradicts/supports our findings: No (see P-0002).


## P-0012: AcceptMoE: Commitment-Weighted Self-Sizing Verifier Expert Sets for Efficient MoE Speculative Decoding. Shuang Liang, Hao (Mark) Chen, Zhiwen Mo, Qianzhou Wang, Guoyu Li, Lingxiao Ma, Wayne Luk (Imperial College London, Tile-AI). 4 Aug 2026.

- ID read: arXiv 2608.02989v1 (PDF raw, 340 lines layout; read fully; Fig 3/4 bar labels recovered from text)
- Raw text read: True
- Setup: Qwen3-30B-A3B-Instruct-2507, Qwen3-Coder-30B-A3B (128 experts top-8), GPT-OSS-120B (128 experts top-4, MXFP4) each with EAGLE-3 drafts (5 steps, top-k 10, 64 draft tokens). SGLang 0.5.12 batch 1. Full residency: one RTX PRO 6000 Blackwell, 256 generated tokens, 50 prompts, 3 repetitions. Offloading: one RTX 5090 32GB, per-layer pool of X=48 expert slots (37.5% of 128), full weights in PINNED HOST RAM (Qwen BF16, GPT-OSS packed MXFP4), LRU resident set, 20 prompts, median of 3. Not SSD. Baselines: Vanilla AR, Standard SD (natural routing), Count-B, MoE-Spec at matched budget B0. Accuracy on GSM8K, MATH500, HumanEval, MBPP over full evaluation sets. Method restricts the verifier router (logit mask) to an expert set S built from commitment-weighted router demand, size from effective rank, and under offloading prunes low-demand NON-RESIDENT experts (cache-conditional routing, lossy).

- Headline numbers (what | value | baseline | ratio | where):
  - Full residency throughput, mean over 12 model-task pairs | 1.290x (range 1.217-1.339x; per model 1.249-1.318x) | Standard SD natural routing | 1.29 | Fig 3, Sec Throughput
  - Offloading, RTX 5090, 48 slots | mean 2.06x (per-pair 1.54-2.99x) | Standard SD natural routing (NOT AR) | 2.06; mean of the 12 per-pair ratios I recovered from Fig 4 text = 2.062 | Fig 4 + text
  - Standard SD vs Vanilla AR under offloading | Standard SD is SLOWER than AR on 7 of 8 Qwen pairs; faster than AR on all 4 GPT-OSS pairs | Vanilla AR | <1 on Qwen | Sec End-to-End Throughput
  - Expert-weight H2D traffic per output token (MiB), Qwen-Inst / Coder / GPT-OSS | AR 888 / 721 / 439; Standard SD 2394 / 2025 / 1243; AcceptMoE 633 / 464 / 298 |  | SD/AR = 2.70 / 2.81 / 2.83; AcceptMoE/SD = 0.264 / 0.229 / 0.240 (-73.6 / -77.1 / -76.0%); AcceptMoE/AR = 0.71 / 0.64 / 0.68 | Table 2
  - Cache hit rate | AR 81.0 / 83.5 / 81.0; Standard SD 73.0 / 74.4 / 74.3; AcceptMoE 83.8 / 87.0 / 88.5 |  | +10.8 / +12.6 / +14.2 points vs SD | Table 2
  - Fetch stall share of wall time | Standard SD 65-78%; AcceptMoE 27-35% |  |  | Fig 5a
  - Residency-aware pruning on vs off | traffic -38.6% to -48.6%; hit rate +6.9 to +8.2 pts; throughput +4.6% to +15.1% | same selector without pruning |  | Sec Ablation, Fig 5c-d
  - Accuracy: mean over 12 pairs (Table 1) | AcceptMoE 90.63 (offload 90.25) | Standard SD 90.90, Vanilla AR 91.01 | -0.27 pts vs Standard SD (-0.4% vs AR) | Table 1
  - Accuracy at matched budget B0 (23-34 experts) | AcceptMoE-B 90.60, MoE-Spec 88.15, Count-B 82.72 |  | MoE-Spec -3.1% vs AR, Count-B -9.1% | Table 1
- Catalog / first-pass numbers checked:
  - [yes] Token count, union size and transfer traffic are separate costs -> Abstract
  - [partly] 2.06x under physical expert offloading -> True but versus Standard SD, which itself is slower than Vanilla AR on 7 of 8 Qwen pairs; the gain over AR is not given as a number in the text (AcceptMoE is faster than AR on all 12 pairs)
  - [yes] 1.29x with all experts in GPU memory -> Versus Standard SD with natural routing
  - [yes] First-pass: 48-slot pool, host RAM not SSD, RTX 5090 32GB, host-to-device traffic -73.6 to -77.1%, hit +10.8-14.2 pts, accuracy -0.27 -> as above
  - [partly] First-pass: Fig 4 text only gives the mean 2.06x -> Per-pair ratios are legible in the extracted figure text; they average 2.062
- Transfers to our regime: partly. Why: The most relevant mechanism to us: restrict verifier eligibility to what is cached. Breaks for us because: GPU + pinned host RAM, not SSD to CPU; pool of 48 of 128 slots (37.5% of the experts per layer) versus our cache of about 4% (8GB) to 40% (32GB, ~2-20 footprints); a trained EAGLE-3 draft with 64-token trees; the method is lossy (-0.27 pts on its own benchmarks, but MoE-Spec at same size -3.1% and Count-B -9.1%); baseline is Standard SD, not AR or an LRU-tuned best path.
- Quality loss: Mean accuracy -0.27 pts vs Standard SD over 12 model-task pairs (per-pair -1.83 to +1.22 for pruning; AcceptMoE vs SD per pair mean abs 0.43, max 1.22). Gets worse at fixed small budgets (App Fig 5b: 23-92 points lost at the smallest budgets). No perplexity numbers.
- Ablation: Yes and relevant: Fig 5 and Table 2 isolate residency-aware pruning from set construction (traffic -38.6% to -48.6%, throughput +4.6-15.1%) and compare AR, SD, SD+AcceptMoE traffic and hit rate. Self-sizing vs fixed budget; matched-budget MoE-Spec/Count-B comparison.
- Limitations (authors): No dedicated limitations section. Authors say the mask changes the target router logits so any such selector is an approximation, not distribution-preserving; GPT-OSS gain smaller (lower fan-out).
- Contradicts/supports our findings: Supports us: under a 48-slot LRU pool, natural-routing Standard SD moves 2.7-2.8x the expert bytes of AR per token (Table 2) and is slower than AR on 7/8 Qwen pairs; hit rate falls 8-9 points. Our s1 #0 found drafted-block LRU bytes >= 1.002x of k=1 (up to 3.6x). Their 2.7x sits inside our range. It also shows a cache-conditioned (lossy) restriction recovers part of it, i.e. the lever our s1 #6 'router with temporal stickiness' tests, now with a published equivalent.


## P-0013: Bundle row: SP-MoE (the linked paper, arXiv 2510.10302v2), MoE-SpeQ (2511.14102), DraftExpert (2607.24434), S2-MoE (2608.15018). Each read and itemised under P-0006, P-0004, P-0007.

- ID read: 2510.10302v2, 2511.14102v1/v3, 2607.24434v1, 2608.15018v2, all raw PDF text
- Raw text read: True
- Setup: See P-0006 (SP-MoE, MoE-SpeQ/Specter), P-0004 (DraftExpert), P-0007 (S2-MoE).

- Headline numbers (what | value | baseline | ratio | where):
  - MoE-SpeQ | v1 abstract 2.34x; v1 body 3.29x vs Mixtral-Offloading-SC; v3 (Specter) 1.50x avg/1.88x max vs SpecMoEOff |  |  | P-0006
  - SP-MoE | 1.07x-3.5x TPOT; avg 34%/19%/12% less TPOT than three SD-enabled offloaders |  |  | P-0006
  - DraftExpert | 1.45x avg (1.29-1.61x) vs AR offload |  |  | P-0004 Table 2
  - S2-MoE | up to 5.3x, ~2.0x avg vs AR |  |  | P-0007
- Catalog / first-pass numbers checked:
  - [partly] MoE-SpeQ reports up to 2.34x over offloading baselines -> v1 abstract only; later versions differ (Specter 1.50x avg, 1.88x max vs SpecMoEOff)
  - [partly] These use GPU/PCIe offload with a draft model that predicts experts -> True for SP-MoE and MoE-SpeQ; DraftExpert's MN setting is phone Flash to NPU in llama.cpp with a learned per-layer draft expert; S2-MoE is self-speculative (no separate draft), llama.cpp, Jetson SSD tier
  - [absent] S2-MoE (arxiv 2608.15018) ... [catalog row text truncated] -> row text was cut off at 'S2-MoE (https://arxiv.org/pdf/2608.' in my extract
- Transfers to our regime: partly. Why: Only S2-MoE and DraftExpert-MN touch llama.cpp/flash tiers; all four use either trained/auxiliary drafts or lossy gating; none has an LRU-tuned best-path baseline or an NVMe-to-CPU laptop. See individual rows.
- Quality loss: SP-MoE, MoE-SpeQ, DraftExpert: none (exact). S2-MoE: PPL x1.012 from reuse-aware gating.
- Ablation: See individual rows.
- Limitations (authors): See individual rows.
- Contradicts/supports our findings: No.


## P-0014: Hiding offloading latency with speculative decoding. SAME PAPER as P-0005 (SpecMoEOff, arXiv 2508.21706, pdf link).

- ID read: arXiv 2508.21706v2 PDF, raw text (see P-0005)
- Raw text read: True
- Setup: See P-0005.

- Headline numbers (what | value | baseline | ratio | where):
  - Throughput vs MoE-Lightning | 1.6-2.9x avg 2.1x (text) ; '2.5x' in abstract/intro/conclusion | MoE-Lightning |  | Sec 6.2
  - SD alone | 1.53x APPS / 1.45x CNN/DM | SpecMoEOff-w/o-sd |  | Sec 6.2
  - Global batch in the roofline analysis | 824 (A30), 515 (4090D) |  |  | Sec 3.3
- Catalog / first-pass numbers checked:
  - [partly] Speculation enlarges per-expert workload, so one transfer is amortised over more tokens when transfer dominates -> Correct for hundreds of sequences per batch; at batch 1 the draft tokens come from one sequence and add experts
  - [no] This is the same mechanism -> The mechanism in the paper is workload enlargement across a huge batch with CPU attention; ours is one sequence, where it does not apply
  - [yes] First-pass: PDF link not directly readable, content from html v1 -> now raw PDF v2 read
- Transfers to our regime: no. Why: See P-0005: large-batch throughput system with 190-250 GB host RAM.
- Quality loss: None.
- Ablation: See P-0005.
- Limitations (authors): See P-0005.
- Contradicts/supports our findings: No.


## P-0016: EStream: Fast and Memory-Efficient MoE Prefill through Expert Virtualization on Mobile NPUs. Junming Zhang, Zhenzhe Zheng, Fan Wu (SJTU), Xiaoyao Huang (China Telecom), Jie Wu (Temple). 6 Sep 2026.

- ID read: arXiv 2609.06551v1 (PDF raw; sections 1-5 and discussion read; Fig 3(a) bar values per model NOT recoverable from text)
- Raw text read: True
- Setup: OnePlus PLK110/15 phone, Android 16, 15.1 GiB RAM, Snapdragon 8 Elite Gen 5 (Hexagon v81 NPU), UFS 4.1 (bandwidth not stated in text read). Models: OLMoE-1B-7B (16 layers, 64 experts top-8), LFM2.5-8B-A1B (32 experts, top-4), DeepSeek-V2-Lite (64 + shared, top-6); scaling: Qwen3-30B-A3B, Mixtral-8x7B. Weight-only GGUF Q4_0, FP32 activations. Prefill-only (no decode). Baselines llama.cpp-CPU, MNN-CPU/OpenCL, ONNX Runtime GenAI-CPU (non-offloading), EdgeMoE, PowerInfer-style parameter streaming (offloading). Median TTFT of three fresh-process runs, OS page cache dropped.

- Headline numbers (what | value | baseline | ratio | where):
  - Pure-prefill TTFT speedup over the fastest completed non-offloading baseline, 18 settings (256-4096 tokens) | 2.25-27.57x | fastest baseline per cell |  | Abstract, Sec 5.2, Fig 8
  - 4K tokens TTFT, OLMoE / LFM2.5 / DeepSeek-V2-Lite | 2.12 / 2.14 / 9.53 s | 58.45 / 51.15 / 120.02 s fastest non-offloading | 27.6x / 23.9x / 12.6x | Sec 5.2
  - Vs PowerInfer (strongest offloading baseline) | 2.96-50.78x faster, 1.16-1.69x less memory | PowerInfer streaming |  | Sec 5.2
  - Peak memory | 0.59-1.76 GiB; 6.45-12.29x lower than the fastest non-offloading |  |  | Abstract, Sec 5.2
  - Layer-expert pairs activated by one prompt | 80.1-98.1% at 512 tokens; 92.4-100% at 4K tokens (three models in Fig 3a) |  |  | Sec 2.2, Fig 3a
  - Short prompts | CPU runtimes faster at mean 16 and 23 tokens; crossover between 66 (within 8%) and 80 tokens (1.03-1.21x faster) |  |  | Sec 5.2
  - Streaming overhead vs all-resident NPU reference | 4.3-23.0% higher latency at 4K; 3.73-5.41x less memory; 94.9% of UFS loading overlapped (LFM2.5) | NPU-resident |  | Sec 5.5, Fig 11
  - Larger models | Qwen3-30B-A3B TTFT 4.84-6.84 s (1-4K) in 1.46-1.87 GiB; Mixtral 7.09-13.58 s in 2.77-4.11 GiB |  |  | Table 3
  - Quality (Q4, same routing) | OLMoE/LFM2.5 PPL change <0.1%, task <=1.5 pts; Qwen3 PPL +5.51% (31.41 -> 33.14), up to 3 pts | Host CPU reference |  | Table 4
- Catalog / first-pass numbers checked:
  - [yes] Demand-paged mmap postpones expert residency rather than avoiding it -> Sec 2.2: 'Demand-paged mmap delays rather than eliminates physical residency'
  - [yes] Across a multi-token prefill the union of activated experts covers nearly the whole pool -> 80.1-98.1% at 512 tokens, 92.4-100% at 4K (the three phone models: 32-64 experts per layer)
  - [yes] First-pass: TTFT 2.25-27.57x, memory 6.45-12.29x -> Abstract says memory 1.19-12.29x; Sec 5.2 gives 6.45-12.29x vs the fastest completed non-offloading baseline; the abstract range includes other comparisons
  - [yes] First-pass: 4K 2.12 s vs 58.45 s -> Sec 5.2
  - [yes] First-pass: Fig 3(a) 80.1-98.1% / 92.4-100% -> Sec 2.2
  - [yes] First-pass: CPU runtimes win below 66-80 tokens -> Sec 5.2
- Transfers to our regime: partly. Why: Supports the high prefill union (our assumed 0.75-0.95 vs decode-trace proxy 0.55-0.61 at 128-249 tokens) at 512 tokens, but only for 32-64-expert models (union saturates faster than 512 experts top-10) and it is a phone NPU with UFS, not a CPU with NVMe. Prefill-only; decode not addressed. Baselines are not an explicit-read, expert-major CPU path: the headline is mostly vs CPU/GPU runtimes at the NPU's compute advantage (NPU 4.4-4.6x faster than CPU/GPU on routed matmuls, route count >= 64).
- Quality loss: No approximation of weights beyond Q4 (shared with baseline); only order-of-operation effects: Table 4 as above (Qwen3 +5.51% PPL).
- Ablation: Fig 11: NPU-resident vs CPU-resident vs streaming; graph sharing vs QNN-switch (6.27-21.02x faster); pipeline G/Q/B sweep of 184 configs; layer vs expert-group streaming. A cache-vs-layout-vs-prefetch decomposition for decode: none.
- Limitations (authors): Authors: prefill-only, no decode; Q4_0 only (mixed precision could shorten loads but complicates dequant); Qualcomm-specific.
- Contradicts/supports our findings: No. Agrees that prefill touches most experts (so sorting does not reduce bytes; gain vs mmap is the read pattern). Our decode-proxy union (0.55 at 128 tokens, 128 experts) is lower than their 80-98% at 512 tokens, as expected since prefill tokens are more diverse; a real 125B number remains unmeasured.


# Family section: speculative / grouped-token verification for MoE expert streaming (s1 #0, #1, #2)

## Further prior art found (read at the same depth where marked)
READ (raw PDF text, key sections):
- Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference (Skliar, van Rozendaal et al., Qualcomm AI Research; TMLR June 2025; arXiv 2412.00099). Cache-aware routing: add a cache prior to router logits (Cache-Prior), training-free, batch 1, llama.cpp CPU on 12/16 GB phones with LRU expert cache and flash tier. Claims miss rate -50%+, PPL +0.1-3%, accuracy delta <0.1%, ~2x on-device throughput vs LRU (Qwen1.5-MoE 4/8-bit, cache 30/45 of 60 experts). CLOSEST prior art to our family s1 #6 ('router with temporal stickiness') and to the lossy part of S2-MoE/AcceptMoE. No speculation.
- MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE (Huang et al., Tsinghua/Huawei; arXiv 2505.19645v4, 16 Feb 2026). SD helps MoE MORE than dense at medium batch (all experts already active), up to 2.29x for Qwen2-57B-A14B on GPU; theoretical target-efficiency model. Opposite regime (batch large enough that the union is saturated).
- SpecMoE: A Fast and Efficient MoE Inference via Self-Assisted Speculative Decoding (Bang et al., KAIST; arXiv 2604.10152, 11 Apr 2026). CPU-offloaded MoE with self-assisted SD, H100 + 1 TB DDR5, up to 4.30x throughput, mostly large batches; an SSD-offload figure for NLLB-MoE (Fig 16) shows throughput vs batch size only.
- MoE-SpAc: speculative activation utility for heterogeneous edge MoE inference (Li, Lin, Ge, Ye; arXiv 2603.09983v2, 23 Oct 2026). Uses SD as a lookahead sensor for prefetch/eviction (utility estimator), RTX 4090 + PCIe 4.0; claims +42% TPS over the best SD baseline, 4.04x avg over standard baselines. GPU + CPU RAM, not SSD.
- SliceMoE: bit-sliced expert caching under miss-rate constraints (Choi et al., KAIST; arXiv 2512.12990). Cache-aware routing + mixed-precision slices, decode latency up to 1.81x/1.64x, energy 2.37x/2.85x on DeepSeek-V2-Lite / Qwen1.5-MoE; cites 'SSD offloading for LLM MoE weights considered harmful in energy efficiency' (IEEE CAL 2025).
Named in the papers I read, NOT opened (title/citation only, UNREAD): EVICT (Pan et al. 2026: 74.7% fewer verified tokens but only 32.5% fewer activated experts on Qwen3-30B, as quoted by AcceptMoE), XShare (Vankov et al. 2026, per-batch expert subset), Opportunistic Expert Activation / batch-aware routing (Oncescu et al. 2025), Dynamic expert sharing for diffusion LLMs (Chen et al. 2026), 'Self-Speculative Decoding for On-device MoE Acceleration' (Zheng, Xu, Wang; WWW 2026, cited by DraftExpert), SpecMoE (earlier version listing), MoE-Infinity, AdapMoE, ProMoE, Klotski, Fiddler, PowerInfer-2 (phone UFS), Mixtral-Offloading (Eliseev & Mazur 2023).
From memory (not re-read here): Pre-gated MoE, SwapMoE, HOBBIT, Fate, DAOP, EdgeMoE, Sequoia, SpecInfer, Medusa, EAGLE series, llama.cpp EAGLE-3 PR #18039.

## What is actually new in OUR variant
Our variant (pubtech/A.md s1 #0/#1): lossless group-verify of k same-sequence tokens by expert-major union, CPU-only, experts streamed from NVMe through a tiny per-layer LRU (2.1-4 token-footprints, 4-12% of experts), batch 1, 512-expert top-10 125B-class model, judged against the best non-speculative path (explicit reads + LRU).
- Not new: the observation that verification touches the union (Cascade, MoE-Spec, EcoSpec, DraftExpert, S2-MoE all state it), and that natural-routing SD under offload can be slower than AR (DraftExpert Table 2 top-r 0.45-0.54x of AR; AcceptMoE Table 2/Fig: SD moves 2.7-2.8x the bytes of AR and is slower on 7/8 Qwen pairs).
- Not new: cache-aware lossy routing (Cache-Conditional Experts 2412.00099, S2-MoE reuse-aware gating, AcceptMoE residency pruning). Our s1 #6 (router with temporal stickiness) is a variant of it.
- Closest published runs: S2-MoE (llama.cpp, SSD on Jetson, 16-64 GB, batch 1-2, GPT-OSS-120B at 16 GB: AR 0.95 -> 1.65 tok/s; lossy gating; no LRU baseline), AcceptMoE (cache-conditional restriction with an LRU pool; GPU+host RAM; 37.5% cache), DraftExpert (flash->NPU phone, trained draft expert).
- What none of them did (so it is open or at least un-measured in the literature): (1) an exact (lossless) count of bytes per accepted token against an LRU at cache sizes of 2-4 footprints; (2) an NVMe-to-CPU laptop (8/16/32 GB) with measured sequential-read throughput; (3) 512 experts/top-10 scale; (4) comparison against an LRU+explicit-read best path rather than naive offload; (5) the kill-rule result that, for natural routing, accepted-token bytes are >= LRU for acceptance <= 0.9. Our measured numbers (ratio >=1.002; 2.7-2.8x extra bytes reported by AcceptMoE at a 37.5% cache) are mutually consistent, but our result is a trace-based simulation on a 128-expert 30B model, not a run.

## Does anything here change what we should believe about our own measurements?
1. Direction confirmed, magnitude independent: AcceptMoE Table 2 gives a real-hardware data point (SD with natural routing moves 2.70/2.81/2.83x the expert bytes per output token of AR; hit rate 73-74% vs AR 81-84%). Our simulator gives LRU-relative bytes per accepted token 1.0-3.6x (wrong-branch accounting). No contradiction; slight support for the 'wrong-branch' pessimistic end (their ratio 2.7x includes 64-token trees, so it is not like-for-like).
2. The published routes to a win are all LOSSY (S2-MoE gating, AcceptMoE residency pruning, MoE-Spec budget, Cache-Conditional Experts) or need training (DraftExpert, EcoSpec predictor). The lossless byte-side refutation therefore stands; what is untested by us is lossy cache-conditional verification at 512 experts. S2-MoE's 1.3-5.3x on SSD-bound llama.cpp is the strongest claim that a lossy version works in our setting; its baseline has no cache, so the like-for-like figure vs LRU is unknown.
3. Where our assumed numbers may be too pessimistic: S2-MoE accepted lengths are 3-7 (OLMoE 5.0-7.0) with an expert-sparse self-draft; our table assumed per-token acceptance 0.5-0.9 with k=8, giving expected accepted tokens 2-5.7. At 0.9+ acceptance k=8 hits the single cell where our simulator beat LRU (0.90 ratio). So if a draft with >90% per-token acceptance exists for a 125B, the margin is thin but positive for the union approach even with exact routing, and larger with lossy gating.
4. Prefill (s1 #2): EStream's 80-98% of layer-expert pairs at 512 tokens (32-64 expert models) is above our decode-proxy 0.55 at 128 tokens; our assumed 0.75-0.95 for the 125B remains plausible, still unmeasured.
5. Catalog hygiene: P-0011 duplicates P-0002; P-0014 duplicates P-0005; P-0013 bundles four papers; P-0006 bundles two. pubtech/A.md labels Cascade as P-0039 for s1 #1 while TESTLIST lists it as P-0010 for s1 #1 (already noted in the first pass).
