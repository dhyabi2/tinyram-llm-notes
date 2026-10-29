# Published-technique family tests, group E (second screen, families 10 to 17)

Code: `<tinyram-llm>/tools/pubtech/f_E.py` (functions `fam_s2_10` ... `fam_s2_17`, dict `FAMILIES`).
Tests: `<tinyram-llm>/tests/test_pubtech_E.py` (120 checks, about 28 s). All numbers below are produced by those functions.

Common caveats. Every trace number comes from the three Qwen3-30B-A3B decode traces (128 experts, top-8, 48 layers, 249 / 249 / 299 tokens), standing in for the 125B (512 experts, top-10). Cache entries = footprints x 384 accesses per token; the 8 / 16 / 32 GB machines are 2.1 / 7.6 / 20.8 footprints. Throughput uses the combo_model constants (55 GFLOP/s decode, 12 GFLOP per token with 56.7% dense, 3.11 GB/s explicit reads, 1.296 GB of expert bytes per token). The "overlap bound" is the ideal full-overlap limit 1/max(compute, I/O), an upper bound, not an implemented path. No model was run (the Mac load average was 11-19 and OLMoE has no `override-kv` path in `llama-bench`, no `llama-eval-callback` binary is built, and no numpy / gguf package is installed).

---

## Family 10 [screen 2 #10]: second LAN machine as expert compute server

Idea: split the experts per layer 256/256 over two nodes, each streams and multiplies its own; node 0 keeps the dense part. Screen claim: more than 3x; screen's corrected figure 2.1x serial / 1.6x overlap at 8 GB, bound 2/1.26 = 1.59x on compute.

| Quantity | Measured | Screen |
|---|---|---|
| straggler, random halves, 20 seeds x 3 traces (mean / min / max) | 1.266 / 1.252 / 1.281 | 1.25-1.28 |
| exact uniform-draw straggler, k=8 of 128 / k=10 of 512 | 1.265 / 1.244 | "similar or worse" |
| best balanced split (greedy by usage), in-sample / held-out / first half to second half | 1.230 / 1.250 / 1.237 | kill line 1.15 |
| pooled-cache LRU hit at 8 GB (one node / two nodes, each with the same cache) | 0.567 / 0.811 | 0.55 / 0.8 |
| compute gain of 2 nodes, dense part on node 0 (Amdahl) | 1.19x | 1.59x |
| compute gain if ALL compute were split | 1.58x | 1.59x |
| 8 GB gain over one node, serial / ideal overlap | 1.68x / 1.10x | 2.1x / 1.6x |
| 16 and 32 GB gain (serial / overlap) | 1.14-1.16x / 1.10x | 1.4-1.45x |

Verdict: **partly confirmed, claim lowered.** The straggler (1.25-1.28) and the pooled hit (0.8) reproduce. The balanced split does not get below 1.15 (kill criterion tripped). The screen's compute bound 1.59x only holds if the dense 57% of the FLOPs is split too, which the design (node 0 keeps the dense part) does not do; with dense serial the compute gain is 1.19x, and 48 round trips add 14 ms. The real gain is about 1.7x at 8 GB serial (cache pooling), about 1.1x when overlap already hides I/O, and 1.1-1.16x at 16/32 GB. It needs two machines.

Published methods vs measured: OD-MoE reports about 75% of a GPU-cached Mixtral speed across many nodes; exo reports 5.37 tok/s for 671B on 8 M4 minis with pipeline/tensor splits; none reports a straggler or Amdahl number for a 2-node 512-expert split, so our 1.27 straggler and 1.19x dense-limited compute gain are new local numbers. The expert placement and affinity papers address the same load-balance problem; our held-out balanced split recovers only 1.250 from 1.266.

Catalog rows mapped to this test (`fam_s2_10`):
- P-0185 [screen 2 #10] idea row
- P-0186 [screen 2 #10] published method: OD-MoE (arXiv 2512.03927)
- P-0187 [screen 2 #10] published method: exo blog, DeepSeek V3 on 8x M4 Mac Mini
- P-0188 [screen 2 #10] published method: Latency-optimized expert placement for distributed edge MoE (arXiv 2508.12851)
- P-0189 [screen 2 #10] published method: Inter-layer expert affinity (arXiv 2401.08383)
- P-0190 [screen 2 #10] published method: Reddit snapshot on llama.cpp RPC over Thunderbolt

Unmeasured: the real round-trip time on a LAN, peer jitter, the 125B's expert popularity skew (only the exact uniform case is computed), and the second node's own cache size (assumed equal to node 0's, so node 1's saved dense memory is not credited).

---

## Family 11 [screen 2 #11]: layer-split expert store (10 layers local NVMe, 38 in a peer's RAM)

Screen claim: about 3x (2.9-3.8 tok/s over a 1.2 baseline). Screen corrected: about 1.0x versus the best path.

| Quantity | Measured |
|---|---|
| combo_model, 8 GB serial, peer link 2.5 / 1.1 / 0.11 GB/s as fraction of local 3.11 GB/s | 0.90 / 0.54 / 0.07 |
| combo_model, 16 GB serial, same links | 0.98 / 0.86 / 0.30 |
| peer DRAM (13 GB) as extra cache, 8 GB: hit 0.56 to 0.96; serial vs local at 2.5 GB/s / 1.1 / 0.11 | 0.91 / 0.57 / 0.08 |
| the idea as stated (10 layers local, 38 peer, parallel fetch), 8 GB serial vs local at 3.11 / 2.5 / 1.1 GB/s | 1.11 / 1.01 / 0.64 |
| same, 16 GB serial vs local at 3.11 / 1.1 / 0.11 | 1.02 / 0.90 / 0.35 |
| maximum overlap-bound gain from an infinitely fast link | 1.00x (compute ceiling 4.58 tok/s) |
| peer capacity: 38 layers need / a 16 GB peer holds | 52.5 GB / 11 layers |

Verdict: **refuted as a gain, confirmed as roughly 1.0x.** No row beats the local path by more than 1.11x serial and none at the overlap bound. A 16 GB peer cannot hold 38 layers (52.5 GB needed). The screen expected the 2.5 GB/s peer-cache row within 5% of local; we get 0.91 serial / 0.97 overlap (about 10% worse serial). The 3x claim came from the 1.2 tok/s baseline.

Published methods vs measured: exo + RDMA (Qwen3 235B 19.5 to 31.9 tok/s on 4 nodes) and llama.cpp RPC ship activations, a different mechanism with real speedups; Harvest and RMAI (up to 10x faster expert switching than disk) compare against slow tiers, not against a 3 GB/s NVMe with explicit 2.7 MB reads; FlashMoE / MoE-SpeQ are single-machine SSD levers already in our best path. We did not reproduce any of them.

Catalog rows mapped (`fam_s2_11`):
- P-0191 [screen 2 #11] idea row
- P-0192 exo with RDMA over Thunderbolt 5 on M3 Ultra Mac Studios (Geerling benchmark, via blog write-ups)
- P-0193 Harvest: Opportunistic Peer-to-Peer GPU Caching for LLM Inference (arXiv 2602.00328)
- P-0194 RMAI: Rethinking memory for AI inference (Noohi et al., Edinburgh)
- P-0195 llama.cpp RPC backend guides and kjaiswal/llama-cpp-distributed-benchmarks
- P-0196 FlashMoE, SSD offloading (arXiv 2601.17063) and MoE-SpeQ (arXiv 2511.14102)

Unmeasured: any real link rate between two machines (the rates 3.11 / 2.5 / 1.1 / 0.11 GB/s are inputs, not measurements); prefetch overlap across the link (the 0.55 predictor precision is not replayed here).

---

## Family 12 [screen 2 #12]: zone-map block sparsity inside each expert (needs a model)

Screen claim: 2.6x (50% dead blocks plus 42% cross-token reuse = 0.38 GB/token). Screen corrected: 1.0-1.3x; gate-first caps bytes at 1.5x at d=0.5.

| Quantity | Value |
|---|---|
| gate-first bytes gain at d = 0.3 / 0.5 / 0.6 / 0.8 | 1.25 / 1.50 / 1.67 / 2.14x |
| compute-ceiling lift at d = 0.5 (dense 57% fixed), gate-first / predictor | 1.17x (5.36 tok/s) / 1.28x (5.85 tok/s) |
| d needed for a 2.6x byte gain under gate-first | 0.92 |
| read rate for a 32-neuron block (35 KB, H=2048) / 128-neuron block (141 KB), vs 2.7 MB reads | 0.37 / 0.71 of the large-read rate (latency 80 us assumed) |
| I/O-time gain at d = 0.5 after fragmentation, block 32 / block 128 | 0.55x (a loss) / 1.06x |
| live share of the union of k tokens, d = 0.5, k = 2 / 4 / 8 (independence) | 0.75 / 0.94 / 0.996 |
| MEASURED: token-to-token expert persistence on the traces | 0.432 (0.425 / 0.455 / 0.417) |
| MEASURED: LRU hit at 8 GB (2.1 footprints) | 0.567 |

Verdict: **needs a model for the decisive number (dead fraction d), but the closed-form part refutes the claim.** The "42% reuse" is the measured expert persistence (0.43), and the 8 GB LRU already hits 0.57, so it double-counts the best path: the byte gain over the best path at the claimed d=0.5 is 1.5x at most, 1.17x on the compute ceiling, and small-block reads (an assumed latency model, not measured) turn it into a loss. 2.6x would need d of about 0.92.

Published methods vs measured: TEAL reports 40-50% sparsity with 1.53x / 1.8x decode speedup on dense models; TurboSparse shows SwiGLU experts are only about 40% sparse until ReLU-fied (retraining); PowerInfer-2 / FloE / DualSparse / CATS use predictors or calibrated thresholds. Our closed-form is consistent with the TEAL-scale ceiling (1.5-1.8x) only for the dense-model decode case; on this MoE the dense share and the gate-first read cap it lower. d itself was not measured.

Catalog rows mapped (`fam_s2_12`):
- P-0197 [screen 2 #12] idea row
- P-0198 TEAL, training-free activation sparsity (ICLR 2025)
- P-0199 FloE, on-the-fly MoE inference on memory-limited GPU
- P-0200 TurboSparse / dReLU (Mixtral)
- P-0201 PowerInfer-2
- P-0202 DualSparse-MoE
- P-0203 CATS / Enabling Dynamic Sparsity in Quantized LLM Inference

Unmeasured: the dead fraction of the SwiGLU intermediate at a fixed output error, the k-token union dead fraction, perplexity with thresholds. These need `ffn_moe_swiglu` activations of a real small MoE (OLMoE); the build here has no eval-callback tool and the model run was not done.

---

## Family 13 [screen 2 #13]: typing-time chunked prefill

Screen claim: 40x TTFT; screen corrected: TTFT 1x for pasted prompts, 5-10x at best for typed prompts with large chunks, tok/s 1.0x. Expected chunk I/O ratio 5-10x at c = 16-32.

| Chunk c (tokens) | 8 | 16 | 32 | 64 | 128 | 500 |
|---|---|---|---|---|---|---|
| MEASURED unique (layer, expert) reads vs one batch (30B traces, mean of 3) | 12.3x | 8.3x | 5.3x | 3.2x | 1.9x | 1.0x |
| 125B uniform-routing model, same ratio (500-token prompt) | 9.2x | 8.7x | 7.5x | 5.7x | 3.7x | 1.0x |
| 125B chunk I/O seconds at 3.1 GB/s | 3.1 | 5.8 | 10.0 | 15.3 | 19.7 | 21.4 |
| chunk fill time at 7 tok/s typing (s) | 1.1 | 2.3 | 4.6 | 9.1 | 18.3 | 71.4 |
| keeps up with typing | no | no | no | no | no | yes |
| TTFT speedup after send (cold = 21.4 s, I/O bound) | 0.17x | 0.18x | 0.23x | 0.35x | 0.84x | 1.0x |

Typing-speed sensitivity (best chunk): 1 tok/s gives 6.9x (c=8), 2 tok/s 5.0x (c=8), 3.5 tok/s 1.35x (c=64), 7 and 14 tok/s 1.0x. Cold 500-token prefill: 66 GB read, 21.4 s I/O vs 19.3 s compute at the batched 311 GFLOP/s.

Verdict: **refuted for normal typing, partly true for very slow typists.** The chunk I/O ratio (8.3x at c=16, 5.3x at c=32) matches the screen's 5-10x. At 7 tok/s no sub-batch chunk keeps up, the disk is busy 1.1-2.8x the typing time (so it also competes with decode), and the best TTFT speedup is 1.0x. Only at 1-2 tok/s typing does a small-chunk pipeline give 5-7x. Decode tok/s is unchanged by construction (not a speed lever). Traces are decode routing, used as a proxy for prefill routing; trace length is 249-299 tokens, not 500.

Published methods vs measured: OpenAI forum and the Google patent only establish that the idea is requested / patented for other products [corrected after the paper check: OpenAI now documents prompt_cache_options.prewarm (GPT-5.6+) for a known fixed prefix, and Anthropic's cookbook describes typing-time cache warming for fixed context, so a prewarm is shipped for fixed prefixes, not for the typed prompt]; the KV disk-vs-recompute study says disk reload does not pay below about 50K tokens, which we did not test [corrected after the paper check: that is a GPU-prefill result; with CPU prefill (~19-21 s per 500 tokens) reloading a saved KV cache wins at any length]; SpecPrefill is a different technique; LMCache's 13.6x TTFT is multi-turn prefix reuse on a server, not typing-time. None is reproduced.

Catalog rows mapped (`fam_s2_13`):
- P-0204 [screen 2 #13] idea row
- P-0205 OpenAI forum request for prefill-only requests and cache warm
- P-0206 Google patent family, fulfillment ahead of selecting an autocomplete suggestion
- P-0207 KV cache disk retrieval vs recompute (photonic-CXL study)
- P-0208 Speculative Prefill (SpecPrefill)
- P-0209 LMCache shared MoE KV pool

Unmeasured: real batched prefill expert reuse (traces are decode), KV persistence cost, page-cache warm effects on the first decode tokens, real typing-speed distributions.

---

## Family 14 [screen 2 #14]: per-sequence ARC expert cache

Screen claim: 1.3-4x; screen corrected: about 0 beyond the best path; expected ARC within +0.00-0.02 of LRU, Belady +0.05-0.08.

| Footprints | 1 | 2 | 2.1 (8 GB) | 3 | 4 | 7.6 (16 GB) |
|---|---|---|---|---|---|---|
| LRU hit (mean of 3) | 0.306 | 0.549 | 0.567 | 0.686 | 0.793 | 0.951 |
| ARC minus LRU (mean; min..max over traces) | +0.018 | +0.012 (0.007..0.018) | +0.009 (0.002..0.014) | +0.005 (-0.001..0.010) | +0.012 | +0.001 |
| Belady minus LRU | +0.299 | +0.221 | +0.214 | +0.174 | +0.118 | +0.010 |
| offline hot list (learned on the other traces) | 0.141 | 0.291 | 0.304 | 0.425 | 0.536 | 0.779 |
| ARC / LRU serial tok/s | 1.015 | 1.012 | 1.009 | 1.006 | 1.016 | 1.002 |

Verdict: **refuted (as a speed lever), with one disagreement.** ARC adds 0.005-0.012 hit at 2-3 footprints, below the 0.03 kill line, worth 0.6-1.6% throughput. The traces are single sequences, so plain LRU is already "per sequence", and an offline hot list is far worse (0.30 vs 0.57 at 8 GB). Disagreement with the screen: Belady minus LRU is 0.17-0.22 at 2-3 footprints, not 0.05-0.08; the offline optimum is far above LRU, but it needs future knowledge, so it is an upper bound on predictor-driven policies, not a reachable gain. The network tier is slower than local NVMe on 1 / 10 GbE (0.117 / 1.15 GB/s vs 3.11).

Published methods vs measured: MoE-Infinity (3.1-16.7x latency gains) compares with GPU-offload baselines, not with an SSD path that already has LRU; DAOP decides placement from the prefill activation pattern per sequence, which our in-sequence LRU approximates for free; OD-MoE and the RPC / mac-tensor rows are about network tiers, which we only bound by link rate.

Catalog rows mapped (`fam_s2_14`):
- P-0210 [screen 2 #14] idea row
- P-0211 MoE-Infinity (request/sequence-level activation tracing for expert prefetch and cache)
- P-0212 DAOP: per-sequence activation pattern from prefill decides expert placement
- P-0213 OD-MoE: distributed edge-node on-demand expert loading
- P-0214 llama.cpp RPC backend / mac-tensor (experts over network among Macs)

Unmeasured: ARC with prefetch, multi-sequence behaviour, any real network tier. Traces are about 250-300 tokens, so cold-start misses weigh more than in a long session.

---

## Family 15 [screen 2 #15]: all experts in a second machine's RAM (hardware)

Screen claim: 1.4-4x (3.8 tok/s from 1.3 GB at 5 GB/s). Screen corrected: 0.04-0.4x on 1-10GbE, 0.9-1.6x only with RDMA / TB5.

Tok/s serial / ideal-overlap bound (compute ceiling 4.58):

| Link (GB/s) | uncached | with 8 GB LRU (hit 0.567) | vs local NVMe 3.11 GB/s (serial / overlap) |
|---|---|---|---|
| 1GbE (0.117) | 0.09 / 0.09 | 0.20 / 0.21 | 0.08 / 0.05 |
| 2.5GbE (0.29) | 0.21 / 0.22 | 0.46 / 0.52 | 0.19 / 0.11 |
| 10GbE (1.15) | 0.74 / 0.89 | 1.42 / 2.05 | 0.56 / 0.45 |
| TB-class (2.8) | 1.47 / 2.16 | 2.39 / 4.58 | 0.95 / 1.00 |
| RDMA (5.0) | 2.09 / 3.86 | 3.03 / 4.58 | 1.21 / 1.00 |
| local NVMe (3.11) | 1.58 / 2.40 | 2.51 / 4.58 | 1.00 / 1.00 |

Reading 1.3 GB takes 11.1 s (1GbE), 4.5 s (2.5GbE), 1.13 s (10GbE), 0.46 s (TB-class), 0.26 s (RDMA), 0.42 s (local NVMe); the screen's 0.3 s kill line needs 4.32 GB/s. With 16 / 32 GB local caches (hit 0.95 / 0.96) the machine is compute-bound and a remote tier adds nothing.

Verdict: **needs hardware; the arithmetic part confirms the screen.** Uncached 10GbE (0.74 / 0.88) and RDMA (2.09 / 3.85) reproduce the screen's table. Only an RDMA-class link beats local NVMe (1.21x serial, 1.0x at the overlap bound). The link rates are assumptions, not measurements.

Published methods vs measured: Geerling's exo + RDMA run (DeepSeek V3.1 21.1 / 27.8 / 32.5 tok/s on 1 / 2 / 4 nodes) shows the cluster works when compute is large; it ships no per-token weights from RAM to a thin client. OD-MoE (AAAI listing, transport unconfirmed), WDMoE (Wi-Fi 54 Mbps bottleneck) and llama.cpp RPC (4-8 tok/s for 200B+ MoE on 10GbE by shipping activations) all keep weights in place and move activations; none is reproduced here.

Catalog rows mapped (`fam_s2_15`):
- P-0215 [screen 2 #15] idea row
- P-0216 exo 1.0 with RDMA over Thunderbolt 5 on Mac Studio clusters (Jeff Geerling test)
- P-0217 OD-MoE, distributed on-demand expert loading without caches, ten-node testbed
- P-0218 WDMoE and the Wi-Fi collaborative edge paper, experts on worker devices
- P-0219 llama.cpp RPC backend (TCP) benchmarks and 10GbE guide

Unmeasured: throughput and round-trip time of any real link, a second host with 69+ GB RAM, the top-4 add-on (quality).

---

## Family 16 [screen 2 #16]: active experts 10 to 4 plus 2-bit experts

Screen claim: 2.5x on the compute ceiling (4.6 to 11.5 tok/s). Screen corrected: ceiling 6.0 tok/s (1.3x); 2.4x serial / 1.3-1.4x overlap at 8 GB; 1.4x at 16 / 32 GB; quality gated.

| Quantity | Measured |
|---|---|
| compute ceiling, k = 10 / 8 / 6 / 4 / 2 | 4.58 / 5.02 / 5.54 / 6.19 / 7.02 tok/s |
| ceiling lift at top-4 (dense share fixed) | 1.35x (claimed 2.5x; 11.5 tok/s unreachable) |
| trace-driven, ideal-overlap gain, 4-of-8 / 3-of-8 experts (10 to 4 lies between) | 1.28x / 1.37x on all machines |
| trace-driven serial gain, 8 GB, 4.3 bit / 2.5 bit, 4-of-8 | 1.61x / 2.10x; 3-of-8: 1.88x / 2.32x |
| trace-driven serial gain, 16 / 32 GB, 4-of-8 / 3-of-8 | 1.28-1.34 / 1.38-1.44x (2-bit adds at most 0.06) |
| combo_model gain top-4, 16 / 32 GB (serial and overlap) | 1.38-1.45x |
| combo_model, 8 GB serial top-4 4.3 bit / 2.5 bit | 2.15x / 2.43x |
| prior OLMoE perplexity (earlier runs, not repeated): top-6 / top-4 / Q2_K / top-4 + 2-bit | +8.8% / +45.6% / +26% / +83.5% (gate 5%) |

Verdict: **partly confirmed: the corrected 1.3-1.4x lift is reproduced, the 2.5x claim is refuted, and the quality gate fails.** Dense work (57% of FLOPs by the model's split, an estimate) limits the ceiling lift to 1.35x. The 8 GB serial gain (1.6-2.4x) comes mostly from the cache holding more footprints; at the ideal-overlap bound it is 1.28-1.37x. The decisive model test (perplexity and decode speed with the top-k override on OLMoE) was not run.

Published methods vs measured: Elbow routing (adaptive k, 5.3% latency reduction, no accuracy loss) and HookMoE (over 50% fewer active experts, 1.42x prefill, about 2.5% degradation) show gentler k reductions keep quality, but with calibration; MC-MoE combines 2-bit mixed quant with pruning; the Qwen 3.5 122B 8-to-4 post needs more training. Our R16 row found the same ceiling (4.57 to 5.4-6 tok/s, 1.3x at 8 GB). We reproduce the speed side only; none of the quality claims.

Catalog rows mapped (`fam_s2_16`):
- P-0220 [screen 2 #16] idea row
- P-0221 Elbow-based MoE routing (OLMoE)
- P-0222 HookMoE
- P-0223 MC-MoE (mixed 2-bit expert quant + dynamic pruning)
- P-0224 Qwen 3.5 122B 8-to-4 experts post
- P-0225 Our own first screen R16 (top-k 10 to 6 or 4)

Unmeasured: perplexity at k = 8 / 4 / 2 on the same OLMoE Q4_K_M (the file is not on disk; only Q2_K, Q3_K_S, Q8_0 are), decode speed vs k on a real model, fine-tune recovery, dequant cost of 2-bit.

---

## Family 17 [screen 2 #17]: shared SVD basis plus rank-r residual per expert (needs weights)

Screen claim: 2x-10x, FLOPs 0.28 vs 12 GFLOP (40x), conditional on residual energy at most 10% at r = 64. Screen corrected: at most about 2x at 8 GB, 1.3-1.7x at 16 / 32 GB.

Reading used: W_e = C_e B + U_e V_e (shared B of r x n, coefficients C_e of m x r, residual U_e V_e of rank r); variant B is residual only. Per matrix of the 30B shapes (gate 768 x 2048):

| Rank r | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| elements stored vs dense (variant A) | 3.7% | 7.3% | 14.6% | 29.2% | 58.4% |
| bytes vs 4.3 bpw dense with fp16 factors (A) | 0.14 | 0.27 | 0.54 | 1.09 | 2.17 |
| expert FLOPs vs dense (A) | 0.039 | 0.077 | 0.154 | 0.308 | 0.617 |
| whole-model ceiling lift (A, dense part fixed) | 1.71x | 1.67x | 1.58x | 1.43x | 1.20x |

Break-even rank where factor bytes equal the dense matrix: 150 (fp16 factors), 300 (fp8), 559 (4.3-bit). Even with free expert FLOPs the ceiling lift is capped at 1.76x (dense share). Structure-free Gaussian matrices keep 66% / 43% / 68% residual energy at ranks 17% / 33% / 12.5% of the smaller dimension, so the 10% residual premise at r = 64 (8% of the rank of a 768-row matrix) needs strong structure in the real weights. Routing usage on the traces: 38% of a layer's experts are never routed to in a 250-300 token trace; the hottest half carries 98% of accesses (the zero-rank idea of RFID-MoE only applies to experts that are cold over a long run, not shown by a short trace) [corrected after the paper check: RFID-MoE does not give never-activated experts zero rank; the paper floors every group at rank 1 (Eq 13)].

Verdict: **needs a model (weights); the closed-form part says at most about 1.6x.** At r = 64 the storage is roughly 54% of the dense bytes (consistent with the published 40-60% compression) and the ceiling lift is 1.58x, not 40x. Whether the residual energy is 10% is unmeasured.

Published methods vs measured: D2-MoE (40-60% compression, truncation-aware SVD on deltas), Sub-MoE (keeps 96% / 86% of Mixtral at 25% / 50% reduction), MoBE (argues SVD baselines lose quality), RFID-MoE and REAL (activation-aware truncated SVD, accuracy cost at 50%) agree with our closed-form range (about 50% compression, roughly 1.5x) and none reaches 90%+ compression. We reproduce only the arithmetic.

Catalog rows mapped (`fam_s2_17`):
- P-0226 [screen 2 #17] idea row
- P-0227 D2-MoE: delta decompression, shared base plus per-expert SVD-compressed delta
- P-0228 Sub-MoE: group experts, joint SVD with shared U per group
- P-0229 MoBE: Mixture-of-Basis-Experts
- P-0230 RFID-MoE: rank allocation by effective rank and routing frequency
- P-0231 REAL: low-rank compression of the weakest experts

Unmeasured: residual energy of real expert tensors at rank 64 (needs dequantised weights; no numpy or gguf package, and a pure-python SVD of a 512-expert stack is infeasible), quality after truncation, kernel cost of the rank-r path, the 125B's matrix shapes (30B shapes used).

---

## Summary

| Family | Verdict | Key number |
|---|---|---|
| s2_10 | partly (claim lowered) | straggler 1.27; dense-limited compute gain 1.19x (not 1.59x); 8 GB 1.68x serial / 1.10x overlap |
| s2_11 | refuted as a gain (about 1.0x) | max 1.11x serial, 1.00x overlap; 16 GB peer holds 11 of 38 layers |
| s2_12 | needs model (closed form refutes 2.6x) | gate-first cap 1.5x at d = 0.5; persistence 0.43 double-counts LRU 0.57 |
| s2_13 | refuted at 7 tok/s typing | chunk I/O 8.3x at c = 16; best TTFT speedup 1.0x; 5-7x only at 1-2 tok/s |
| s2_14 | refuted (as speed lever) | ARC minus LRU +0.009 at 8 GB; Belady minus LRU +0.21 (screen expected 0.05-0.08) |
| s2_15 | needs hardware (arithmetic confirms) | only RDMA beats local, 1.21x serial, 1.0x overlap |
| s2_16 | partly (1.3-1.4x yes, 2.5x no) | ceiling lift 1.35x; quality gate fails (+83% prior) |
| s2_17 | needs model (closed form about 1.6x) | r = 64: 54% of bytes, 0.154 expert FLOPs, ceiling lift 1.58x |

Open items for the project: the dead-fraction measurement (family 12), top-k perplexity and speed on a real model (16), and the one-layer SVD energy (17) all need a numpy-capable environment or a built eval tool; family 15 needs two hosts with a real link.


## Corrections from the deep paper check (papers/DEEP_SUMMARY.md)
- RFID-MoE does not give never-activated experts zero rank; the paper floors every group at rank 1 (Eq 13).
- "Reload does not pay below 50K tokens" is a GPU-prefill result; with CPU prefill (~19-21 s per 500 tokens) reloading a saved KV cache wins at any length.
- Top-k quality: renormalisation convention matters (denominator anchored to the native top-10 mass recovers 10->5 from 2.10 to 0.55 MMLU points on a 512-expert model); our OLMoE figure uses a router without renormalisation.
- Family 12: 50% magnitude sparsity is nearly free on SwiGLU (TurboSparse Table 3); 80-90% costs 7+ points.
