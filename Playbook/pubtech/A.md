# Published-technique family tests, group A (brainstorm screen 1, ideas #0 to #7)

Code: <tinyram-llm>/tools/pubtech/f_A.py (functions fam_s1_0 .. fam_s1_7, dict FAMILIES). Test: <tinyram-llm>/tests/test_pubtech_A.py (169 checks, about 60 s on the loaded Mac; all pass).
Data: the three Qwen3-30B-A3B router traces (128 experts, top-8, 48 layers, 249 to 299 decode tokens each: db, story, code). They stand in for the 125B (512 experts, top-10): bytes are mapped by token footprints (one token = 384 trace accesses = 480 slices x 2.7 MB = 1.296 GB). Caches: 8/16/32 GB machines = 2.07/7.58/20.8 footprints (2.5/9.15/25.15 GiB). Throughput constants are the screens' own: explicit 4 MiB reads 2.4 GB/s, decode 55 GFLOP/s, 12 GFLOP per token (compute ceiling 4.58 tok/s), prefill 311 GFLOP/s.
No model was run and nothing was built: every number is computed from the traces or is closed-form arithmetic. The installed llama.cpp build has no routing-trace patch, and OLMoE exists here only as Q8_0/Q3_K_S/Q2_K, so the model-run steps of the decisive tests were not done and are listed as unmeasured per family.

Common caveats. (1) 250 to 300 tokens per trace: the LRU hit rate saturates at 0.962 for large caches because of cold-start misses only. (2) 30B routing locality is assumed to transfer to the 125B; the uniform-routing union bound is 0.81 (128-choose-8, k=8) versus 0.93 (512-choose-10), so a real 125B union is probably larger than the measured 0.455. (3) Block simulator: a block of k positions loads, layer by layer, the union of the experts of k positions through one shared LRU. With acceptance a below 1, the accepted prefix n has P(n >= j) = a^(j-1); the block advances by n. Mode "oracle" loads the next k TRUE tokens (optimistic: a wrong draft is assumed to need the experts of the tokens that really follow); mode "wrong" loads the true tokens for the accepted prefix plus the experts of random other tokens of the same trace for the rejected positions (pessimistic). Real drafts lie between. A differential check ties k=1 of this simulator to the repo's cachesim.lru_hits exactly.

## Summary

| family | idea | verdict | key number |
|---|---|---|---|
| s1 #0 | k=8 draft tree, expert-major union verify | refuted vs the LRU path (screen: drop) | LRU-relative bytes per accepted token >= 1.002 at acceptance <= 0.9; only the all-accepted k=8 cell beats LRU (0.90, i.e. 1.11x) |
| s1 #1 | decode k tokens as a group | refuted (screen: drop) | best grouping gain over k=1 LRU 1.04x; drafted best 0.96x; kill rule (needs 1.3x) met |
| s1 #2 | 500-token prefill as expert-major work | needs model (proxy done) | decode-trace window union 0.55 at 128 tokens; gain vs best path 1.0x by construction; vs mmap 2.9-7x depending on assumptions |
| s1 #3 | route 6 of 10 experts instead of 10 | partly confirmed (bytes yes, quality unmeasured) | K=6 vs K=8 misses 1.46-1.52x fewer (reproduces the screen exactly); Amdahl compute gain 1.19x; quality loss unmeasured |
| s1 #4 | k same-sequence positions expert-major | refuted (screen: drop) | k=8 vs k=1 at 2 footprints 1.01x; k=4 1.35x worse; flat at 4 footprints |
| s1 #5 | small draft + union verify + confidence gate | partly: bytes refuted, speed conditional | bytes gain at a=0.7 <= 1.00x (needs 1.3x); tok/s gain 1.31-1.60x at k=2 only if batch-2 compute runs at 311 GFLOP/s (unmeasured) |
| s1 #6 | router with temporal stickiness | partly (hit gain real, prefill and tok/s claims refuted) | f=0.5: hit 0.763 vs 0.549 at 2 footprints = 1.33x serial / 1.12x overlap; 128-token window union still 0.49 of experts (claim: 0.125) |
| s1 #7 | mixed 2/3/4-bit experts | partly (small win at 8 GB, ~nothing above; size claim refuted) | 8 GB serial 1.49x / overlap 1.08x for 2.6 bpw; 33 GB at 2.06 bpw still does not fit the 27 GB cache |

## Family s1 #0: k=8 lookahead/draft tree, expert-major verify

Idea. Verify a k=8 draft tree by loading each expert slice once per layer for all positions, so the verify I/O is the union (29.1 of 64 slots per layer on the 30B traces) instead of 8 token footprints. Screen claim: 1.3 -> 0.59 GB per token (2.2x), only at acceptance near 1; screen's own multiplier vs the best path (explicit reads + LRU): 0.4-1.0x, verdict drop.

Measured (per-layer LRU as in the screen, mean of 3 traces; ratio = misses per accepted token at block k divided by misses per token at k=1 with the same cache; above 1 = worse).

| quantity | measured | screen |
|---|---|---|
| unique experts per layer, k=2 / 4 / 8 | 12.58 / 19.62 / 29.07 | 12.6 / 19.6 / 29.1 |
| uncached GB per accepted token, k=8, a = 0.5 / 0.7 / 0.8 / 0.9 / 1.0 | 2.36 / 1.50 / 1.13 / 0.83 / 0.59 | 2.38 / 1.51 / 1.14 / 0.83 / 0.59 |
| uncached break-even acceptance vs 1.296 GB | 0.76 | 0.75 |
| 2.1 fp, k=8, a=0.9, true tokens / wrong-branch | 1.29 / 1.43 | 1.34-1.38 (E=5.7) |
| 2.1 fp, k=4, a=0.8, true tokens / wrong-branch | 1.24 / 1.68 | 1.61 (E=3) |
| 2.1 fp, k=8, a=0.6, true tokens / wrong-branch | 3.13 / 3.60 | 2.7 (E=3.1) |
| 4 fp, k=8, a=0.9, true / wrong | 1.12 / 1.76 | 1.31-1.44 |
| smallest ratio over a <= 0.9, all 36 such cells | 1.002 | never below 1.0 |
| all-accepted k=8 at 2.1 fp | 0.90 (1.11x fewer bytes) | "in no case did the union beat the cache" |
| compute ceiling of a k=8 verify at 311 GFLOP/s, a=0.8 | 13.5 tok/s | about 13 |

Verdict: refuted as a gain over the best path. The screen's arithmetic is reproduced to within a few percent. One disagreement: at acceptance exactly 1.0 and a 2.1-footprint per-layer LRU, k=8 grouping does beat k=1 by 1.11x (0.90), contradicting the sentence that the union never beats the cache; it is the single cell of 42, and it needs perfect acceptance, so the kill rule (ratio at or above 1.0 for a <= 0.9) still holds. Published methods claim 10-30% (MoE-Spec), up to 1.62x (EcoSpec), up to 2.34x (MoE-SpeQ) on GPU/PCIe offload without an SSD LRU in front; our data say the byte side alone gives no gain over an LRU, so any win must come from batch compute amortisation, which is unmeasured (needs hardware: batch-8 GFLOP/s on this CPU; kill if under about 100). Unmeasured: batch-8 compute rate, real draft acceptance for a 125B, real wrong-branch experts.

Catalog method rows (all mapped to the fam_s1_0 test):
- P-0002  MoE-Spec: Expert Budgeting for Efficient Speculative Decoding (Feb 2026) -> fam_s1_0 test (the paper's speedup is on GPU offload with a draft; our test measures only the union-vs-LRU byte part)
- P-0003  EcoSpec: Less Experts, Faster Decoding (2607.12696) -> fam_s1_0 test (the paper's speedup is on GPU offload with a draft; our test measures only the union-vs-LRU byte part)
- P-0004  DraftExpert: self-speculative decoding for end-device expert offload (2607.24434) -> fam_s1_0 test (the paper's speedup is on GPU offload with a draft; our test measures only the union-vs-LRU byte part)
- P-0005  SpecMoEOff: hiding offloading latency with speculative decoding (2508.21706) -> fam_s1_0 test (the paper's speedup is on GPU offload with a draft; our test measures only the union-vs-LRU byte part)
- P-0006  SP-MoE (2510.10302), MoE-SpeQ (2511.14102) -> fam_s1_0 test (the paper's speedup is on GPU offload with a draft; our test measures only the union-vs-LRU byte part)
- P-0007  S2-MoE self-speculative decoding on edge, built on llama.cpp (2608.15018) -> fam_s1_0 test (the paper's speedup is on GPU offload with a draft; our test measures only the union-vs-LRU byte part)
- P-0008  SpecExec (2406.02532) -> fam_s1_0 test (the paper's speedup is on GPU offload with a draft; our test measures only the union-vs-LRU byte part)

## Family s1 #1: decode k tokens of one sequence as a group

Idea. Union of experts of a k-token group (about 45% of 8k slots at k=8) cuts disk bytes per token at 8 GB. Screen: about 1.0x vs the LRU path, drafted groups worse (1.19 vs 0.44 footprint per accepted token at a=0.7), verdict drop.

Measured (global LRU, mean of 3 traces, footprints missed per token):

| cache (footprints) | k=1 | k=2 | k=4 | k=8 (all accepted) |
|---|---|---|---|---|
| 2.1 | 0.433 | 0.438 | 0.602 | 0.455 |
| 3 | 0.314 | 0.314 | 0.301 | 0.447 |
| 4 | 0.207 | 0.208 | 0.206 | 0.203 |
| 8 | 0.0446 | 0.0447 | 0.0447 | 0.0446 |
| 0 (cold union) | 1.000 | 0.787 | 0.614 | 0.456 |

Drafted (wrong-branch, misses per accepted token in footprints): 2.1 fp, k=8: 2.15 / 1.31 / 0.70 at a = 0.5 / 0.7 / 0.9 (screen: 1.19 at a=0.7, k=8, counted as union-of-8 over 3.1 accepted); k=2, a=0.9: 0.48. Best gain over k=1 LRU anywhere in the grid: 1.04x (all accepted, 3 fp, k=4); best drafted gain 0.96x. Kill rule (no cell gives 1.3x) holds.
Verdict: refuted. Screen's 0.43-0.47 cold union and 0.40-0.46 baseline are reproduced; the drafted cost per accepted token is 1.31 here against the screen's 1.19 (different accounting: LRU misses with wrong-branch loads vs raw union), both far above the 0.43 baseline. Published: Cascade (up to 3x data loaded for 3 drafts, slowdown up to 1.5x) is consistent in direction; MoE-Spec, AcceptMoE (2.06x under physical offload), MoE-SpeQ gains are on GPU/PCIe offload and are not comparable. Unmeasured: real accepted length from a draft model; llama-speculative miss bytes on OLMoE (needs model).

Catalog method rows (all mapped to the fam_s1_1 test):
- P-0039  Utility-Driven Speculative Decoding for MoE (Cascade) -> fam_s1_1 test (same union-vs-LRU byte part; the paper numbers are GPU-offload baselines without an LRU in front)
- P-0011  MoE-Spec: expert budgeting for speculative decoding -> fam_s1_1 test (same union-vs-LRU byte part; the paper numbers are GPU-offload baselines without an LRU in front)
- P-0012  AcceptMoE -> fam_s1_1 test (same union-vs-LRU byte part; the paper numbers are GPU-offload baselines without an LRU in front)
- P-0013  MoE-SpeQ / SP-MoE / DraftExpert / S2-MoE (speculation plus expert prefetch) -> fam_s1_1 test (same union-vs-LRU byte part; the paper numbers are GPU-offload baselines without an LRU in front)
- P-0014  Hiding offloading latency with speculative decoding -> fam_s1_1 test (same union-vs-LRU byte part; the paper numbers are GPU-offload baselines without an LRU in front)

## Family s1 #2: prefill as expert-major work

Idea. Group a 500-token prompt's tokens by routed expert per layer and read each needed slice once. Screen: TTFT 3.5-6x vs plain mmap, about 1.0x vs best path, decode 1.0x; decisive test is a patched OLMoE prefill trace (not possible here: no tracing build, so needs = model).

Computed locally.
- Proxy (decode traces, non-overlapping windows): fraction of the 128 experts touched in a window of W tokens: W=1 0.0625, 8 0.227, 16 0.306, 32 0.387, 64 0.464, 128 0.553, 249 0.614 (screen: 0.28-0.33 at 16, 0.52-0.61 at 128). A marginal-iid estimate gives 0.547 at 128 (agrees) and 0.61 at 500, but this cannot see unseen experts and real prefill tokens are more diverse than decode, so the 0.75-0.95 assumed for the 125B is untested. Uniform routing would touch 0.9997, so the routing is far from uniform.
- TTFT arithmetic for a 500-token prompt (expert file 66.4 GB at 512 x 48 x 2.7 MB; the idea quotes 69 GB; compute floor 19.3 s at 311 GFLOP/s): union fraction 0.75 -> 49.8 GB: mmap 146 s, explicit reads overlapped 20.7 s (7.06x), serial 40.0 s (3.66x); at the 1.56 GB/s read rate of 1 MiB reads the overlapped gain is 4.59x. Union 0.95 -> 63.0 GB: mmap 185 s, overlapped 26.3 s, serial 45.6 s (4.07x). So the screen's 3.5-6x sits inside the range 2.9x (serial, union 0.5) to 7.06x (overlapped at 2.4 GB/s); the gain is the read-size ratio 2.4/0.34 and is not caused by the sorting.
- Token-major no-sharing baseline of the claim is 500 x 1.296 = 648 GB, about 10x the union bytes (the screen's "wrong baseline").
- Layer-major (one batch of W tokens) vs token-major LRU misses per token on the decode traces: W=128 at 2.07 fp: 27.7 vs 168.5 (6.1x fewer); at 7.58 fp a 128-token batch is WORSE than token-major LRU (24.6 vs 18.8, ratio 0.76) because its union exceeds the cache; at 20.8 fp both miss only the cold set (ratio 1.0). This says batching beats thrashing only when the cache is small, and llama.cpp already batches prefill (mul_mat_id), so the best path already has this.

Verdict: needs a model run for the decisive number (real 500-token union on OLMoE); the computable parts agree with the screen's verdict (about 1.0x vs the best path, gain vs mmap equal to the read-size lever). Published: EStream, Fate, DAOP say prefill touches most experts (consistent with a high union); Fiddler 1.30x on long prefill is a CPU-compute alternative and not comparable; FlashMoE (4x faster loading vs llama.cpp) and Flash-MoE (2.41 ms/layer I/O) are not reproduced here. Unmeasured: real prefill union, mmap vs explicit-read wall time, 125B union fraction.

Catalog method rows (all mapped to the fam_s1_2 test):
- P-0016  EStream: MoE prefill through expert virtualization on mobile NPUs -> fam_s1_2 test (prefill claim only; our proxy uses decode traces, the paper's prefill numbers are not reproduced)
- P-0017  Fate: fast edge MoE inference via cross-layer gate -> fam_s1_2 test (prefill claim only; our proxy uses decode traces, the paper's prefill numbers are not reproduced)
- P-0018  DAOP: data-aware offloading for MoE -> fam_s1_2 test (prefill claim only; our proxy uses decode traces, the paper's prefill numbers are not reproduced)
- P-0019  Fiddler (ICLR 2025) -> fam_s1_2 test (prefill claim only; our proxy uses decode traces, the paper's prefill numbers are not reproduced)
- P-0020  FlashMoE: SSD I/O bottlenecks with ML cache replacement -> fam_s1_2 test (prefill claim only; our proxy uses decode traces, the paper's prefill numbers are not reproduced)
- P-0021  llama.cpp ggml_mul_mat_id (PR #4406) -> fam_s1_2 test (prefill claim only; our proxy uses decode traces, the paper's prefill numbers are not reproduced)
- P-0022  Flash-MoE (397B on a 48GB MacBook) -> fam_s1_2 test (prefill claim only; our proxy uses decode traces, the paper's prefill numbers are not reproduced)

## Family s1 #3: route 6 experts per token instead of the model's top-k

Idea. Drop the lowest-weight experts per token (6 of 10). Screen: 1.3x (overlap) to 1.9x (serial) at 8 GB together with 2.5 bpw requantisation, 1.2-1.4x at 16 GB, 1.18x at 32 GB; quality high risk and unmeasured.

Measured (ids[:K] of each record, assuming the trace order is gate-weight order). Per-layer LRU, 24 entries per layer (same entries for all K): misses per token K=8: 122.8 / 103.1 / 136.6 (db / story / code); K=6: 80.6 / 69.4 / 93.3; K=5: 60.9 / 54.1 / 72.0. Miss gain of K=6 vs K=8: 1.52 / 1.49 / 1.47 (screen 1.52 / 1.49 / 1.46: reproduced). Global LRU at the real cache sizes (misses per token, K=8 / 6 / 5): 8 GB 168.3 / 117.6 / 94.5 (gain 1.43 / 1.78), 16 GB 18.8 / 13.6 / 12.2 (1.38 / 1.54), 32 GB 14.5 / 12.9 / 12.0 (1.13 / 1.21). The repo's combination simulator agrees (top-6 vs full at 8/16/32 GB: 1.43 / 1.38 / 1.13) and cache-aware 6-of-8 beats rank-order top-6 by another 1.33 / 1.26 / 1.19 (1.90 / 1.73 / 1.35 vs full).
Compute: experts are about 40% of per-token FLOPs (1.3 GB at 4.3 bpw = 2.4 B of 6 B active params), so the ceiling rises 4.58 -> 5.46 tok/s (1.19x) for 6-of-10, 1.11x for 6-of-8, 1.32x for 4-of-10 (screen 1.18x; the vllm-mlx guide measured 1.16x for 8 -> 4 on Qwen3-30B-A3B, a different expert share).
Throughput from the top-k cut alone, K=5-of-8 as the analogue of 6-of-10 (keeps 0.6 of the compute): 8 GB serial 1.44x, overlap 1.29x; 16 GB 1.22x / 1.19x; 32 GB 1.19x / 1.19x. K=6-of-8: 8 GB 1.26x / 1.21x.
Disagreement to note: the screen's "1.3x overlap to 1.9x serial at 8 GB" includes the 2.5 bpw requant; the top-k cut alone gives 1.29x overlap and 1.44x serial.
Quality: not measured here. The earlier OLMoE perplexity loss quoted in the repo's simulator is top-6 +8.8% and top-4 +45.6% (above the 5% gate for top-6); published: Opportunistic Expert Activation shows large drops for Qwen3-30B-A3B when k is lowered to 3; NAEE, LExI, REAP show small losses only for skipping, adaptive k or domain pruning.
Verdict: partly confirmed. The byte and compute arithmetic is right; the idea is a quality-for-speed knob that is only worth keeping if a perplexity gate passes (model run: OLMoE with the expert-count override, not done). Unmeasured: perplexity at K=6, tok/s gain of a lower top-k on this CPU, trace order assumption.

Catalog method rows (all mapped to the fam_s1_3 test):
- P-0024  Not All Experts are Equal (NAEE): expert pruning and dynamic skipping -> fam_s1_3 test (lower top-k byte and Amdahl part; the paper's quality numbers are not re-measured)
- P-0025  Opportunistic Expert Activation: batch-aware routing -> fam_s1_3 test (lower top-k byte and Amdahl part; the paper's quality numbers are not re-measured)
- P-0026  LExI: layer-adaptive active experts -> fam_s1_3 test (lower top-k byte and Amdahl part; the paper's quality numbers are not re-measured)
- P-0027  REAP: router-weighted expert activation pruning (Cerebras, ICLR 2026) -> fam_s1_3 test (lower top-k byte and Amdahl part; the paper's quality numbers are not re-measured)
- P-0028  SliceMoE: bit-sliced expert caching with cache-aware routing -> fam_s1_3 test (lower top-k byte and Amdahl part; the paper's quality numbers are not re-measured)
- P-0029  DyMoE: 8/4-bit/skip per expert for edge devices -> fam_s1_3 test (lower top-k byte and Amdahl part; the paper's quality numbers are not re-measured)
- P-0030  vllm-mlx --moe-top-k guide -> fam_s1_3 test (lower top-k byte and Amdahl part; the paper's quality numbers are not re-measured)

## Family s1 #4: k same-sequence positions processed expert-major

Idea. Read each expert slice once for all k positions. Screen: about 1.0x or lower at 8 GB (LRU already captures the reuse), 2.2x only against a cacheless baseline, 1.3-2x on 32 GB only through batch compute.

Measured (global LRU, miss bytes per token as a fraction of the 1.296 GB full read, all k accepted):

| cache (footprints) | k=1 | k=2 | k=4 | k=8 | k=8 / k=1 |
|---|---|---|---|---|---|
| 0 | 1.000 | 0.787 | 0.614 | 0.456 | 0.456 (2.19x gain) |
| 2 | 0.451 | 0.440 | 0.610 | 0.456 | 1.01 |
| 4 | 0.207 | 0.208 | 0.206 | 0.203 | 0.98 |

k=4 at 2 footprints is 1.35x worse than k=1 (screen 1.3x). With drafts (a=0.7, wrong-branch) bytes per accepted token are 1.23-4.55x those of k=1 in every cell (2 fp: k=2 1.23, k=4 2.35, k=8 2.91).
Batch-compute break-even (serial model; compute rate at which the verify block ties plain cached decode at 55 GFLOP/s): 8 GB k=2 needs 91 GFLOP/s at a=0.7 and 70 at a=0.85; k=4 at a=0.7 and k=8 at a=0.7 can never break even (the extra bytes alone cost more than a whole baseline token); k=4 at a=0.85 needs 566 and k=8 at a=0.85 needs 1403 GFLOP/s (above the 311 prefill rate). 16 GB: k=2 66 (a=0.7), k=4 92, k=8 162. 32 GB (bytes equal): (k/E) x 55, e.g. k=2 a=0.7 65, k=4 87, k=8 140.
Union fraction at k=8 0.454 (screen 29/64 = 0.455). Uniform-routing bound for the target shape is 0.93 (512-choose-10, k=8) against 0.81 for the 30B shape, so the real union is likely larger.
Verdict: refuted (screen: drop). Published DraftExpert (1.45x average, 84-87% acceptance, 86-88% prefetch hit), AcceptMoE (2.06x, 73.6-77.1% less traffic vs standard spec decoding) and MoE-Spec are gains over spec decoding or an offloading baseline, not over an LRU path; the Reddit post (low reliability) that spec decoding is dead under SSD offload agrees with our byte numbers. Unmeasured: batched-matmul GFLOP/s at n=2..8 (hardware), prompt-lookup acceptance on OLMoE (model run), the 125B union.

Catalog method rows (all mapped to the fam_s1_4 test):
- P-0032  DraftExpert (self-speculative decoding for end-device MoE, flash-resident experts) -> fam_s1_4 test (union-vs-LRU byte part; paper speedups are not reproduced)
- P-0033  AcceptMoE (verifier expert sets under offloading) -> fam_s1_4 test (union-vs-LRU byte part; paper speedups are not reproduced)
- P-0034  MoE-Spec (budgeted expert set during verification) -> fam_s1_4 test (union-vs-LRU byte part; paper speedups are not reproduced)
- P-0035  Reddit snapshot claiming spec decoding is dead under SSD offload (low reliability) -> fam_s1_4 test (union-vs-LRU byte part; paper speedups are not reproduced)

## Family s1 #5: small resident draft, union verify, plus a confidence gate

Idea. Draft k tokens with a small model, load the union once, verify in one batch; the gate lets the draft emit tokens without the big model. Screen: lossless 1.1-1.5x, gate-skip up to 2.5x but lossy; "17x" and "11 tok/s" claims do not survive.

Measured. Unique experts per layer for k = 2 / 3 / 4 / 8: 12.58 / 16.31 / 19.62 / 29.07 (screen 12.6 / 16.3 / 19.6 / 29.1). Plain uncached explicit-read decode 1.32 tok/s (screen 1.31). Uncached speculation with verify compute at 311 GFLOP/s, using the screen's acceptance convention (k drafts plus a bonus token): k=2, a=0.7 2.36 tok/s (screen 2.35), k=3 2.08, k=4 1.88, k=8 1.41 (screen 1.40): reproduced. With the stricter convention (k positions verified, expected accepted (1-a^k)/(1-a)) the same cells are 1.84 / 1.80 / 1.71 / 1.38, so the "1.8x" depends on whether the bonus token is counted.
With an LRU in front (2.1 and 3 footprints) the BYTES gain per accepted token at a=0.7 is at most 1.00x over every k, in either mode (k=2 true-token 0.99, wrong-branch 0.78; k=3 0.73 / 0.53; k=4 0.46 / 0.41; k=8 0.37 / 0.33). Kill rule (needs 1.3x at some k) is met: the byte side is dead. The tok/s gain comes only from the assumption that a verify block runs at the 311 GFLOP/s prefill rate: k=2, a=0.7, 2.1 fp gives 1.60x (true tokens) or 1.31x (wrong-branch) over cached k=1, while k=4 and k=8 lose (0.79 / 0.61). Idea #4's break-even table shows batch-2 needs only about 91 GFLOP/s at 8 GB, so the k=2 case is a plain speculative-decoding question about batch compute, which is unmeasured.
Gate arithmetic (60% of tokens taken from the draft): compute-only ceiling 11.5 tok/s (the "11 tok/s" figure); with the I/O of the remaining 40% uncached 3.30 tok/s, with a 2.1-footprint LRU 5.53 tok/s; those tokens are unverified (quality is the draft's) and skipped positions need KV backfill, not computed.
Verdict: partly. Bytes gain refuted (<= 1.00x), the screen's 17x and 11 tok/s refuted by arithmetic, small-k lossless speculation remains a compute-side question. Published: MoE-Spec (10-30% over EAGLE-3), SP-MoE, Cascade, Faster Cascades, the hackmd test (no mode beat baseline) are not reproduced; the single-tester hackmd result agrees in direction with our bytes finding. Unmeasured: real acceptance, draft/big agreement vs confidence (gate), draft RAM and compute, KV backfill cost.

Catalog method rows (all mapped to the fam_s1_5 test):
- P-0037  MoE-Spec: Expert Budgeting for Efficient Speculative Decoding -> fam_s1_5 test (union-vs-LRU bytes plus closed-form gate arithmetic; paper speedups are not reproduced)
- P-0038  SP-MoE: Speculative Decoding and Prefetching for MoE inference -> fam_s1_5 test (union-vs-LRU bytes plus closed-form gate arithmetic; paper speedups are not reproduced)
- P-0039  Utility-Driven Speculative Decoding for MoE (Cascade) -> fam_s1_5 test (union-vs-LRU bytes plus closed-form gate arithmetic; paper speedups are not reproduced)
- P-0040  Faster Cascades via Speculative Decoding (ICLR 2025) -> fam_s1_5 test (union-vs-LRU bytes plus closed-form gate arithmetic; paper speedups are not reproduced)
- P-0041  Single-user llama.cpp MoE speculative test (hackmd) -> fam_s1_5 test (union-vs-LRU bytes plus closed-form gate arithmetic; paper speedups are not reproduced)

## Family s1 #6: router fine-tuned for temporal stickiness

Idea. A temporal-stickiness term makes consecutive tokens reuse experts. Screen: decode 1.3x serial / 1.10x overlap over the best path if 50% swap costs no quality; claims 7x TTFT and 5-8 tok/s are above the physical ceilings.

Measured. Simulated stickiness f: of each token's picks that were not in the previous (already modified) token, a fraction f is swapped for previous-token experts (seeded). Global LRU hit rate, mean of 3 traces:

| f | token overlap | hit 1.5 fp | hit 2 fp | hit 3 fp | 128-token window union |
|---|---|---|---|---|---|
| 0 | 0.432 | 0.454 | 0.549 | 0.686 | 0.546 |
| 0.25 | 0.563 | 0.607 | 0.661 | 0.761 | 0.523 |
| 0.5 | 0.692 | 0.718 | 0.763 | 0.833 | 0.491 |
| 0.75 | 0.826 | 0.842 | 0.865 | 0.902 | 0.435 |
| 1.0 (degenerate) | 1.000 | 0.996 | 0.996 | 0.996 | 0.0625 |

At f=0.5, 2 footprints: hit 0.763 vs 0.549 (screen 0.76 vs 0.54), serial 2.89 vs 2.17 tok/s = 1.33x (screen 1.31x), overlap 1.12x (screen 1.10x): reproduced. Overlap at f=0.5 is 0.69 (screen 0.69). The claimed prefill effect (a 500-token prefill touching about 64 of 512 experts, 12.5%) is not supported: even at f=0.75 a 128-token window still touches 0.43 of the experts, 0.49 at f=0.5. The claimed 5-8 tok/s is above the 4.58 tok/s compute ceiling, and every simulated throughput stays at or under it. The 7x TTFT is capped by 19.3 s of prefill compute.
Verdict: partly. The decode hit-rate gain is real and matches the screen, bounded to 1.3x serial / 1.1x overlap at 8 GB; prefill and 5-8 tok/s claims refuted. Published: ReMoE (+26% reuse, +15.7% LRU hit, TPOT -43.6 to -49.8% on Jetson, quality preserved) is consistent in direction (our f=0.5 hit is +39% relative at 2 fp, a larger perturbation); Cache-Conditional Experts (up to 2x vs LRU, training-free) is the inference-time version; Not All Models Suit Expert Offloading (cache about 2x active experts) matches the 2-footprint operating point; vLLM issue 38256 and ExpertFlow are not reproduced. Unmeasured: any quality effect (needs model runs; a 125B router fine-tune is out of reach), the choice of which picks a trained router would keep, no router logits in the traces.

Catalog method rows (all mapped to the fam_s1_6 test):
- P-0043  ReMoE: router fine-tuning with a temporal term for expert reuse in memory-constrained MoE inference (ICML 2026) -> fam_s1_6 test (the stickiness effect on LRU hit rate, simulated by swapping picks; no router training)
- P-0044  Mixture of Cache-Conditional Experts (Skliar et al., TMLR 2025), training-free cache-aware routing -> fam_s1_6 test (the stickiness effect on LRU hit rate, simulated by swapping picks; no router training)
- P-0045  Not All Models Suit Expert Offloading: local routing consistency of 20 MoE LLMs (ICLR 2026) -> fam_s1_6 test (the stickiness effect on LRU hit rate, simulated by swapping picks; no router training)
- P-0046  vLLM issue 38256: dedup of expert loads in batched prefill -> fam_s1_6 test (the stickiness effect on LRU hit rate, simulated by swapping picks; no router training)
- P-0047  ExpertFlow (arXiv 2410.17954) -> fam_s1_6 test (the stickiness effect on LRU hit rate, simulated by swapping picks; no router training)

## Family s1 #7: mixed 2/3/4-bit experts

Idea. Requantise experts (69 GB to about 34 GB) so the file nearly fits in 32 GB. Screen: 8 GB 1.8x serial / 1.1x overlap, 16 GB 1.07x, 32 GB 1.0x; 34 GB claim wrong (42-48 GB realistic).

Measured (byte-capacity LRU on the three traces, cache sized as footprints of the target at each width; serial / overlap gain vs the 4.3 bpw path at the same RAM):

| cache | 3.0 bpw | 2.6 bpw | 2.06 bpw |
|---|---|---|---|
| 8 GB (2.5 GiB): hit 4.3 bpw 0.562 | 1.35x / 1.08x | 1.49x / 1.08x | 1.72x / 1.08x |
| 16 GB (9.15 GiB): hit 4.3 bpw 0.951 | 1.05x / 1.00x | 1.06x / 1.00x | 1.07x / 1.00x |
| 32 GB (25.15 GiB): hit 4.3 bpw 0.962 | 1.03x / 1.00x | 1.03x / 1.00x | 1.05x / 1.00x |

At the screen's own budgets (4.5 / 12 / 26 GB of cache) 2.6 bpw vs 4.3 bpw gives 1.45x / 1.04x / 1.03x serial and 1.00x overlap. Disagreement: the screen's 8 GB figure 1.8x compares a 4.5 GB cache at 2.6 bpw (hit 0.88) with a 2.5 GiB cache at 4.3 bpw (hit 0.56); at an equal 4.5 GB cache the gain is 1.45x, and at an equal 2.5 GiB it is 1.49x. The 16 and 32 GB gains (1.03-1.07x serial) are cold-start artefacts of a 250-token trace (hit capped at 0.962); the overlapped path is compute-bound at 4.58 tok/s, so the gain is 1.0x.
Sizes: expert file at 3.0 / 2.6 / 2.06 bpw = 48.1 / 41.7 / 33.1 GB (from 69 GB at 4.3). Even 2.06 bpw does not fit the 27.0 GB expert cache of a 32 GB machine, so "nearly fits" is refuted. Speculative k=2 on a 2.6 bpw 8 GB cache (a=0.7, wrong-branch) reads 1.34x more bytes per accepted token than k=1, so the screen's "add up to 1.2-1.4x from spec" is not supported on the byte side.
Quality (quoted from earlier OLMoE perplexity runs recorded in the repo: Q3_K_S +10.4%, Q2_K +26%, requantised so upper bounds): not re-measured.
Verdict: partly confirmed. A real but small 8 GB serial gain (1.35-1.72x depending on width, 1.08x overlapped), about nothing above; accuracy cost is high and unmeasured. Published: Eliseev and Mazur (2-bit Mixtral with LRU), HOBBIT (up to 9.93x vs offloading baselines, low-bit only for low-importance misses), MoE-SpeQ, EVICT, and low-rank compensation are not reproduced (GPU or accuracy results). Unmeasured: accuracy / perplexity of mixed 2/3/4-bit 125B experts, dequant cost of 2-bit kernels, validation of 30B hit rates at 512 experts.

Catalog method rows (all mapped to the fam_s1_7 test):
- P-0049  Eliseev and Mazur, Fast Inference of MoE LMs with Offloading -> fam_s1_7 test (byte-capacity LRU at lower bits per weight; paper accuracy and speed numbers are not reproduced)
- P-0050  HOBBIT mixed-precision expert offloading (on llama.cpp) -> fam_s1_7 test (byte-capacity LRU at lower bits per weight; paper accuracy and speed numbers are not reproduced)
- P-0051  MoE-SpeQ, speculative quantized decoding with expert prefetch -> fam_s1_7 test (byte-capacity LRU at lower bits per weight; paper accuracy and speed numbers are not reproduced)
- P-0052  Adaptive Verification for MoE speculative decoding (EVICT) -> fam_s1_7 test (byte-capacity LRU at lower bits per weight; paper accuracy and speed numbers are not reproduced)
- P-0053  Bandwidth-efficient adaptive MoE via low-rank compensation -> fam_s1_7 test (byte-capacity LRU at lower bits per weight; paper accuracy and speed numbers are not reproduced)

## What could not be done

- No model run: the build here has no routing-trace patch and OLMoE exists only as Q8_0/Q3_K_S/Q2_K, so the OLMoE prefill union (#2), OLMoE perplexity at K=6 and at requantised widths (#3, #7), llama-lookup/llama-speculative acceptance (#4, #5, #1) and batch-n GFLOP/s measurements (#0, #4, #5) were not done. They are hardware or model steps and are marked in each family's 'unmeasured' list.
- All 45 catalog method rows are mapped to a family test above; papers' own speed numbers are compared by mechanism only, none is reproduced.


> Correction from the paper check (papers/deep_P01b.md, supersedes the earlier note): for P-0025 the figures 43.0 -> 22.2 and 51.6 -> 26.5 are average activated experts per layer (Table 4), not accuracy. But the paper's 'pruned' columns are a plain top-k0 cut, and they do show large drops: Qwen3-30B (k=8) at k0=3 AIME 80.4 -> 51.2, GPQA 60.2 -> 45.7, LiveCodeBench 62.1 -> 37.4; k0=4 loses 4.6 AIME and 5.9 GPQA. The batch-16 recovery applies only to the method itself, not to a plain cut. The Cascade row is P-0010, not P-0039.
