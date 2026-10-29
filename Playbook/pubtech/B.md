# Published-technique families, group B (screen 1 #8 to #15)

Code: <tinyram-llm>/tools/pubtech/f_B.py (functions fam_s1_8 ... fam_s1_15, dict FAMILIES). Tests: <tinyram-llm>/tests/test_pubtech_B.py (201 checks, about 35 s; 202 with PUBTECH_IO=1).
Both files pass the leak scan patterns (scanned by hand with the hygiene scanner because the files are not yet tracked) and tests/test_hygiene_data.py passes (171/171).

Common basis and caveats (apply to every section)
- Data: the three Qwen3-30B-A3B decode traces (128 experts per layer, top-8, 48 layers; db 299, story 249, code 249 tokens). Cache sizes are mapped by token footprints (entries = footprints x 384). 8/16/32 GB machines = 2.5/9.15/25.15 GiB of expert cache = 2.1/7.6/20.8 footprints of the 125B.
- Throughput arithmetic uses the screen's constants: 1.3 GB of expert bytes per token at hit 0, compute ceiling 4.57 tok/s, mmap 0.34 GB/s, 1 MiB reads 1.56, 4 MiB reads 2.4, prefill 311 GFLOP/s, 12 GFLOP per token, 69 GB of experts. These are carried estimates, not remeasured, except where a section says an I/O run was made.
- Traces are decode-only (the trace hook skips prefill), so every "prefill" number here is a proxy from decode routing. 30B stands in for the 125B. Traces are short, cold start counted.
- No model was run for this group. The build under runs/llama-work has no prefill trace hook, there is no text corpus on the machine for perplexity, and the Mac was under load (load average 8 to 50) with other agents benchmarking, so model runs were skipped on purpose. Everything that needs a model forward pass, a perplexity number or a GPU is listed as unmeasured.
- I/O numbers (families 9, 14, 15) come from an optional read-only run (PUBTECH_IO=1) against the local Qwen3-30B Q4_K_M file (18.6 GB, bigger than RAM). Python threads, F_NOCACHE preads, mmap "copy" via ctypes memmove (GIL released). The machine was shared and loaded, so rates move by tens of percent between runs (one run gave 8-thread 4 MiB pread/mmap-copy 1.5x, another 0.7x on the same extents). Treat them as order of magnitude only. They are not asserted tightly in the test.

## s1 #8 Pin ~24.5% of experts and bias router logits toward the pinned set

Idea: static pin (26 GB of the 125B) plus a logit bias so about half of each token's experts come from it. Needs: model (quality of the bias). Computed here: all hit-rate and throughput arithmetic.

| Measured (3 traces, mean) | value | screen |
|---|---|---|
| Pin hit learned on the other prompts, 7% / 17% / 24.5% / 40% of pairs | 0.159 / 0.386 / 0.528 / 0.725 | same |
| Optimistic 2-swap bias at those pins (7/17/24.5/40) | 0.409 / 0.631 / 0.760 / 0.901 | 0.41/0.63/0.76/0.90 |
| Same, 4-swap | 0.656 / 0.841 / 0.919 / 0.976 | 0.66/0.84/0.92/0.98 |
| Pin capped by pins actually present in the layer, 24.5%, force 4 of 8 pinned | hit 0.609, 8.1% of selections changed | not in screen |
| Pin picked from the first 16 tokens, 24.5%, scored on the rest | 0.689 | not in screen |
| LRU at 8 GB (2.1 fp), plain / bias m=1 / m=2 / m=4 | 0.562 / 0.680 / 0.789 / 0.945 (selections changed 12% / 24% / 42%) | 0.56 -> 0.78 |
| Static pin at the same 8 GB capacity (13% of pairs) | 0.301 | "4-7% of slices fit as pin" |
| 8 GB serial/overlap tok/s: LRU, LRU+bias m=2, static pin | 2.19/4.21, 3.00/4.57, 1.67/2.64 | 2.19/4.2, 2.96/4.57 |
| Gain of LRU+bias(m=2) over LRU at 8 GB | 1.37x serial, 1.08x overlap | 1.35x, 1.09x |
| Static pin vs LRU at 8 GB | 0.76x serial | "worth little" |

Verdict: partly confirmed. The screen's arithmetic reproduces to the third decimal. The bias adds at most 1.37x serial and 1.08x overlap on top of LRU at 8 GB and about nothing at 16/32 GB (already at the ceiling); a static pin loses to LRU at equal bytes. The cost is quality: lifting hit from 0.56 to 0.79 changes 24% of the router's chosen experts, and 0.95 changes 42%. The substitute quality is not modelled (router scores are not in the trace), so the optimistic model is an upper bound. Published methods claim 0.1-3% perplexity change with trained or tuned routers; that is not tested here.
Unmeasured: perplexity or any quality metric for a bias, 125B (512 experts) pin rates.

Catalog rows mapped to this test:
- P-0054 idea row; P-0055 Cache-Aware Joint Router Adaptation for Memory-Efficient MoE Inference (trained routers, no throughput number to compare); P-0056 llama.cpp discussion: MoE offload to disk with on-demand paging (RFC+PoC) (no bias there, but the slot pool is what the LRU rows model); P-0057 Mixtral-offloading / SpecMD study of speculative prefetch; P-0058 Research Square paper with a '0.15 rule' on a llama.cpp-based engine (2.8-3.0x vs stock on a different machine: not comparable, not reproduced).

## s1 #9 Expert-major prefill

Idea: read each expert once per layer for the whole prompt; claims 650 GB -> 69 GB and 271 s -> 29 s. Needs: model (stock mmap prefill rate and real prefill coverage).

| Measured / computed | value | screen |
|---|---|---|
| Unique experts per layer in first 16/32/64/128/299 tokens, db | 43.2 / 53.2 / 60.5 / 69.2 / 78.8 of 128 | 43/53/60/69/79 |
| Coverage after all tokens: db / story / code | 0.615 / 0.552 / 0.698 | 0.62 / 0.55 / 0.70 |
| Independent-draw model at 500 tokens (mean) | 0.61 (plateaus on the 30B) | n/a |
| Uniform-routing coverage, 125B, 500 tokens | 0.99995 | "about 1.0" |
| TTFT mmap 0.34 GB/s serial, full coverage | 222.2 s | 222 s |
| TTFT explicit 4 MiB overlapped / serial | 28.75 s / 48.0 s | 29 / 48 s |
| Gain vs mmap 0.34, overlap / serial | 7.7x / 4.6x | 7.6x / 4.6x |
| Gain vs mmap at 1 GB/s, overlap / serial | 3.07x / 1.84x | "2.4-3x" |
| With the measured trace coverage (0.62): gain vs mmap 0.34, overlap / serial | 7.5x / 3.9x | n/a |
| Compute floor 500 x 12 GFLOP at 311 / at 55 GFLOP/s | 19.3 s / 109 s | 19 s / 109 s |
| Prompt length where prefill turns compute-bound | 745 tokens | about 750 |

I/O run (shared, noisy): 4 MiB random extents, 8 threads: pread nocache 2.8 GB/s, mmap copy 1.8 GB/s; 0.9 MB extents: 2.2 vs 0.67 GB/s. With those two rates TTFT is 57 s (mmap, serial) vs 24.9 s (explicit, overlapped) = 2.3x. So the 7.6x depends entirely on stock mmap prefill being as slow as 0.34 GB/s; the machine here, reading in parallel from several threads, did not show that. The stock sequential-fault pattern of llama.cpp prefill was not measured.
Verdict: partly. Arithmetic confirmed (including the screen's correction that compute is 19 s, not 109 s). Gain is a read-size/overlap effect, not an expert-major effect; the screen's claim that stock ubatch-512 prefill is already expert-major is not tested here. Coverage is far below 1.0 on the 30B decode proxy, but that is not evidence for 512-expert prefill.
Unmeasured: real prefill routing and coverage; stock prefill mmap rate; expert-major vs ubatch comparison.

Catalog rows: P-0059 idea; P-0060 SeqMoE (chunked prefill repeats transfers; supports the "prefill touches broad expert sets" premise, not reproduced); P-0061 ExpertFlow (token grouping by route); P-0062 MoE-Lightning (CGOPipe, paged weights, batch offloading; 10.3x is large-batch throughput, not single-prompt TTFT); P-0063 fMoE fine-grained expert offloading (TTFT -30-44%; consistent with load-dominated prefill); P-0064 Reddit/blog larger ubatch for gpt-oss-120b (batching in prefill is already standard; no repro). All mapped to fam_s1_9.

## s1 #10 Pin after prefill

Idea: pin the 36% of experts the prompt used. Needs: model (does real prefill routing predict decode better than the first decode tokens). Free proxy computed here with the first W decode tokens.

| W=64, mean of 3, share of pairs | pin | LRU | hybrid (half pin, half LRU) | in-sample pin (oracle) |
|---|---|---|---|---|
| 6% (0.96 fp) | 0.306 | 0.000 (cliff) | 0.196 | 0.406 |
| 10% (1.6 fp) | 0.429 | 0.490 | 0.477 | 0.546 |
| 20% (3.2 fp) | 0.666 | 0.728 | 0.721 | 0.787 |
| 36% (5.8 fp) | 0.870 | 0.937 | 0.934 | 0.954 |

W=16 is worse (pin 0.789 at 36%). A 36% pin learned on other prompts only: 0.685. The claimed "40% reuse" is the hit of a 10% pin (0.43), not of the 36% pin (0.87). Throughput at 8 GB: pin-by-share (hit 0.26) 1.61/2.49 tok/s vs LRU 2.19/4.20 = 0.74x serial, 0.59x overlap (screen 0.6-0.75x). Claimed hit 0.40 gives 1.84 serial.
Verdict: refuted as a gain over LRU, as the screen found: pin is below LRU at every share of 10% and above on all three traces; hybrid minus LRU is -0.003 at 36% (kill threshold for the idea to live is +0.05). A pin only wins below one footprint, where strict LRU is zero.
Unmeasured: actual prefill routing as predictor (needs a prefill trace); 125B share mapping; drift beyond about 300 tokens.

Catalog rows: P-0065 idea; P-0066 ProMoE (prefetch, not pinning; 2.2x claim is a different mechanism); P-0067 DALI (adaptive replacement, opposite of a static pin); P-0068 ELDR (prefill/decode expert correlation on Qwen3-30B-A3B; supports the premise, not tested here); P-0069 Fate (99% hit; different stage split); P-0070 MoE-Infinity (per-request activation matrix; closest prior work); P-0071 In-depth analysis of caching and prefetching in MoE offloading (LRU beats random by 5-15%; consistent with LRU>pin here); P-0072 llama.cpp issue #20757; P-0073 FlashMoE (learned replacement). All mapped to fam_s1_10.

## s1 #11 Prefill-only top-4

Idea: top-4 of 10 only in prefill to cut expert reads and TTFT. Needs: model (prefill tok/s with the override, quality).

| Measured | value | screen |
|---|---|---|
| Unique experts in the first 16 tokens, k=8 vs k=4: db / story / code | 43.2/28.2, 36.6/24.1, 43.3/27.0 | 43.2/28.2 etc. |
| First 64 tokens, k=8 vs k=4: db / code | 60.5/42.0, 69.4/49.1 | 60.5/42.0, 69.4/49.1 |
| Ratio k8/k4 falls with window (sliding windows, db): 16 -> all | 1.55 -> 1.37 | "1.3-1.4x at W=64" |
| Prefill bytes ratio k4/k10: trace-skew model at 500 tokens / uniform 125B | 0.75 / 0.98 | "about 25 s vs 29 s" |
| TTFT gain overlapped, uniform 125B | 1.02x (serial 1.13x) | 1.1-1.2x |
| TTFT gain overlapped if the 30B skew carried over | 1.33x | n/a |
| Prefill compute at k=4 (FLOP factor 0.74) | 14.3 s vs 19.3 s | 14 s |
| Decode misses/token at fixed 2.1 fp cache, k=8 / 4 / 3 (db) | 170.1 / 70.9 / 48.3 | same |
| Decode miss ratio k4/k8; decode compute ceiling at k=4 | 0.425; 6.19 tok/s | 2.2-2.4x fewer; about 6 |

Verdict: refuted for prefill at 125B-like (near-uniform) routing: about 1.0-1.15x, because nearly every expert is still read. It would be 1.3x only if routing at 512 experts were as skewed as the 30B decode traces, which is unverified. The decode side (k=4 everywhere) is a separate knob; see s1 #13. Quality of k=4 not measured here (the paper-cited drop is 77.87 to 70.72 MMLU for 8 to 4 on Qwen3-30B).
Unmeasured: prefill tok/s ratio with the expert-count override; real prefill coverage; gate-score order of the trace ids (assumed ranked).

Catalog rows: P-0074 idea; P-0075 Prefill-stage top-k reduction (AAAA 2026, title not retrieved); P-0076 Lynx (prefill highly sensitive to expert dropping: runs against the idea); P-0077 Alloc-MoE (1.15x prefill at half budget on a GPU-resident model: consistent with a small prefill gain); P-0078 Distribution-consistent inference for dynamic sparse MoE (MMLU 77.87/70.72/27.97); P-0079 Training-free halving of activated experts; P-0080 llama.cpp expert_used_count override. All mapped to fam_s1_11.

## s1 #12 Discrete GPU as a resident tier

Needs: hardware (the 2-2.6x headline requires a GPU). Computed here: the cache-hit side the screen proposed, plus I/O-bound arithmetic.

| Measured | value | screen |
|---|---|---|
| LRU hit at 29% of experts, same prompt: db / story / code | 0.856 / 0.884 / 0.809 | 0.855/0.884/0.809 |
| Warm from another prompt (db, one pairing) | 0.862 | 0.889 |
| Static pin learned on other prompts, 29% | 0.69 / 0.51 / 0.58 | 0.57/0.44/0.70 (screen used one source prompt) |
| Hit below one footprint (6%) | 0.000 (cliff) | cliff |
| Hit at 29% with experts split 1x / 2x / 4x sparser (hash geometry) | 0.850 / 0.825 / 0.789 | keep if >=0.75, kill if <0.5 |
| Unit error check | 278 x 0.9 MB claim gives 6.2 tok/s; 278 x 2.7 MB gives 2.08 | 2.1 |
| 24 GB card (20 GB cache): hit, I/O bound tok/s, dense / 4x sparse | 0.849 / 12.3, 0.789 / 8.7 | about 12 |
| 12 GB card (10 GB cache) | hit 0.596, I/O bound 4.6 (4x sparse 0.565, 4.2) | hit 0.6, about 4 |
| 6 GB card (5 GB cache) | hit 0.431 (4x sparse 0.327), I/O bound 3.2 (2.7) | hit 0-0.2 |
| CPU-only best path, 8/16/32 GB serial | 2.19 / 4.08 / 4.18 (overlap 4.21 / 4.57 / 4.57) | 2.18 / near ceiling |

Verdict: partly. The hit-rate part holds at the 4x sparser geometry (0.79 is above the screen's 0.75 keep line), so a 20 GB tier is I/O-bound at 8.7 to 12 tok/s and the CPU compute ceiling would then bind. The 6 GB case measures a higher hit (0.43) than the screen's 0 to 0.2, because the screen read the cliff at 4 GB; at 5 GB it is already above one footprint on this mapping. The sparse test is pessimistic (sub-experts independent), not a 512-expert model. Whether a GPU lifts the ceiling to the I/O bound is a hardware question; the 40 tok/s GPU figure in the result dict is tagged an assumption and is not measured. The idea corrects to "known and shipped" as the screen says.
Unmeasured: any GPU timing, PCIe costs, OLMoE (64 experts) sweep (no OLMoE routing trace exists in the repo and no model was run), real 125B hit rates.

Catalog rows: P-0081 idea; P-0082 llama.cpp GPU expert cache --moe-cache-mib (34.8 -> 64.4 tok/s on a 4090; hardware, not reproduced); P-0083 KTransformers hot experts on GPU; P-0084 Fiddler CPU-GPU orchestration; P-0085 HybriMoE / HOBBIT / MoE-Infinity / DAOP. All mapped to fam_s1_12; their GPU speed-ups are unmeasured here.

## s1 #13 Top-4 to cut TTFT, dense part on GPU

Needs: model (perplexity or KL, prefill tok/s with override); GPU part needs hardware.

| Measured / computed | value | screen |
|---|---|---|
| Decode misses/token at 2.1 fp, k=8/6/4/3 (db) | 170.1 / 118.1 / 70.9 / 48.3 | 170.1 / - / 70.9 / 48.3 |
| Miss ratio vs k=8: k6, k4, k3 | 0.70, 0.425, 0.29 | 2.2-2.4x fewer at k=4 |
| P(expert touched in 500 tokens), top-4 / top-10 of 512 | 0.980 / 1.000 | 0.98 / 1.00 |
| Compute gain of top-4 with fixed dense part | 1.35x (FLOP factor 0.74) | 1.3x |
| Prefill TTFT gain, overlapped / serial | 1.02x / 1.13x | 1.0-1.1x |
| Decode k=4 at 8 GB, serial / overlap vs k=8 | 3.81 vs 2.19 (1.74x) / 6.19 vs 4.21 (1.47x) | 1.3x / 1.4x |
| 16 GB: serial / overlap gain | 1.39x / 1.36x | 1.2-1.3x |
| 32 GB: serial / overlap gain | 1.35x / 1.36x | 1.3x |

Disagreement with the screen: the decode serial gain at 8 GB is 1.74x here (screen 1.3x). With the cache held fixed in bytes, misses per token fall to 0.43 of the k=8 value (the screen's own trace result), giving 0.24 GB per token, 0.10 s of I/O; the screen's 2.8 tok/s needs about 0.19 s of I/O, which does not follow from its own miss counts. The overlap gain (1.47x vs 1.4x) and the ceiling (6.19 vs about 6.0) agree. Gain shrinks with RAM as expected. These decode numbers carry the quality cost of k=4, cited from the earlier OLMoE run (+45.6% relative perplexity, not re-measured).
Verdict: prefill claim refuted (about 1.0-1.1x, read-bound); decode k=4 partly confirmed as a speed knob (1.4-1.7x) but with a large unmeasured/high quality cost. GPU offload of the dense part not tested.
Unmeasured: any perplexity/KL, prefill tok/s with the override, GPU.

Catalog rows: P-0086 idea; P-0087 LExI layer-adaptive active experts (training-free); P-0088 Expert-Sample (greedy stable at half experts, sampling worse); P-0089 MoDES (2.16x prefill / 1.26x decode, GPU-resident, not disk-bound: not comparable); P-0090 llama.cpp runtime top-k override and --override-kv; P-0091 Hybrid CPU-GPU MoE TTFT (KTransformers prefill rates; hardware). All mapped to fam_s1_13.

## s1 #14 Repack GGUF expert tensors into aligned contiguous extents

Needs: local (done: layout and arithmetic; I/O optional).

| Measured / computed | value | screen |
|---|---|---|
| LRU hit at 8/16/32 GB footprints (2.1/7.6/20.8) | 0.562 / 0.951 / 0.962 [corrected after the paper check: this is the 30B footprint mapping; public benchmarks on the 125B-class model suggest hit 0.72-0.77 at 13-16% of experts, so 0.95 at 16 GB is optimistic] | 0.56 / 0.95 / 0.97 |
| Explicit 4 MiB vs mmap, 8 GB serial / overlap | 4.15x / 7.06x (2.19 vs 0.53; 4.21 vs 0.60) | 4x / 7x |
| 16 GB serial | 1.65x (2.46 -> 4.08) | 1.6x |
| 32 GB serial | 1.52x (2.75 -> 4.18) | 1.1-1.2x (screen lower) |
| One 2.7 MB extent vs three 0.9 MB reads, 8 GB serial | 1.38x | "about 1.2x lower at 1 MiB" |
| Reads per token on the 30B: stock / repacked | 1152 / 384 | 3 per expert / 1 |
| 69 GB load, mmap vs explicit | 203 s vs 28.75 s (7.1x) | 7x |

Stock layout of the local Qwen3-30B Q4_K_M file (real header parse): 48 layers x 128 experts; per-expert slices are 884,736 bytes (gate), 884,736 (up) and 1,290,240 (down), so an expert is 2.86 MB (the screen assumed 3 x 0.9 MB). No expert slice starts on a 4 KiB boundary (tensor data is only 32-byte aligned) so direct I/O needs trimming or the repack; reading unaligned slices at 16 KiB page granularity costs 1.8% extra. The gate and up extents of one expert are 114 MB apart, so the three reads cannot be merged without a repack.
I/O run (shared): pread nocache 0.9 MB x 4 threads 2.5 GB/s, 2.7 MB x 8 threads 2.4 GB/s, 4 MiB x 8 threads 2.8 GB/s; single thread 1.0-1.4 GB/s; mmap copy 8 threads 0.67 / 1.02 / 1.83 GB/s for 0.9 / 2.7 / 4 MiB, single thread 0.15-0.5 GB/s. Ratio pread/mmap 2.4x at 2.7 MB, 8 threads. The 0.34 GB/s mmap figure the 4-7x rests on was not reproduced: this mmap emulation (multi-thread memmove) was 3-6x faster than 0.34 GB/s, so the true multiplier over stock llama.cpp depends on its single-fault-stream behaviour, which is unmeasured.
Verdict: confirmed as arithmetic and as a layout fact, with the same bounded multiplier as the screen (4x serial/7x overlap at 8 GB, fading at 16/32 GB, 1.0x vs our own explicit-read path). The measured rate gap between explicit reads and mmap was only about 1.5-2.4x on this machine.
Unmeasured: a real repacked file and loader; 125B file; Linux/Windows direct-I/O alignment.

Catalog rows: P-0092 idea; P-0093 SGLang Expert Pack (offline repack, direct I/O; layout claim supported by the 114 MB gate/up separation); P-0094 BigMoeOnEdge benchmarks (O_DIRECT 0.687 -> 2.191 tok/s, 3.2x; hardware, not reproduced); P-0095 Flash-MoE (pread streaming, 4.4 tok/s, 48 GB MacBook; not reproduced); P-0096 Jetson Orin Nano streaming fork (1.5x; not reproduced); P-0097 Apple LLM in a flash, HN comment on io_uring (under 2x; not reproduced). All mapped to fam_s1_14.

## s1 #15 No cache, one expert-aligned pread per routed expert into a slab

Needs: local (arithmetic done; replay optional).

| Measured / computed | value | screen |
|---|---|---|
| Previous-token reuse: db / story / code | 0.425 / 0.455 / 0.417 | same |
| LRU hit at 1/2/3 fp, db | 0.301 / 0.540 / 0.684 | 0.301/0.540/0.684 |
| No cache, 4 MiB reads: serial / overlap | 1.31 / 1.85 | 1.31 / 1.85 |
| 1 MiB reads | 0.95 / 1.20 | 0.95 / 1.20 |
| 0.9 MB reads | 0.82 / 1.00 | 0.82 / 1.0 |
| vs plain mmap at 8 GB | 2.50x serial, 3.11x overlap | 2.5x / 3.1x |
| vs our best path (explicit + LRU, hit 0.56) | 0.60x serial, 0.44x overlap | 0.60x / 0.44x |
| I/O-time speedup of previous-token slab reuse / LRU-2 over no cache | 1.76x / 2.22x | "kill if >1.7x" |

The screen's own kill rule fires: both the previous-token slab and LRU-2 cut I/O time by more than 1.7x, so the "no cache needed" part is false; keep at least the previous-token slab.
Replay run (shared machine, 3 decode tokens of the db trace after a 20-token warm-up, 8 threads, real tensor offsets): no cache, 3 reads per expert 0.60 s/token (1.8 GB/s); no cache, one read per expert 0.32 s/token (3.4 GB/s); previous-token reuse 0.15 s/token (1.77 GB read vs 3.29 GB); LRU-2 0.12 s/token (1.49 GB); mmap copy of the same extents 0.95 s/token (1.15 GB/s). So split reads were only 1.58x faster than the mmap replay (screen kill line: 1.5x), and one-read-per-expert 3.0x faster. Single short run, noisy.
Verdict: confirmed as arithmetic (2.5-3.1x over mmap, below the best path) and the cache-removal part refuted by the screen's own kill rule. Not a multiplier by itself.
Unmeasured: end-to-end tok/s; OS read-ahead; 125B offsets.

Catalog rows: P-0098 idea; P-0099 LLM in a flash (Apple) (larger contiguous reads, windowing; up to 4x on CPU; consistent in direction with the read-size rows, not reproduced); P-0100 Edge0 (prerouter prediction 20.4 vs 3.9 tok/s [corrected after the paper check: Edge0 reaches 19.9 tok/s on 24 GB without the prerouter (20.4 with); the +80% prerouter gain is on a 16 GB M2 with K=4 and a trained predictor/LoRA; it does not show naive streaming is ineffective]; not reproduced); P-0101 llama.cpp discussion 19883 (mmap thrashing; MoE at 100-150% RAM runs fine with mmap; not tested) [corrected after the paper check: the discussion is a 4.6 GB model that fits in VRAM, with a RAM-mirror question solved by --no-mmap; no MoE, no maintainer statement, no thrashing; the catalog description was wrong]. All mapped to fam_s1_15.

## Summary of verdicts
- #8 partly: bias gain on LRU up to 1.37x serial/1.08x overlap at 8 GB, quality unmeasured; static pin 0.76x of LRU.
- #9 partly: arithmetic confirmed; gain is read rate and overlap, 7.6x only if stock mmap is 0.34 GB/s (not reproduced here); needs model for prefill.
- #10 refuted vs LRU: pin 0.87 vs LRU 0.94 at 36%; 0.74x serial at 8 GB.
- #11 refuted for prefill: 1.02x overlap at uniform routing (1.33x only if 30B skew carried over).
- #12 needs hardware: hit 0.79 at 4x sparser geometry supports an I/O bound of about 8.7 tok/s at 24 GB; no GPU timing.
- #13 prefill refuted (1.0-1.1x); decode k=4 1.47-1.74x at 8 GB (screen said 1.3-1.4x), quality unmeasured.
- #14 confirmed as a bounded read-size effect; measured rate gap smaller than the screen's 7x.
- #15 confirmed as arithmetic; the no-cache part refuted by its own kill rule.


## Corrections from the deep paper check (papers/DEEP_SUMMARY.md)
- Edge0 reaches 19.9 tok/s on 24 GB without the prerouter (20.4 with); its +80% prerouter gain is on a 16 GB M2 with K=4 and a trained predictor/LoRA. It does not show naive streaming is ineffective.
- Hit-rate mapping: public benchmarks on the 125B-class model suggest hit 0.72-0.77 at 13-16% of experts, below the 0.95 our 30B footprint mapping gave at 16 GB.
