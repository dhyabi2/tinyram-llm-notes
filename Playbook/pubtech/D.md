# Published-technique family tests, group D (screen 2, ideas #2 to #9)

Files (public repo `tinyram-llm`): `tools/pubtech/f_D.py` (functions `fam_s2_2` ... `fam_s2_9`, dict `FAMILIES`), `tests/test_pubtech_D.py` (`run() -> T`, 141 checks, about 26 s; 146 checks and about 2 minutes when `PUBTECH_GGUF` points at a Q8_0 GGUF so the weight-tensor probes run). Run: `python3 tests/test_pubtech_D.py`. Print one family's numbers: `python3 tools/pubtech/f_D.py s2_4`.

Full paths: <tinyram-llm>/tools/pubtech/f_D.py, <tinyram-llm>/tests/test_pubtech_D.py.

## How to read this

- Traces: the three recorded Qwen3-30B-A3B decode traces (128 experts, top-8, 48 layers, 250-300 tokens each) stand in for the 125B (512 experts, top-10). Cache entries = footprints x 384 accesses; 8/16/32 GB machines hold about 2.1/7.6/20.8 footprints. 20.8 footprints exceeds the 30B model's whole 6144-expert set, so the hit rate there is capped by cold-start misses (0.962). Because of that cap, serial gains at 16/32 GB of 1.03-1.10x in the tables below are partly a cold-start artefact of 250-token traces; the overlap gains at 16/32 GB are exactly 1.00 because the 4.57 tok/s compute ceiling binds.
- Closed throughput model (same as the screens): serial = 1/(io + compute), overlap = 1/max(io, compute), compute 1/4.57 s per token, io = miss bytes / rate (2.38 GB/s where the screen used it: families 2, 5, 8; 3.1 GB/s: families 3, 4, 6, 7, 9). Baseline "best path" = explicit large reads + LRU: 8 GB serial 2.19 / overlap 4.19 (rate 2.38), 16 GB 4.07 / 4.57, 32 GB 4.18 / 4.57.
- Weight probes (families 5, 8, 9) read one tensor of the OLMoE Q8_0 GGUF (16 layers, 64 experts, top-8) with a pure-python reader. Scalar "Q2_K-like" / "Q4_like" quantisers are my stand-ins (not bit-exact K-quants; the Q4-like one uses exact scales so it is optimistic: 0.04 relative error vs 0.10 the screen expected for Q4_K). Nothing larger than OLMoE was read and no model was run by the tests.
- "Screen" = the earlier brainstorm screen's own analysis in idea_screen2_raw.json. Where the recomputation disagrees it is listed under "Disagreements".
- Not measured anywhere in this group: any model-level quality (perplexity/KL) of a lossy lever, any 125B tensor, any hidden-state predictor, any hardware other than this x86 Mac.

## Summary

| family | idea (claim) | verdict | key number |
|---|---|---|---|
| s2_2 | 8-dim VQ + LUT dot (about 8x fewer ops, 37 tok/s) | partly: bytes half confirmed, compute half refuted | 2-book LUT: 1.09x fewer ops on 768x2048, AVX2 gather kernel 0.06-0.17x of the int8 dot; bytes: 1.57x serial / 1.09x overlap at 8 GB, 1.0x overlap at 16/32 GB |
| s2_3 | persist prefix KV on disk (more than 3x TTFT) | confirmed as TTFT/UX feature, not a tok/s lever (already ships) | decode 1.0x; hit TTFT ratio 48-51x; request-time gain 1.24x for 200 tokens |
| s2_4 | cross-layer expert prefetch (2.3x) | refuted | 1.00-1.075x at 2.1 footprints, held-out precision 0.31-0.43 |
| s2_5 | shared low-rank basis + 2-3 bit residual (3.6x bytes) | refuted | no shared low-rank structure (16 of 64 components = 27.8% energy vs 25% chance); at most 1.53x serial / 1.09x overlap even with a free basis |
| s2_6 | run 16 of 48 layers (3.3x) | partly: arithmetic holds, quality unmeasured, redundancy proxy negative | 3.03x overlap at 67% skip = 1/kept; adjacent-layer Jaccard 0.031-0.033 = chance |
| s2_7 | pin prefill-streamed experts for decode (3x) | confirmed: no gain over LRU | pin minus LRU: -0.023 / +0.003 / +0.063 (20 tokens), -0.05..-0.09 over the rest |
| s2_8 | 1.3 bpw VQ (3.3x bytes) | refuted at 1.3 bpw | 256-word 1.0 bpw VQ error 0.58 vs 0.33 for 2-bit scalar; gain at most 2.0x serial / 1.09x overlap even if quality survived |
| s2_9 | 2-bit base + 2-bit residual for top-4 (2.4x bytes) | partly: bytes 1.4-1.5x reproduced, 2.4x never reached | 1.43-1.52x at 2.2 footprints, max 1.88x; 8 GB 1.17x serial / 1.00x overlap |

---

## Family s2_2 - 8-dim vector quantisation with lookup-table dot products

**Idea.** Quantise experts with 8-dim VQ (2 additive 256-word codebooks, about 2.4 bpw with scales); use the codes as table indices so each 8-weight block costs one gather+add instead of 8 FMAs. Claimed: bytes 4.3 to 2.4 bpw, about 8x fewer ops, compute ceiling 4.6 to 37 tok/s.

**Measured (function `fam_s2_2`).**

| quantity | value |
|---|---|
| op reduction, 768x2048 matrix, 1 / 2 codebooks (table build counted) | 2.18x / 1.09x |
| op reduction, 2048x768 (down), 1 / 2 codebooks | 4.0x / 2.0x |
| asymptote for many rows, 1 / 2 codebooks | 8x / 4x |
| rows needed for 2x fewer ops, 1 / 2 codebooks | 683 / 2048 |
| gate+up with one shared table over 20 consumers (10 experts x 2), 1 / 2 books | 7.06x / 3.53x |
| whole expert (gate, up, down) with own tables, 1 / 2 books | 2.57x / 1.29x |
| table size, 2048 columns, 1 book | 128 KiB (fits a 32 KiB L1 only up to 512 columns) |
| byte cut 4.3 to 2.4 bpw | 1.79x (claim: about 3x) |
| LUT GEMV equals dense dot of the same codes | max abs diff 9e-16 |

Throughput, 2.4 bpw vs 4.3 bpw best path (hit rate, serial / overlap tok/s, gain serial / overlap):

| RAM | baseline serial / overlap (hit) | 2.4 bpw, kernel 0.7x dot | kernel 1.0x | kernel 1.5x |
|---|---|---|---|---|
| 8 GB | 2.19 / 4.19 (0.562) | 2.60 / 3.20 = 1.19x / 0.76x | 3.44 / 4.57 = 1.57x / 1.09x (hit 0.763) | 4.59 / 6.86 = 2.10x / 1.64x |
| 16 GB | 4.07 / 4.57 (0.951) | 3.09 / 3.20 = 0.76x / 0.70x | 4.34 / 4.57 = 1.07x / 1.00x | 6.36 / 6.86 = 1.56x / 1.50x |
| 32 GB | 4.18 / 4.57 (0.962) | 3.09 / 3.20 = 0.74x / 0.70x | 4.34 / 4.57 = 1.04x / 1.00x | 6.36 / 6.86 = 1.52x / 1.50x |

**Hardware microbenchmark (one-off, not part of the tests because timing is not deterministic).** This Mac is x86 with AVX2 and AVX512; I wrote a C kernel pair (source kept outside the repo), built with `clang -O3 -mavx2 -mfma`, single thread, one token (GEMV), weights in cache. Baseline A = u8 x s8 `maddubs` int8 dot. B = per 8-wide activation chunk build the 256-entry table, then `vpgatherdd` lookups (8 rows per gather). B was verified against a scalar reference (0 mismatches). The machine was shared (load average about 20), so absolute times vary by +-20%, but every ratio below is 5x or larger.

| shape (rows x cols), books | A dot | B table build | B lookups only | B build+lookups | B(build+lookups) / A speed |
|---|---|---|---|---|---|
| 768 x 2048, 1 | 43.9 us (72 GFLOP/s-equiv) | 45.7 us | 309 us (10 GFLOP/s-equiv); chunk-major variant 265 us | 353 us | 0.12x (0.14x chunk-major) |
| 768 x 2048, 2 | 42.8 us | 54.5 us | 544 us | 656 us | 0.07x |
| 2048 x 768, 1 | 47.9 us | 13.3 us | 288 us | 287 us | 0.17x |
| 2048 x 768, 2 | 49.2 us | 27.0 us | 540 us | 795 us | 0.06x |
| 1536 x 4096, 1 (earlier run) | 226 us | 83.5 us | 1180 us | 1482 us | 0.15x |

Even with the table build free (shared codebook across experts), lookups alone are 0.09-0.19x of the dot. The screen's expected range was 0.5-1.5x and its kill line "B below 2x of A"; the kill line is met by a wide margin. Caveat: this is an AVX2 gather implementation of mine; T-MAC's bit-serial pshufb tables and an AVX512 permute design were not tried, and the target laptop's gather throughput may differ.

**Verdict.** Partly. The bytes half is real but small: 1.57x serial, 1.09x overlap at 8 GB, none at 16/32 GB (compute-bound). The compute half (8x fewer ops, 37 tok/s) is refuted: the op count with a table build is 1.09-2.2x on the actual expert shapes, and the measured AVX2 gather kernel is slower than the int8 dot, which turns the 2.4 bpw byte gain into a loss whenever the kernel is below 1.0x of the dot (rows kernel 0.7x above).

**Published vs measured.** T-MAC/AQLM report 2.7-6.6x, but against dequantise or FP32 baselines at 1-2 bit; against our int8 dot baseline none of that appears.

**Disagreements with the screen.** Screen said 1.5x serial / 1.1x overlap: reproduced (1.57 / 1.09). Screen's 16/32 GB "1.0x from bytes": our serial gain is 1.04-1.07x (cold-start artefact, see above), overlap 1.00x as stated.

**Unmeasured.** Multi-thread behaviour, a pshufb/permute LUT, AVX2 laptop-class gather throughput, 2-bit VQ model quality, the 125B (no fp checkpoint).

**Catalog method rows (all 7, mapped to `fam_s2_2`).**

| id | catalog method (full name) | claim vs what the family test shows |
|---|---|---|
| P-0140 | T-MAC (Microsoft, EuroSys 2025), LUT mpGEMM on CPU | Claims a 3.6x average kernel speedup at 1-2 bit, but against llama.cpp dequantise kernels, not an int8 dot. Our LUT-vs-int8-dot microbenchmark on AVX2 (gather, 8-dim codebooks) gave 0.06-0.17x, so the published gain does not transfer to this baseline. T-MAC itself uses bit-serial pshufb tables, which we did NOT test. |
| P-0141 | T-MAC Microsoft Research blog (Snapdragon X Elite) | ARM (tbl) numbers on a Snapdragon; not transferable to the x86 AVX2 gather we measured. Unmeasured here. |
| P-0142 | AQLM with CPU Numba LUT kernels | Speedup is against FP32 (up to about 4x). Against an int8 dot kernel our gather LUT gives 0.06-0.17x; the AQLM Numba kernel itself was not run. |
| P-0143 | AQLM HF docs kernel table | Same source as above (CPU Numba kernel, about 4x vs FP32; not in llama.cpp). Covered by the same microbenchmark; unmeasured as a product. |
| P-0144 | llama.cpp IQ2_XS/IQ2_XXS codebook-grid quants (2.06-2.5 bpw) | Codebook-grid 2-bit quants already ship (grid lookup then int8 dot). Their byte lever equals our 2.0-2.6 bpw rows: 1.5-1.75x serial and 1.09x overlap at 8 GB, 1.0x overlap at 16/32 GB. |
| P-0145 | KBVQ-MoE (ICLR 2026), VQ for MoE LLMs | Model-level quality at 3 bit is not reproducible here. Our weight probe only shows that a plain 8-dim 256-word codebook at 1 bpw is far worse than 2-bit scalar (family 8). |
| P-0146 | Shared-codebook VQ with single table lookup for MoE experts on tensor accelerators | Accelerator-oriented, no CPU kernel; not testable here. The only related number we have is the shared-table op count (7.1x fewer ops for gate/up with one shared codebook and 20 consumers). |

---

## Family s2_3 - persist the KV cache of a finished prompt

**Idea.** Save the exact KV of a finished prompt to disk keyed by a hash of prefix, model, quant and router; on a repeat or shared prefix, load it instead of prefilling. Claimed more than 3x TTFT on hits; decode unchanged.

**Measured (function `fam_s2_3`, arithmetic + traces; plus a one-off model-plumbing run on OLMoE-Q2_K).**

| quantity | value |
|---|---|
| KV per token, 30B shape (48 layers, 4 KV heads, dim 128, fp16) | 98,304 B; 49.2 MB per 500 tokens (shape arithmetic, not read from the 30B file) |
| KV per token, small MoE (16 layers, 16 KV heads, dim 128) | 131,072 B (from GGUF metadata) |
| family's "1-2 GB for 500 tokens" over the estimate (30B shape x3 for the 125B) | 6.8x - 20x too large |
| prefill compute, 500 tokens x 12 GFLOP | 109.1 s at the 55 GFLOP/s decode rate; 19.3 s at the 311 GFLOP/s batched rate |
| expert stream I/O for a cold prefill (69 GB) | 22.3 s at 3.1 GB/s; 29.0 s at 2.38 GB/s |
| cold TTFT, best path (compute and I/O overlapped / serial) | 22.3 s / 41.6 s |
| hit TTFT (KV load + one decode step), KV 1x / 2x / 3x the 30B shape | 0.434 / 0.450 / 0.466 s |
| TTFT ratio on an exact hit | 51.3x / 49.5x / 47.8x |
| whole 200-token request, hit vs miss, at 2.18 tok/s | 1.24x |
| decode tokens/s change | exactly 1.0 |

Cross-prompt hot-list reuse (a static set learned on the other two prompts, hit rate on the third; in-sample in brackets):

| trace | 1 fp | 2.1 fp | 4.6 fp | 7.5 fp |
|---|---|---|---|---|
| db | 0.175 (0.380) | 0.371 (0.605) | 0.690 (0.889) | 0.869 (0.987) |
| story | 0.099 (0.434) | 0.248 (0.657) | 0.510 (0.915) | 0.677 (0.993) |
| code | 0.148 (0.359) | 0.293 (0.554) | 0.577 (0.833) | 0.777 (0.964) |

Model plumbing (one-off, OLMoE Q2_K, 4 threads, shared machine; KV size does not depend on the quant): `llama-completion --prompt-cache` with a 419-token prompt. Cold run prompt eval 6.0 s (69 tok/s; 8.6 s in an earlier loaded run); warm run evaluates 1 token (0.0 ms). Generated 12 greedy tokens byte-identical (md5 equal). Cache file 54,926,297 B = 131,072 B/token x 419 + header. The test suite does not repeat this (timings are not deterministic).

**Verdict.** Confirmed as a TTFT / chat-UX feature and refuted as a throughput lever: decode is untouched and the feature already ships in stock llama.cpp. The hit-case TTFT ratio is about 50x (more than the claimed 3x), the whole-request gain is 1.24x because 200 decoded tokens dominate.

**Published vs measured.** hearth-llm's 99 s vs 1.8 s (30k tokens) matches the shape of our 6.0 s to about 0 on a small prompt; llama.cpp's own docs/commit are what we ran.

**Disagreements with the screen.** (1) Request-time gain: screen 1.3-1.6x, ours 1.24x (the screen's range assumed a 30 s prefill and 4 tok/s decode; at our best-path 2.18 tok/s decode it is lower). (2) Screen text "hot lists hit only 5-21% across prompts": we get 0.10-0.18 at 1 footprint but 0.25-0.37 at the 8 GB cache size of 2.1 footprints and 0.51-0.69 at 4.6 footprints, so the statement only holds for small lists. (3) Cold TTFT: our 22.3 s lies below the screen's 25-35 s (we take the max of compute and I/O).

**Unmeasured.** Real 125B KV shape, real chat hit rates, the slot-API server path, correctness across a changed build (no output-identity check on the patched build yet).

**Catalog method rows (all 7, mapped to `fam_s2_3`).**

| id | catalog method (full name) | claim vs what the family test shows |
|---|---|---|
| P-0148 | llama.cpp discussion #18244, restoring KV for a 950k-token prompt | Stock llama.cpp already has --prompt-cache and slot save/restore. We exercised --prompt-cache on a small MoE: 419-token prompt, cold prompt eval 6.0 s, warm run evaluates 1 token (0 ms), generated text byte-identical, cache file 54.9 MB = 131072 B/token. |
| P-0149 | hearth-llm (PyPI), a caching proxy for llama-server | Same feature as a proxy. Self-reported 99 s vs 1.8 s on 30k tokens; our small-MoE plumbing test shows the same shape (6.0 s to about 0). Not re-run on their proxy. |
| P-0150 | DiskLLM (PyPI), session persistence over the slot API | Session persistence over the slot API; same mechanism. Not run (server slot API untested). |
| P-0151 | llama.cpp commit: save and restore KV cache for a single seq id (#6341) | The 2024 per-sequence save/restore is the mechanism behind --prompt-cache; our run confirms it works with the project's llama.cpp build. |
| P-0152 | Tutti: SSD-backed KV cache for long-context serving, arXiv 2605.03375 | Server-setting TTFT numbers (78% cut); not comparable. Our hit-case arithmetic (TTFT ratio 48-51x) is for the 125B target with a 22 s cold expert stream. |
| P-0153 | py-kvcache: external KV caching for vLLM on NVMe, arXiv 2609.11744 | Load-vs-recompute: for the target, loading takes 0.016-0.05 s against 19-22 s of prefill, so loading wins by a wide margin (arithmetic only). |
| P-0154 | SGLang-LSM, arXiv 2511.16138 | Quotes about 120 KB of KV per token for Llama3-8B; our shapes give 98 KB/token (30B shape) and 131 KB/token (small MoE, confirmed by the cache file size). The 143% hit gain is a server workload, unmeasured. |

---

## Family s2_4 - cross-layer expert prefetch

**Idea.** Predict layer l+1's experts from layer l's chosen set (claimed precision 0.55), prefetch about 0.72 GB while computing, read only the misses. Claimed 2.3x, lossless.

**Measured (function `fam_s2_4`).** Predictor = layer-to-layer co-occurrence counts, prefetch top-m by summed counts. Held-out = trained on the other two traces. Event simulation: per-layer LRU, one FIFO disk at 3.1 GB/s, 3.375 MB per slice, compute 4.57 tok/s split over 48 layers, prefetch for layer l+1 queued after layer l's demand reads.

| trace | precision top-8 held-out (in-sample) | precision top-4 held-out | previous-token persistence |
|---|---|---|---|
| db | 0.430 (0.588) | 0.518 | 0.425 |
| story | 0.307 (0.593) | 0.367 | 0.455 |
| code | 0.370 (0.570) | 0.453 | 0.417 |

Random predictor precision would be 8/128 = 0.0625.

| footprints | trace | no prefetch tok/s (GB/token) | top-4 speedup (GB/token) | top-8 speedup (GB/token) | oracle top-4 / top-8 |
|---|---|---|---|---|---|
| 2.1 | db | 2.457 (0.583) | 1.075 (0.726) | 1.045 (0.938) | 1.250 / 1.590 |
| 2.1 | story | 2.566 (0.530) | 1.038 (0.760) | 0.939 (1.064) | 1.235 / 1.563 |
| 2.1 | code | 2.414 (0.606) | 1.058 (0.753) | 1.009 (0.987) | 1.249 / 1.620 |
| 4.0 | db / story / code | 3.27 / 3.45 / 3.05 | 1.016 / 1.000 / 1.026 | 1.023 / 0.985 / 1.027 | - |
| 7.5 | db / story / code | 4.18 / 4.18 / 3.96 | 1.005 / 1.006 / 1.005 | 1.004 / 1.005 / 1.003 | - |

One-layer lead = 4.56 ms against 8.71 ms to read one layer's experts.

**Verdict.** Refuted. 1.00-1.075x at 8 GB, 1.00x at 16/32 GB; the kill criterion (below 1.15x at 2.1 footprints with a held-out predictor) is met on every trace. Even a perfect predictor with only a one-layer lead gets 1.25x (top-4) to 1.6x (top-8), well under the 1.9x perfect-overlap bound, because the disk is the shared bottleneck.

**Published vs measured.** ProMoE's 1.66x on llama.cpp and HOBBIT's 9.93x rely on stronger predictors / lossy substitution [corrected after the paper check: against an LRU baseline ProMoE is 1.31x (transformers) and 1.09x (llama.cpp) and HOBBIT's prefetch alone is 1.01-1.05x; the larger headline ratios are byte cuts against weak or thrashing baselines, not stronger predictors]; our lossless id-based predictor reproduces the "net 0%" kind of result (budgeting-bytes style).

**Disagreements with the screen.** Held-out top-8 precision 0.43 / 0.31 / 0.37 and previous-token persistence 0.42-0.46 reproduce exactly; speedups reproduce within 0.02 (screen top-4 1.07/1.04/1.06, top-8 1.02/0.92/0.98; ours 1.075/1.038/1.058 and 1.045/0.939/1.009). The screen's "top-4 precision 0.18-0.26" is hits divided by 8; precision of the four prefetched experts is 0.37-0.52 (definition only, not a speed difference).

**Unmeasured.** Pre-attention / hidden-state predictors (need a real model's router weights), 512-expert precision, Apple-silicon prefetch behaviour.

**Catalog method rows (all 4, mapped to `fam_s2_4`).**

| id | catalog method (full name) | claim vs what the family test shows |
|---|---|---|
| P-0156 | ProMoE: proactive caching with learned per-layer predictors | Reports 2.07x on transformers and 1.66x on llama.cpp with learned predictors on richer inputs [corrected after the paper check: vs an LRU baseline it is 1.31x and 1.09x; the headline ratios come from weak baselines, not stronger predictors]. Our expert-id co-occurrence predictor gets 0.31-0.43 held-out precision and 1.00-1.075x at 2.1 footprints; the learned-predictor result is not reproduced. |
| P-0157 | HOBBIT: mixed-precision expert offloading, built on llama.cpp | Up to 9.93x includes lossy low-precision substitution for misses [corrected after the paper check: HOBBIT's prefetch alone is 1.01-1.05x, and the headline is against a thrashing baseline]; we measured only the lossless prefetch part (1.00-1.075x). Substitution is a quality-trading lever, not tested. |
| P-0158 | Budgeting Bytes: storage roofline for storage-bound LLM decoding | Reports 18.8% previous-token prefetch hit on a top-2-of-8 model; our previous-token persistence on top-8-of-128 is 0.42-0.46 (different model). Direction agrees: cheap predictors are weak and storage bandwidth binds. |
| P-0159 | Pre-attention expert prediction (arXiv 2511.10676) and Speculating Experts (arXiv 2603.19289) | 93-98% accuracy comes from hidden-state routers, which need a real model; not testable on id traces. Our oracle rows (1.25x top-4, 1.59x top-8 at 2.1 footprints with a one-layer lead) bound what such a predictor could reach. |

---

## Family s2_5 - shared low-rank basis plus low-bit residual

**Idea.** Per MoE layer factor all experts onto one resident low-rank basis, store each expert as a 2-3 bit residual on disk. Claimed 0.36 GB per token instead of 1.3 GB (3.6x), 1.85 to 6.6 tok/s.

**Measured (function `fam_s2_5`).** Closed form on the traces plus a weight probe on OLMoE.

| quantity | value |
|---|---|
| residual bpw needed for 0.36 GB/token | 1.19 bpw (a 2-3 bit residual cannot give it) |
| bytes ratio for 2.0 / 2.5 / 3.0 bpw | 0.465 / 0.581 / 0.698 |
| claimed 6.6 tok/s vs compute ceiling | 6.6 > 4.57, impossible |
| resident basis, rank 4 / 16 / 64, 4.3 bpw, 48 layers | 0.52 / 2.07 / 8.3 GB |

8 GB machine (gain vs 4.3 bpw best path; free basis, rank 4 resident, rank 16 resident, free basis with 25% extra FLOPs):

| residual bpw | serial gain, free | rank 4 | rank 16 | overlap gain, free | overlap gain, +25% FLOPs |
|---|---|---|---|---|---|
| 2.0 | 1.75x | 1.62x | 1.26x | 1.09x | 0.87x |
| 2.5 | 1.53x | 1.42x | 0.85x | 1.09x | 0.87x |
| 3.0 | 1.35x | 1.24x | 0.76x | 1.09x | 0.87x |

16 GB: serial gain 1.05-1.08x, overlap 1.00x; 32 GB: serial 1.03-1.05x, overlap 1.00x; with 25% extra FLOPs overlap falls to 0.80x at both.

Weight probe (OLMoE-Q8_0, `ffn_up_exps` layer 8, 64 experts, a 32-row x 1024-column sub-block of each, expert-axis PCA via a pure-python Jacobi eigen-solver):

| quantity | value |
|---|---|
| share of total energy in the mean expert | 1.6% |
| top-1 ... top-16 eigenvalue of the centred Gram | 9.07 ... 8.37 (flat, ratio 0.92) |
| centred energy captured by 4 / 16 shared components | 7.2% / 27.8% (chance: 6.3% / 25.4%) |
| relative error of the low-rank part alone, rank 16 | 0.843 |
| relative error with a Q2_K-like quantised residual: rank 0 / 4 / 16 | 0.326 / 0.314 / 0.277 |
| direct Q2_K-like / Q4-like error | 0.327 / 0.043 |

**Verdict.** Refuted. Experts of this model are essentially independent in the expert axis (flat spectrum at chance level), so a shared basis does not shrink the residual; the byte lever tops out at about 1.5x serial and 1.09x overlap at 8 GB even if the basis were free and the quality were fine, and a resident basis of rank 16 (2.07 GB) would consume the whole 8 GB expert cache. The 3.6x needs 1.19 bpw.

**Published vs measured.** MoBE/D2-MoE/MoE-SVD get 24-60% parameter cuts with fitted merging, pruning and fine-tuning, in the 1.3-2.5x range, not 3.6x.

**Disagreements with the screen.** Its serial 1.3x / overlap 1.1x was for a 1.7x byte cut; with the trace-measured cache effect a 2.5 bpw residual gives 1.53x serial (higher) and the same 1.09x overlap. Its expectation "residual keeps over 90% of the energy, not low-rank" is supported (72% of centred energy sits outside 16 components).

**Unmeasured.** Model quality of any shared-basis scheme; 512-expert spectra (probe has 64 experts of a small model); other layers/tensors (only layer 8 up-projection); training-free fitting at 125B scale.

**Catalog method rows (all 5, mapped to `fam_s2_5`).**

| id | catalog method (full name) | claim vs what the family test shows |
|---|---|---|
| P-0161 | D2-MoE, Delta Decompression for MoE LLMs (ICML 2025) | 40-60% compression with measurable loss. Our probe on a small MoE finds no shared low-rank structure across experts (16 of 64 shared components hold 27.8% of the centred energy, chance 25%), consistent with needing fitted merging/pruning rather than plain SVD. |
| P-0162 | MoBE, Mixture-of-Basis-Experts | Parameter cut 24-30% (1.3-1.4x) is far below the 3.6x of the family claim; our byte arithmetic gives at most 1.7x for a 2.5 bit residual even with a free basis. |
| P-0163 | MoE-SVD (ICML 2025) | Shared V with trimmed U gives 60% compression; unmeasured beyond the same expert-axis spectrum probe (an input-side shared V was not separately tested). |
| P-0164 | DeltaMoE, memory-efficient inference for merged MoE with delta compression (ICLR 2025 workshop) | 2.34x compression and 2.57x throughput with structured sparsity and quantisation, on GPU/merged models; unmeasured here. Our closed-form byte lever at 2.0 bpw gives 1.75x serial / 1.09x overlap at 8 GB. |
| P-0165 | Bandwidth-efficient adaptive MoE via low-rank compensation | Same shape as our closed-form lever: streaming a low-bit base is a byte cut and gains vanish when decode is compute-bound (overlap 1.0x at 16/32 GB). Compensator quality unmeasured. |

---

## Family s2_6 - run only about 16 of 48 layers per token

**Idea.** A distilled per-layer router picks about 16 layers per token; all 10 experts stay active in the layers that run. Claimed 3.3x (12 to 4 GFLOP/token).

**Measured (function `fam_s2_6`).**

Step 1 of the screen (free): adjacent-layer expert-set Jaccard averaged over tokens (random 8-of-128 sets: 0.034).

| trace | mean Jaccard | max pair | pairs above 0.3 / above 0.1 (of 47) |
|---|---|---|---|
| db | 0.033 | 0.065 | 0 / 0 |
| story | 0.032 | 0.055 | 0 / 0 |
| code | 0.031 | 0.055 | 0 / 0 |

Throughput vs no skipping (mean over traces; static = same layers every token; dynamic = random layers per token; cache size unchanged), gain serial / overlap:

| RAM | skip | 1/kept | static | dynamic |
|---|---|---|---|---|
| 8 GB | 25% | 1.33 | 1.47 / 1.33 | 1.31 / 1.33 |
| 8 GB | 50% | 2.00 | 2.68 / 2.00 | 1.93 / 2.00 |
| 8 GB | 67% | 3.00 | 4.86 / 3.03 | 2.87 / 3.02 |
| 16 GB | 25% / 50% / 67% | 1.33 / 2.00 / 3.00 | 1.36 / 2.04 / 3.09 serial; 1.33 / 2.00 / 3.03 overlap | 1.31 / 1.91 / 2.77 serial; overlap as static |
| 32 GB | 25% / 50% / 67% | 1.33 / 2.00 / 3.00 | 1.33 / 2.00 / 3.03 serial; overlap same | 1.31 / 1.89 / 2.74 serial; overlap 1.33 / 2.00 / 3.03 |

Analytic model (combo_model 'exit' lever, likely scenario) at 75% / 50% / 33% kept: 8 GB 1.48 / 2.66 / 4.65, 16 GB 1.36 / 2.04 / 3.09, 32 GB 1.33 / 2.00 / 3.03.

**Verdict.** Partly. The arithmetic is confirmed as plain 1/kept once compute binds (3.03x at 67% skip, the claimed 3.3x slightly overstates it); the 8 GB figure of 4.65x is overstated, the honest value is the dynamic per-token one (2.9x) because skipping fixed layers also shrinks the cache working set. The router-trace redundancy proxy gives no support to skipping: adjacent-layer expert sets overlap exactly at chance (but expert ids are not comparable across layers, so this is a weak proxy). Quality at any skip fraction is unmeasured and is the decisive test (needs a model run with a layer mask).

**Published vs measured.** ShortGPT/Dr.LLM remove or skip only 10-25% of layers without retraining [corrected after the paper check: Dr.LLM skips only about 1% of layer applications, so it is no evidence for 25-67% skip; only ShortGPT matches the 10-25% claim]; at that range our arithmetic gives 1.11-1.33x (overlap), matching the screen's 1.1-1.4x.

**Disagreements with the screen.** None in numbers: combo_model reproduces 4.65 / 3.09 / 3.03; trace-driven overlap gain at 16/32 GB is 1/kept within 1%.

**Unmeasured.** Perplexity / task quality under 25-67% layer skipping, router distillation, KV for skipped layers, per-token dynamic routing interaction with prefetch.

**Catalog method rows (all 4, mapped to `fam_s2_6`).**

| id | catalog method (full name) | claim vs what the family test shows |
|---|---|---|
| P-0167 | Dr.LLM: Dynamic Layer Routing in LLMs (Heakl et al., ICLR 2026) | Skips about 5 layers (10-15%) per example with trained routers [corrected after the paper check: about 1% of layer applications skipped, so no evidence for 25-67% skip]; our arithmetic at 25% skip is 1.33x (overlap) and at 67% skip 3.0x. Their accuracy numbers are not reproduced. |
| P-0168 | LayerSkip (Meta, ACL 2024) | Training-time layer dropout plus early-exit loss with self-speculative decoding; not testable on traces. Its 1.8-2.2x comes from draft-and-verify, a different mechanism. |
| P-0169 | ShortGPT (Men et al. 2024) | About 25% of layers removable at a cost (MMLU 55.0 to 52.2 on a dense 13B). The router traces cannot measure block influence; quality unmeasured. |
| P-0170 | Early-exit analysis for MoE models (secondary overview article that came up in the search) | Secondary source: Qwen3-MoE defers decisions to deep layers. Our adjacent-layer expert-set Jaccard is exactly at chance (0.031-0.033 vs 0.034), so the traces give no redundancy signal either way. |

---

## Family s2_7 - session-pinned RAM set from prefill

**Idea.** Keep experts streamed during expert-major prefill in a pinned RAM set (about 20 GB), so decode reads RAM at the 4.6 tok/s ceiling. Claimed about 3x.

**Measured (function `fam_s2_7`).** Static per-layer top-N by frequency over the first W tokens vs a per-layer LRU of N entries warmed on the same W tokens; hit rate over the next 20 tokens. N = 19 / 37 / 64 of 128 (15% / 29% / 50%).

| trace | W | 15% pin / LRU | 29% pin / LRU (diff) | 50% pin / LRU |
|---|---|---|---|---|
| db | 64 | 0.577 / 0.579 | 0.814 / 0.837 (-0.023) | 0.938 / 0.962 |
| db | 128 | 0.642 / 0.670 | 0.863 / 0.877 (-0.014) | 0.957 / 0.976 |
| story | 64 | 0.638 / 0.617 | 0.867 / 0.864 (+0.003) | 0.976 / 0.979 |
| story | 128 | 0.633 / 0.697 | 0.845 / 0.926 (-0.081) | 0.968 / 0.982 |
| code | 64 | 0.567 / 0.565 | 0.821 / 0.758 (+0.063) | 0.961 / 0.952 |
| code | 128 | 0.592 / 0.594 | 0.835 / 0.826 (+0.009) | 0.960 / 0.969 |

Longer horizon and other-prompt pin (29%, W=64): rest of the trace pin minus LRU: db -0.067, story -0.050, code -0.091; set learned on the other two prompts: db 0.638 vs LRU 0.837, story 0.574 vs 0.864, code 0.608 vs 0.758 (differences -0.20 / -0.29 / -0.15).

LRU hit curve (mean of 3 traces): 1.0 fp 0.306, 2.1 fp 0.567, 4.0 fp 0.793, 7.5 fp 0.950, 20.8 fp 0.962. Best path at 3.1 GB/s: 8 GB 2.49 serial, 16 GB 4.18, 32 GB 4.26, overlap 4.57 (ceiling) at 16/32 GB. The claimed 3x against a 2-4 tok/s baseline is really 1.14-2.3x (4.57 / 4 or 4.57 / 2).

**Verdict.** Confirmed that pinning is not a separate multiplier. Hit rates of 0.81-0.87 at 29% are real, but a warmed LRU of the same size gets the same on average (mean pin minus LRU +0.014 over three traces), and over a longer horizon or for a hot set from other prompts LRU wins clearly.

**Disagreements with the screen.** The screen's kill criterion was "pin minus LRU below +0.05 on all 3 traces"; code reaches +0.063 on the 20-token window (db -0.023, story +0.003), so the criterion is met on two of three, and the verdict rests on the mean (+0.014) and on the longer horizon where LRU wins by 0.05-0.09. The pin hit rates 0.81 / 0.87 / 0.82 reproduce within 0.01.

**Published vs measured.** MoE-Infinity's 3.1-16.7x is against much weaker systems; against explicit reads plus LRU our free analogue shows no gain.

**Unmeasured.** Prefill routing (only decode traces exist), the 125B working-set size, the effect of expert-major prefill order on LRU state, SeqMoE's own variant.

**Catalog method rows (the single row, mapped to `fam_s2_7`).**

| id | catalog method (full name) | claim vs what the family test shows |
|---|---|---|
| P-0172 | MoE-Infinity, arXiv 2401.14361 | Activation-count guided cache replacement. Our static frequency pin (the closest free analogue) matches a warmed LRU within -0.02..+0.06 on a 20-token window and loses by 0.05-0.09 over the rest of the trace, so frequency pinning alone does not beat LRU here. MoE-Infinity's 3.1-16.7x is against vLLM/Ollama/DeepSpeed, not against explicit reads plus LRU. |

---

## Family s2_8 - 8-dim VQ at 1.3 bpw (256-entry codebook)

**Idea.** One 8-bit index per 8 weights into a 256-entry codebook, claimed 1.3 bpw, 3.3x fewer bytes per token.

**Measured (function `fam_s2_8`).**

| quantity | value |
|---|---|
| bpw of an 8-bit index per 8 weights | 1.0 (codebook adds 0.021 bpw on a 1.57M-weight matrix); the claimed 1.3 is not derived |
| a 2-codebook layout / a 65,536-entry single codebook | 2.0 / 2.0 bpw |
| Gaussian reference, normalised MSE at 1 bpw: Shannon bound / 1-bit Lloyd-Max | 0.25 / 0.363 |

Throughput gain vs 4.3 bpw best path:

| RAM | 1.0 bpw | 1.3 bpw | 2.0 bpw | 2.4 bpw | 2.625 bpw |
|---|---|---|---|---|---|
| 8 GB serial / overlap | 2.04x / 1.09x | 2.00x / 1.09x | 1.75x / 1.09x | 1.57x / 1.09x | 1.48x / 1.09x |
| 16 GB serial / overlap | 1.10x / 1.00x | 1.09x / 1.00x | 1.08x / 1.00x | 1.07x / 1.00x | 1.06x / 1.00x |
| 32 GB serial / overlap | 1.07x / 1.00x | 1.06x / 1.00x | 1.05x / 1.00x | 1.04x / 1.00x | 1.04x / 1.00x |

Weight probe (OLMoE-Q8_0 `ffn_down_exps` layer 8; k-means trained on 8192 blocks of experts 0-1, error on 8192 held-out blocks of experts 2-3, 64 rows x 512 columns each; 8 Lloyd iterations; random Gaussian x error equals the weight (Frobenius) error in expectation: 0.3197 measured over 24 x vs 0.3258):

| quantiser | bpw | relative error |
|---|---|---|
| 8-dim, 256 words (1 codebook) | 1.0 | 0.581 |
| two additive 256-word codebooks | 2.0 | 0.339 |
| Q2_K-like scalar (stand-in) | 2.625 | 0.326 |
| Q4-like scalar (stand-in, exact scales) | about 5.0 | 0.040 |

**Verdict.** Refuted at 1.3 bpw: the screen's kill criterion (256-entry VQ error above the Q2_K error) is met by a wide margin (0.58 vs 0.33). Even granting the quality, the gain is capped at 2.0x serial / 1.09x overlap at 8 GB and about 1.0x overlap above. The 1.0 bpw VQ error 0.58 is near the screen's expected 0.45-0.6.

**Published vs measured.** KBVQ-MoE, VPTQ, GPTVQ get near-FP16 only at about 3 bit, with shared codebooks, Hessian weighting and fine-tuning; our plain k-means shows ~2 bpw of plain VQ is about a 2.6 bpw scalar quantiser.

**Disagreements with the screen.** Q2_K error expected 0.25, our stand-in 0.33; Q4_K expected 0.10, our stand-in 0.04 (stand-ins are not the real formats, so only the VQ-vs-stand-in comparison is meaningful). 16 GB serial gain 1.09x vs the screen's 1.05x (cold-start artefact).

**Unmeasured.** Perplexity / KL of any VQ at model level, real K-quant bit-exact errors, 65,536-entry codebooks, fine-tuning recovery, other tensors/layers (one tensor of one layer).

**Catalog method rows (all 7, mapped to `fam_s2_8`).**

| id | catalog method (full name) | claim vs what the family test shows |
|---|---|---|
| P-0174 | KBVQ-MoE, vector quantisation for MoE (ICLR 2026) | Near-FP16 at 3 bit with shared components and bias correction. Our probe: a plain 8-dim, 256-word VQ at 1.0 bpw has 0.58 relative error vs 0.33 for a Q2_K-like scalar quantiser; two additive 256-word books (2.0 bpw) reach 0.34. So about 2 bpw is where plain VQ matches 2.6 bpw scalar here; 1.3 bpw is not reachable with this codebook. |
| P-0175 | QMoE sub-1-bit compression (Frantar and Alistarh) | 0.8 bpw needs dictionary coding of ternary weights on a very redundant 1.6T model with GPU kernels; not comparable, not tested. |
| P-0176 | MoQE, 2-bit expert-only quantisation (Microsoft) | 2-bit experts only, 79.6% size cut. Consistent with our closed-form (2.0 bpw: 1.75x serial / 1.09x overlap at 8 GB); the quality side was not measured. |
| P-0177 | mixtral-offloading (Eliseev and Mazur): 2-bit HQQ experts plus LRU | Same lever (2-bit experts plus LRU) already tried. Our byte lever reproduces its shape; GPU numbers are not comparable. |
| P-0178 | VPTQ (Microsoft), PyPI page | VQ for LLMs; a llama.cpp port is only a roadmap item. Our k-means VQ is the simplest VQ (no Hessian weighting, no fine-tuning), so its error is an upper bound on what VPTQ reaches at the same bits. |
| P-0179 | GPTVQ | Post-training VQ beating scalar at 2-3 bits on dense Llama/Mistral. Not contradicted: our 2.0 bpw VQ roughly equals a 2.6 bpw scalar stand-in, i.e. VQ saves about 0.6 bpw at equal error even without GPTVQ's refinements. |
| P-0180 | Mixture Compressor for MoE LLMs | Static expert quantisation plus dynamic pruning below 3 bits; same byte lever, quality unmeasured. |

---

## Family s2_9 - 2-bit base plus 2-bit residual enhancement for the top-4 experts

**Idea.** Store every expert as a 2-bit base plus a 2-bit residual; keep bases cached/resident; the router's top-4 experts per layer also load the enhancement, the rest run at base precision. Claimed 2.4x fewer disk bytes, 5-9 tok/s.

**Measured (function `fam_s2_9`).** Replay: Q4 LRU of C = k x 384 entries (bytes per access = miss rate) against an LRU of 2C half-size entries (base key for every chosen expert, enhancement key for the first four ids of each layer record, assuming recorded order = gate rank), at equal cache bytes. Ratio = Q4 bytes / scheme bytes.

| footprints k | db | story | code |
|---|---|---|---|
| 2.2 (screen's) | 1.497 | 1.516 | 1.432 |
| 4.4 | 1.881 | 1.719 | 1.809 |
| 8.0 | 1.262 | 1.161 | 1.458 |
| 2.1 (8 GB) | 1.479 | 1.489 | 1.436 |
| 7.6 (16 GB) | 1.339 | 1.225 | 1.562 |
| 20.8 (32 GB) | 1.158 | 1.118 | 1.143 |

Controls at 2.2 footprints: enhancing all 8 experts gives exactly 1.000 on all traces; enhancing the last four (lowest-ranked) gives 1.365 / 1.398 / 1.310 (rank order matters); no enhancement at all (pure 2-bit base) gives 5.20 / 5.89 / 4.06 (the lossy extreme).

Throughput at the real footprint counts (mean of traces, 3.1 GB/s): 8 GB bytes ratio 1.47, serial 2.50 to 2.92 tok/s (1.17x), overlap 1.00x (ceiling 4.57); 16 GB ratio 1.40, serial 1.03x, overlap 1.00x; 32 GB ratio 1.14, serial 1.01x, overlap 1.00x.

Rank-order check (indirect): mean hotness of the expert at recorded position 0..7 falls monotonically from 77.9 to 55.5 and previous-token persistence from 0.497 to 0.312, consistent with (not proof of) gate-rank order.

Weight probe (OLMoE-Q8_0 `ffn_down_exps` layer 8, experts 2-3, 64 rows x 512): Q2_K-like base alone relative error 0.326 (2.625 bpw); base + Q2_K-like residual 0.085 (5.25 bpw); plain Q4-like scalar 0.040 (about 5.0 bpw).

**Verdict.** Partly. The bytes replay reproduces (1.43-1.52x at 2.2 footprints, up to 1.88x) but the claimed 2.4x is never reached and at the real 8 GB footprint it is 1.47x bytes, 1.17x serial and 1.00x overlap; above 8 GB it is compute-bound. The quality side is unfavourable for a post-hoc split (nested base+residual costs more bits than a 4-bit scalar for a higher error) and base-only experts run at the 2-bit error (0.33 relative); the real quality cost needs trained nesting and a perplexity run.

**Published vs measured.** D2MoE / Matryoshka / DynaExq are the trained versions of this layout; their quality is not reproduced.

**Disagreements with the screen.** Replay numbers reproduce within 0.01 (db 1.50, story 1.52, code 1.43 at 2.2; 1.88 / 1.72 / 1.81 at 4.4; 1.26 / 1.16 / 1.46 at 8). Screen's k=8 range 1.16-1.46 matches; screen's kill line "below 1.5x at the 8 GB footprint" is met (1.44-1.49).

**Unmeasured.** Perplexity of base-only experts (needs per-expert mixed quantisation and a model run), true gate-rank order of the recorded ids, the cost of splitting each expert into two smaller reads.

**Catalog method rows (all 3, mapped to `fam_s2_9`).**

| id | catalog method (full name) | claim vs what the family test shows |
|---|---|---|
| P-0182 | D2MoE: nested MWQ expert weights | Nested low-bit base plus residual bits exists with trained nesting. Our post-hoc split (Q2_K-like base plus Q2_K-like residual, 5.25 bpw) has error 0.085 vs 0.040 for a plain 4-bit scalar at 5.0 bpw, so post-hoc nesting is worse; trained nesting is untested. |
| P-0183 | Matryoshka Quantization / MatGPTQ | Nested int2-in-int4 slicing, trained. The bytes replay (1.43-1.52x at 2.2 footprints) is independent of how the nesting is trained; the quality is not measured. |
| P-0184 | DynaExq and GEMQ per-expert mixed precision | Hotness-driven per-expert bit allocation [corrected after the paper check: DynaExq is hotness-driven; GEMQ is a static gradient-based allocation]. Our replay uses recorded router order as the hotness signal (top-4 enhanced) and gets 1.43-1.52x bytes at 2.2 footprints and up to 1.88x at 4.4; the 7% MMLU drop at 2.5 bit is not reproduced [corrected after the paper check: the 7% drop (MMLU 70.37 -> 63.21, 87 GB -> 16.3 GB) is a static GEMQ allocation result that we never tested, so 'not reproduced' is wrong; it is untested]. |

---

## What was not done

- No hardware other than this x86 Mac; the LUT microbenchmark is single-threaded, an AVX2-gather design only.
- No model-level quality measurement for any lossy lever (families 2, 5, 6, 8, 9).
- No 125B data; all trace results come from the Qwen3-30B traces and all weight probes from one small MoE.
- The small-MoE prompt-cache run (family 3) and the LUT microbenchmark are one-off measurements recorded here, not in the test suite.


## Corrections from the deep paper check (papers/DEEP_SUMMARY.md)
- ProMoE vs an LRU baseline is 1.31x (transformers) and 1.09x (llama.cpp); HOBBIT's prefetch alone is 1.01-1.05x. Their larger headline ratios come from byte cuts against weak or thrashing baselines, not stronger predictors.
- P-0184 (GEMQ, static gradient-based allocation) reports 87 GB -> 16.3 GB with MMLU 70.37 -> 63.21; we never tested it. DynaExq is the hotness-driven paper.
- Dr.LLM skips about 1% of layer applications, so it is no evidence for 25-67% skip; only ShortGPT matches the 10-25% claim.
- Lookup-table kernels: independent data (Intel AVX2 paper, CARDAN, FASQ) agree with our microbenchmark.
