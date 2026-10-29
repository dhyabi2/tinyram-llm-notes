# Group F: screen 2 families #18 to #25

Code: <tinyram-llm>/tools/pubtech/f_F.py (functions fam_s2_18 .. fam_s2_25, registry FAMILIES).
Tests: <tinyram-llm>/tests/test_pubtech_F.py (149 checks, about 50 s). Hygiene suite: 171/171.

Common method: the three recorded Qwen3-30B traces (128 experts, top-8, 48 layers, 249 to 300 decode tokens each) stand in for the 125B. Bytes are mapped by token-footprints (cache entries = footprints x 384). 8/16/32 GB machines = 2.07 / 7.58 / 20.84 footprints. Cold-start-included LRU hit on the traces: 0.562 / 0.951 / 0.962. Rates used: 3.11 GB/s explicit reads (combo model), 2.4 GB/s (the screen's figure), 0.34 GB/s mmap paging. Compute: 12 GFLOP per token, 55 GFLOP/s decode (0.218 s, ceiling 4.583 tok/s), 311 GFLOP/s batched (OLMoE prefill figure, an assumption for k=2..16 token batches).

Caveats that apply to every family: decode traces are not prefill traces; 30B routing is not 125B routing; the 311 GFLOP/s batched rate is measured on OLMoE prefill only; every family run is a projection, never a 125B run. Where a "screen" model and a "combo" model are both given, the screen model assumes 311 GFLOP/s for any batch of 2 or more tokens (favourable), the combo model uses 55 GFLOP/s times the combination model's batch efficiency (1.6x at 4, 2.2x at 8; conservative).

---

## Family 18: expert store in a peer machine's RAM over RDMA (P-0232)

Idea: hold the expert store in a peer's RAM and read missed experts at about 11 GB/s instead of local NVMe.

Measured (closed form on trace LRU hits, serial = compute + reads, overlap = max(compute, reads)):

| RAM | hit | local 3.11 GB/s serial / overlap | peer 11 GB/s serial / overlap | gain serial / overlap vs local | vs mmap (serial) |
|---|---|---|---|---|---|
| 8 GB | 0.562 | 2.50 / 4.58 | 3.71 / 4.58 | 1.49x / 1.00x | 7.0x |
| 16 GB | 0.951 | 4.19 / 4.58 | 4.47 / 4.58 | 1.07x / 1.00x | 1.8x |
| 32 GB | 0.962 | 4.28 / 4.58 | 4.49 / 4.58 | 1.05x / 1.00x | 1.6x |

Against the screen's 2.4 GB/s path at 8 GB: serial 2.20 -> 3.71 = 1.69x, overlap 4.23 -> 4.58 = 1.08x. Combo model cross-check agrees within 1%. Claim arithmetic: 117.8 ms vs 540 ms per token is 4.58x on I/O only; the "8 tok/s" (pure I/O 8.49 tok/s) exceeds the 4.58 tok/s compute ceiling. A 10GbE-class peer (1.1 GB/s, typical of llama.cpp RPC guides [corrected after the paper check: no guide gives a 10GbE figure; the Arm guide has none]) is slower than local NVMe (8 GB: 1.36 vs 2.50 serial). Rate needed to stop being I/O-bound under perfect overlap: 2.6 GB/s at 8 GB, below 0.3 GB/s at 16/32 GB. Per-fetch latency (480 x miss share fetches): 2 ms (RDMA 10 us) or 21 ms (TCP 100 us) per token at 8 GB.

Verdict: confirmed as the screen computed it (about 1.5x serial, 1.1x overlap at 8 GB, about 1.0x at 16/32 GB, about 7x only versus mmap). No disagreement with the screen (its hand figures reproduce to 2 decimals). needs hardware: a real RDMA/TB5 peer was not available, so only the ceiling arithmetic is measured.

Published methods vs measured: RPC backend guides report 2 to 5 tok/s for 70B on CPU clusters with 10GbE as minimum [corrected after the paper check: this line has no source; the Arm guide reports 1.75 tok/s (Llama-3.1-70B on 3 nodes) and no link figure]; our arithmetic shows 10GbE is below local NVMe speed for this layout. MegaScale-Infer is a datacentre GPU/RDMA system, not comparable. Issue 19745 is a load-time hang above 100 GiB, no performance figure.

Catalog rows (all mapped to the fam_s2_18 test): P-0232 idea; P-0233 llama.cpp RPC backend (rpc-server holds weights in remote RAM); P-0234 MegaScale-Infer, disaggregated expert parallelism; P-0235 llama.cpp issue 19745, RPC large-model load hang.

Unmeasured: any real network transfer, fabric latency/queueing, the decode-on-the-peer variant.

---

## Family 19: quantized residual against the previous layer's predicted expert (P-0236)

Idea: store layer-L expert slices as residuals against the expert the layer L-1 router predicts, claiming 2.7 MB -> 0.69 MB for predicted slices.

Measured on the real Qwen3-30B-A3B Q4_K_M weights (own stdlib GGUF reader and Q4_K dequantizer, no downloads): gate matrices of layers 10 and 11, 16 rows spread over each expert (32,768 weights), 8 target experts of layer 11 against 24 candidate experts of layer 10 (192 pairs), plus controls.

| pairs | median abs cosine | max abs cosine | median RMS(a - c b)/RMS(a) | bits/weight saved |
|---|---|---|---|---|
| layer 11 vs layer 10, random | 0.0047 | 0.017 | 0.99999 | 0.00001 |
| same layer, different experts (control) | 0.0055 | 0.013 | 0.99999 | |
| same expert id in the previous layer | 0.0063 | 0.013 | 0.99998 | |
| trace co-activation-chosen base | 0.0035 | 0.013 | 0.99999 | 0.00001 |
| best of 24 candidates per target (selection bias) | | | 0.99991 | 0.00013 |

The identity ratio = sqrt(1 - cos^2) holds exactly with the least-squares scalar. A planted-correlation control (rho 0.8) is detected at ratio 0.60 (0.74 bit saved). Kill threshold of the screen (median ratio above 0.85): reached by a wide margin; the residual compression factor r = 1.000, so the bytes gain bound is 1.00x. Bound table 1/((1-p)+p/r): r=1.9 gives 1.35x, r=4 gives 1.70x, r=infinity 2.22x (p=0.55). The cross-layer predictor itself (top-8 by co-activation): precision 0.584 in-sample, 0.535 held out (first half train, second half test), versus the earlier 0.55 in-prompt / 0.31 to 0.43 held-out figures. The held-out value here is higher than the earlier figure because this predictor is trained on the same prompt's earlier tokens (same distribution), not on other prompts. Claim arithmetic: 1300 MB / 1.9 = 684 MB, not 331 MB; 331 MB implies 3.93x; 3.93 x 2.2 = 8.65 is the 8.7x headline.

Verdict: refuted as a bytes lever (about 1.0x). Matches the screen's expectation (|cos| below 0.02, saving below 0.05 bit) and the literature (near-zero cross-layer cosine). Disagreement: none on the weights; the predictor precision differs from the earlier held-out figure (see above).

Published methods vs measured: D2-MoE, DeRS, the Trilemma paper and LorExperts/BTExperts use a shared base within a layer or upcycled-from-dense experts (where experts are copies); our cross-layer pairs have no such correlation. DIMoE-Adapters (cosine near zero across layers) is reproduced (0.005) [corrected after the paper check: drop this citation; DIMoE-Adapters is a CLIP LoRA continual-learning paper whose matrices are across tasks, not layers; our probe tested gate matrices only without neuron alignment, so the result is 'no byte-level cross-layer residual'; functional cross-layer redundancy is published (CS-MoE, Meta Expert)]. The cross-layer sharing paper (CKA above 0.85) is rotation-invariant and does not imply element-wise similarity; our element-wise test is the relevant one for a residual layout.

Catalog rows (all mapped to fam_s2_19): P-0236 idea; P-0237 D2-MoE: Delta Decompression for MoE-based LLMs Compression (ICML 2025); P-0238 DeRS: Towards Extremely Efficient Upcycled MoE Models; P-0239 Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression; P-0240 Shape Mutating Expert Compression: LorExperts and BTExperts; P-0241 Improving Parameter Utilization by Sharing Neural Experts Across Layers [corrected after the paper check: CKA>0.85 is a post-hoc zero-shot swap on activations, not weights]; P-0242 DIMoE-Adapters [corrected after the paper check: wrong paper for the cross-layer claim, see above].

Unmeasured: 16 rows of the gate matrix only (not up/down, not all rows or layers), the 30B not the 125B, learned shared bases (clustering, SVD) which the published methods use and this raw-neighbour test does not, reconstruction quality.

---

## Family 20: provisional tokens after the first 64-token prompt block (P-0243)

Idea: after block 1 is prefilled start emitting tokens from the partial prompt, verify later.

Measured (share of experts a window of n consecutive decode tokens touches; decode proxy because no prefill trace exists):

| n tokens | measured db / story / code | uniform routing, 30B |
|---|---|---|
| 1 | 0.0625 all | 0.0625 |
| 8 | 0.232 / 0.215 / 0.235 | 0.403 |
| 64 | 0.447 / 0.428 / 0.516 | 0.984 |
| 249 (all) | 0.591 / 0.552 / 0.698 | 1.0 |

Mapped to the 125B (66.4 GB of experts, 3.11 GB/s, 311 GFLOP/s): block 1 (64 tokens) compute 2.47 s, 500-token compute 19.3 s (the claim's 14 s and 109 s use the 55 GFLOP/s decode rate for prefill). Block-1 share of experts touched: 0.46 (decode proxy), 0.34 (proxy clustering scaled to 125B), 0.72 (uniform routing). Block-1 time / full-prompt time: 0.51 / 0.37 / 0.72. So the first block is 1.4x to 2.7x earlier than the full prompt, never 6 to 8x. Token-chunked prefill (8 chunks of 64) reads 4.4x to 6.0x the expert bytes of one pass (the layered-prefill point). Speculative-prefill bound (keep a fraction of the prompt): keeping 50% gives 1.48x, 25% gives 1.80x, 12.5% gives 2.77x TTFT, because expert bytes saturate before compute shrinks (scaled model). The screen's kill criteria (block over 50% of the time, or over 50% of experts) fire only under the uniform and the raw proxy for time; sensitive to the routing model, and prefill routing is likely closer to uniform than decode.

Optional host measurement (OLMoE Q3_K_S, in RAM, shared machine at load 6 to 9, llama-bench -p 64,500 -n 0 -t 4): 64 tokens took 0.84 to 0.87 s, 500 tokens 3.96 to 4.34 s, ratio 0.19 to 0.22 (a first cold run gave 0.40). This is compute only (the model fits in RAM); it says nothing about the I/O side. Not part of the default test (not deterministic).

Verdict: refuted for the claimed 6 to 8x time-to-first-useful-token; partly open on the decisive question whether provisional tokens from a 64-token prompt match the full answer (needs a model; unmeasured). 0x on decode tok/s by the idea's own statement.

Disagreement with the screen: the screen expected 108 to 120 of 128 experts per 64-token window from decode traces; the traces show 55 to 66 (decode is much more clustered than uniform). The conclusion (block 1 is not cheap) survives under uniform routing but not under the decode proxy; flagged as routing-model dependent.

Published methods: SpecPrefill (7.66x TTFT on a 405B by pruning prompt tokens) and Cross-Family Speculative Prefill attack compute and bytes of prefill; our bound shows bytes limit the gain to 1.5 to 2.8x at 50 to 12.5% kept here. Sarathi-Serve is a serving-capacity method and emits nothing early. Layered prefill for MoE: confirmed by the 4.4x to 6.0x re-read figure.

Catalog rows (all mapped to fam_s2_20): P-0243 idea; P-0244 SpecPrefill (ICML 2025), a small draft model picks the important prompt tokens and only those go to the big model; P-0245 Cross-Family Speculative Prefill, ICLR 2026; P-0246 Sarathi-Serve chunked prefill with stall-free batching, OSDI 2024; P-0247 Layered prefill for MoE serving.

Unmeasured: real prefill routing, answer quality from truncated prompts, I/O-bound prefill timing on any real MoE.

---

## Family 21: prompt-lookup / suffix n-gram drafting (P-0248)

Idea: draft spans from repeated context, verify in one batched pass; claimed 1.4x at f=0.6, 1.8x at 0.75, 3x at 0.85.

Measured: greedy prompt lookup (suffix 4 then 3 tokens, draft up to 8) over regex-tokenized text, teacher-forced (the text is what the model would emit), first 30% as prompt. Verify cost per pass from the trace LRU (union of the k tokens' experts, serial, 3.11 GB/s). The corpora are stand-ins: no saved model outputs exist (Python documentation text, this repo's scripts, seeded JSON rows, a 10%-edited copy, random tokens).

| text | f_accepted | serial speedup 8 / 16 / 32 GB (screen batch model) | combo batch model |
|---|---|---|---|
| prose (Python docs) | 0.107 | 0.83 / 1.02 / 1.03 | 0.75 / 0.84 / 0.84 |
| code (repo scripts) | 0.158 | 0.74 / 1.00 / 1.02 | 0.65 / 0.77 / 0.77 |
| JSON rows | 0.590 | 0.91 / 1.58 / 1.65 | 0.74 / 0.96 / 0.97 |
| 10% edited copy | 0.593 | 1.10 / 1.78 / 1.84 | 0.91 / 1.15 / 1.16 |
| random (control) | 0.000 | 1.00 | 1.00 |

Unique experts per layer for k=2/3/4/8 consecutive tokens: 12.58 / 16.31 / 19.62 / 29.07 (vault: 12.6 / 16.3 / 19.6 / 29.1). But with an LRU of 2 to 21 footprints, misses per token for k=8 grouping are 1.02x / 0.97x / 0.97x of k=1: the union saving (0.454) vanishes behind the cache, as the first screen found. The screen's formula at 100% acceptance reproduces: f=0.15 / 0.3 / 0.6 at 8 GB = 1.07 / 1.15 / 1.36 (screen 1.1 / 1.2 / 1.6), at 32 GB f=0.6 = 1.86, f=0.85 = 2.89 (below 3x).

Verdict: refuted for typical text (prose and code: f_accepted below the screen's own 0.3 kill line, speedup 1.0x or less at 8 GB and at most 1.03x at 16/32 GB); only high-repetition outputs (JSON rows, edits) reach 1.6 to 1.8x at 16/32 GB with the favourable batch model, and about 1.0 to 1.16x with the conservative one. The screen's "1.05 to 1.2x typical" is not reproduced (measured at or below 1.0 at 8 GB). Disagreement recorded: a rejected draft costs a whole batched pass of experts, which makes speculative drafting a net loss at 8 GB on low-overlap text.

Published methods: Prompt Lookup / llama.cpp lookup and ngram-cache are exactly the mechanism implemented here; SAM Decoding and N-Gram Trie report gains on dense models where I/O is not binding; "Performance or Illusion" reports that n-gram gain tracks overlap and low-overlap requests lose, which is what the table shows. MoE-Spec/EVICT: deeper drafts activate more unique experts, matching the verify-cost growth (pass cost at 8 GB: 0.40 s for 1 token, 0.44, 1.15, 1.79 s for 2, 4, 8).

Catalog rows (all mapped to fam_s2_21): P-0248 idea; P-0249 Prompt Lookup Decoding (apoorvumang) and llama.cpp examples/lookup (PR #4484), now --spec-type ngram-cache/ngram-simple/ngram-map; P-0250 SAM Decoding: Speculative Decoding via Suffix Automaton (ACL 2025); P-0251 Speculative Decoding: Performance or Illusion? (arXiv 2601.11580); P-0252 MoE-Spec / EVICT / AcceptMoE / SpecMoEOff (MoE speculative decoding); P-0253 N-Gram Trie Speculative Decoding (EMNLP 2025).

Unmeasured: acceptance on real model outputs (none saved), a real llama.cpp lookup run, a suffix automaton (a hash n-gram index with the same matches is used), spans longer than 16 tokens.

---

## Family 22: prefill while the user is typing (P-0254)

Idea: chunked prefill on the partial prompt during typing, so only the last chunk remains at Enter.

Measured: unique experts in a 64-token window of the traces: db 57.2, story 54.8, code 66.0 of 128 (45%, 43%, 52%); the screen expected 90% or more and a premise cut at 85%: not met by the decode proxy (uniform routing would give 98% on the 30B and 72% on the 125B). Event simulation of a 500-token prompt, 64-token chunks, sequential passes (pass time = max(compute, expert bytes / 3.11 GB/s)):

| routing model | one 64-token pass | one 500-token pass | TTFT gain, typed 1 to 3 tok/s | typed 5 tok/s | pasted but chunked anyway | disk bytes vs one pass |
|---|---|---|---|---|---|---|
| decode proxy | 9.9 s | 19.3 s | 2.04x | 2.04x | 0.25x | 6.0x |
| scaled to 125B | 7.2 s | 19.3 s | 3.07x | 3.07x | 0.34x | 4.3x |
| uniform routing | 15.3 s | 21.3 s | 1.56x | 0.64x | 0.18x | 5.7x |

Under uniform routing a chunk pass (15.3 s) is slower than the time the user needs to type 64 tokens at 5 tok/s (12.8 s), so the passes queue behind the typist and the gain turns into a loss. The decode cache also starts cold: first-10-token LRU hit 0.484 versus 0.569 later at 8 GB (0.4 s of extra reads; this penalty exists with or without chunking). Prefix-reuse helper (common_prefix_len) tested on hand cases.

Verdict: partly. The claimed 7.8x TTFT is refuted (3.1x at best, 1.6x to 2.0x under the other models); decode tok/s gain is 1.0x by construction; typed prompts gain, pasted prompts lose if chunking is forced, and the disk work grows 4 to 6x. Disagreement with the screen: it expected a near-full weight read per 64-token chunk (about 21 s); the decode proxy gives 7 to 10 s and uniform gives 15 s (the 14 s kill line is model dependent).

Published methods: same-slot prefix KV reuse, --cache-ram and the radix cache already provide exact-prefix reuse (the mechanism is not new); nothing there triggers on keystrokes. Sarathi-Serve reports higher average TTFT with chunking, consistent with the extra passes here.

Catalog rows (all mapped to fam_s2_22): P-0254 idea; P-0255 llama.cpp same-slot prefix KV reuse (server cache_prompt, LCP slot choice); P-0256 llama.cpp host-RAM prompt cache, --cache-ram; P-0257 Mini-SGLang chunked prefill with radix cache; P-0258 Sarathi-Serve (OSDI 2024), chunked prefill (secondary source).

Unmeasured: real chunk timing, tokenization at chunk boundaries and mid-text edits, KV validity after edits, decode-cache state after chunk passes.

---

## Family 23: ahead-of-time specialised kernels to remove dispatch overhead (P-0259)

Idea: 55 GFLOP/s decode and 311 GFLOP/s prefill are dispatch artifacts; AOT fusion would cut about 60% of prefill (109 s -> 45 s).

Computed: 500 tokens x 12 GFLOP = 6 TFLOP = 109.1 s at 55 GFLOP/s (the claim's baseline) but 19.3 s at 311 GFLOP/s; 311 GFLOP/s x 109 s = 33.9 TFLOP, 5.6x the work of a 500-token prompt, so the claim's baseline is inconsistent. Operations: 144 graph matmuls, 1,440 slice matmuls per decode token, 73,728 expert matmuls per prefill pass.

| dispatch cost per op | decode share (1,440 ops) | decode Amdahl bound | prefill share of 19.3 s | prefill bound |
|---|---|---|---|---|
| 2 us | 1.3% | 1.013x | 0.8% | 1.008x |
| 5 us | 3.3% | 1.034x | 1.9% | 1.020x |
| 10 us | 6.6% | 1.071x | 3.8% | 1.040x |
| 50 us | 33% | 1.49x | 19% | 1.24x |

Removing 60% would need 91 us per slice matmul (decode), 909 us per graph op, or 157 us per expert matmul in prefill (implied rate multiplier 2.5x). For context: if compute could be doubled by any means, serial decode gains 1.37x at 8 GB and 1.84x/1.87x at 16/32 GB, so the question is only the dispatch share.

Host measurements (one-off, shared machine at load 6 to 9, so noisy; helpers olmoe_thread_scaling and tiny_decode_bound in f_F.py, not run by the default test): OLMoE Q3_K_S prefill -p 512 scaled 2.6x to 2.8x from 1 to 4 threads (45.6 to 51.1 tok/s at 1 thread, 120 to 141 at 4), not the 4x of a purely compute-bound run, but the load of other jobs explains an unknown part. A 2-layer tiny MoE decoded 365 to 982 tok/s across runs (-t 1/-t 4); its whole per-token time over its layers is an upper bound on per-layer overhead (it includes real compute and the output head): 0.51 to 1.37 ms per layer, i.e. at most 24 to 66 ms over 48 layers, 11% to 30% of the 218 ms decode token (Amdahl at most 1.13x to 1.43x), far below the 60% needed. No profiler run (the share outside matmul loops) was done.

Verdict: refuted by arithmetic and by the measured upper bound (dispatch is at most about 11 to 30% on a loose bound and about 1 to 7% on any realistic per-op cost; the claim needs 60%); decode at 8 GB is I/O-bound anyway (1.0x). The kernel-efficiency route (better quantised GEMM) is a separate lever not tested here. Disagreement with the screen: none on the arithmetic; the screen's "under 1% of prefill" refers to its 109 s base, against the 19.3 s corrected base the share is up to 3.8% at 10 us per op.

Published methods: TaxBreak/launch overhead is a GPU result; fused GPU kernels (Mixtral 15 to 20%, OLMoE Triton 0.999x) have no CPU analogue; KTransformers' 4.6x to 19.7x prefill comes from AMX arithmetic efficiency and hardware, not dispatch removal; ik_llama.cpp and MobileMoE give no comparable number [corrected after the paper check: ik_llama.cpp PR 229 gives +3-4% prompt processing and +1-2% token generation on CPU, and ASAP measures +6% on NPU; both agree with our 1-7% dispatch share]; ASAP concerns NPU dispatch of tens to hundreds of microseconds per layer (a different regime).

Catalog rows (all mapped to fam_s2_23): P-0259 idea; P-0260 TaxBreak: overhead decomposition of LLM inference; P-0261 Fused-kernel gains on GPU (search snippet; the Mixtral and OLMoE figures come from separate sources that I did not open); P-0262 ik_llama.cpp; P-0263 KTransformers (SOSP 25); P-0264 MobileMoE (search snippet); P-0265 ASAP (NPU MoE prefill).

Unmeasured: a profiler share of time outside the matmul inner loops, any fused/AOT kernel, any x86 AVX2 result (all host numbers are Apple silicon).

---

## Family 24: grammar-constrained jump-forward decoding (P-0266)

Idea: when the grammar leaves one legal token, append it without sampling; claimed 3.3x at f=0.7.

Measured: a token-level grammar automaton over 20 seeded synthetic schema outputs (5 schemas: plan, verdict, tool call, record, claim; compact JSON, fixed property order, enums, optional tail properties; every output parses with json.loads and equals its generated value). A token is forced iff all its characters are forced; a rule-based tokenizer in two variants. Forward sizes follow the jump-forward rule (one forward per free token, carrying the forced run before it, plus a final free end token). Costs from the trace LRU pass costs (serial).

| tokenization | tokens | f | forced runs | mean run | serial speedup 8 / 16 / 32 GB (screen batch model) | combo batch model | ideal 1/(1-f) |
|---|---|---|---|---|---|---|---|
| punctuation as single tokens | 1542 | 0.608 | 175 | 5.35 | 1.46 / 2.22 / 2.27 | 1.22 / 1.48 / 1.49 | 2.55x |
| punctuation merged (BPE-like) | 992 | 0.440 | 134 | 3.25 | 1.24 / 1.77 / 1.80 | 1.06 / 1.26 / 1.27 | 1.78x |

Per-output f ranges 0.25 to 0.58 (merged). At 8 GB with perfect overlap the merged variant is 0.98x (the extra experts of a batch cost bytes). The screen's own bound at f=0.7 reproduces: 1.45x at 8 GB, 2.12x to 2.17x at 16/32 GB, below the 3.3x claim. Free text (control) has f = 0 and exactly 1.0x.

Verdict: partly. The 3.3x claim is refuted (1.1x to 1.5x with the conservative batch model, up to 2.3x with the favourable one, only on schema-constrained output with a mostly fixed structure; 1.0x on free text). Disagreement with the screen: it expected f of 0.2 to 0.4 and mean run 2 to 3; these synthetic outputs give f 0.44 to 0.61 and mean runs 3.3 to 5.4 (merged f is below 0.5, so the screen's kill flag fires for merged tokens only). f here depends on my choices of value length (1 to 9 words), so it is a model of the idea, not a measurement of the MCP planner/verifier outputs (private, not used).

Published methods: LMSYS compressed FSM and the SGLang paper prefill forced tokens in one step (what is simulated); XGrammar and llguidance add masking and fast-forward; vLLM/BentoML note that the saving is batching, not zero work (the batched pass cost is exactly what bounds the gain here); DOMINO/Formatron report 1.5 to 2x, consistent with the 1.3x to 2.3x range [corrected after the paper check: void for Formatron (it measures mask speed); DOMINO reports 1.66-1.91x on GPU with speculative tokens; llguidance reports 12.7% forced tokens, far below our synthetic 0.44-0.61, so the ideal bound is about 1.15x until real planner/verifier outputs are measured; naive forced-token insertion is non-canonical in 97% of cases without token healing].

Catalog rows (all mapped to fam_s2_24): P-0266 idea; P-0267 LMSYS compressed FSM / jump-forward decoding (SGLang); P-0268 SGLang paper; P-0269 XGrammar paper; P-0270 llguidance fast-forward tokens (used in llama.cpp via PR 10224); P-0271 vLLM structured decoding note (BentoML); P-0272 Reviews of constrained-decoding speedups (DOMINO, Formatron arXiv 2506.01151) [corrected after the paper check: arXiv 2506.01151 is the Earley-driven pruning method paper, not a review; its gains are from mask cost].

Unmeasured: real tool-call outputs and the real tokenizer, quality effects of different tokenization of forced text, llguidance behaviour inside llama.cpp.

---

## Family 25: 2-bit base plus 4-bit refinement tiles (P-0273)

Idea: store 2-bit base plus 4-bit tiles (about 6 bpw); read the base for all slices, refinement only for slices above a residual threshold; claimed 3.3 to 4.2 tok/s at 8 GB.

Measured (mean of 3 traces; 2.2 bpw slices make the cache hold 1.955x more entries; ceiling 4.583):

| RAM | hit 4.3 bpw -> 2.2 bpw | baseline serial / overlap at 2.4 GB/s | pure 2.2 bpw | tiles on misses (10%) | tiles every token (10%) |
|---|---|---|---|---|---|
| 8 GB | 0.562 -> 0.798 | 2.20 / 4.23 | 3.65 (1.66x) | 3.52 (1.60x) | 3.08 (1.40x) |
| 16 GB | 0.951 -> 0.962 | 4.09 / 4.58 | 4.37 (1.07x) | 4.34 (1.06x) | 3.59 (0.88x) |
| 32 GB | 0.962 -> 0.962 | 4.19 / 4.58 | 4.37 (1.04x) | 4.34 (1.04x) | 3.59 (0.86x) |

Overlap gain at 8 GB: 1.085x at 2.4 GB/s, 1.00x at 3.11 GB/s. The screen's figures reproduce (2.19 / 4.20 baseline, 1.4 to 1.6x serial, 1.09x overlap at 8 GB, 1.03 to 1.07x at 16/32 GB). New: re-reading the refinement tiles every token (what the idea literally describes) is slower than plain 4-bit LRU at 16/32 GB (0.86x to 0.88x). The file grows to 96 GB at 6 bpw (vs 69 GB).

Residual structure on the real weights (gate matrices of the 30B, 16 experts x 16 rows, layers 10 and 30, requantized to 2 bits per 16-weight group from the already-Q4_K values, so an upper bound on error): relative residual norm 0.321 / 0.320 with a coefficient of variation of only 0.010 / 0.008 across expert slices (the screen: nothing to select if below 0.2); the top 10% of slices hold 14.0% / 14.1% of the residual energy and the top 10% of 32-weight groups hold 19.8% (uniform would be 10%). Control: four heavy-tailed slices among 16 Gaussians raise the top-10% slice share from 14% to 54%, so the statistic can detect selectable heterogeneity. Refining 10% of slices therefore removes only about 14 to 20% of the 2-bit error.

Verdict: partly confirmed as the screen computed it, but as an 8 GB-only trick (1.4 to 1.66x serial, 1.0 to 1.09x overlap, about 1.0x at 16/32 GB), indistinguishable from plain 2.2 bpw, and with nothing to select on (residual spread evenly). Quality is unmeasured (only OLMoE Q2_K +26% perplexity, requantized, an upper bound).

Published methods: DynaExq keeps hot experts at higher precision with a low-precision fallback (2.73x on GPU at batch 32, not comparable); Any-Precision LLM and AnyBCQ are the bit-plane storage layout/kernels (a layout with no byte saving by itself); nothing there supplies a per-slice trigger from residual norm.

Catalog rows (all mapped to fam_s2_25): P-0273 idea; P-0274 DynaExq (2511.15015); P-0275 Any-Precision LLM (2402.10517), github.com/SNU-ARK/any-precision-llm style bit-plane layout; P-0276 AnyBCQ (2510.10467).

Unmeasured: perplexity/KL of a 2-bit base on the 125B, the per-token trigger, tile layout I/O, statistics beyond 16 experts x 16 rows of the gate matrix.

---

## Summary

| family | verdict | key number |
|---|---|---|
| 18 peer RAM over RDMA | confirmed (needs hardware) | 1.49x serial / 1.00x overlap at 8 GB, 1.05 to 1.07x at 16/32 GB; 7x only vs mmap |
| 19 residual vs predicted previous-layer expert | refuted | median |cos| 0.005, residual/target ratio 0.99999, bytes bound 1.00x |
| 20 provisional tokens after block 1 | refuted on time (6 to 8x); quality part needs a model | block 1 is 1.4x to 2.7x earlier than the full prompt; chunking re-reads 4.4 to 6.0x bytes |
| 21 prompt-lookup / n-gram drafting | refuted for typical text | prose f 0.11, code 0.16; speedup at most 1.03x at 16/32 GB, below 1.0x at 8 GB |
| 22 prefill while typing | partly (TTFT only) | 1.6x to 3.1x TTFT for typed prompts, 0.2 to 0.3x if forced on a paste, 4 to 6x disk bytes |
| 23 AOT kernels vs dispatch | refuted (host profile still open) | dispatch 1 to 7% of decode on realistic costs, 60% needed; loose measured bound 11 to 30% |
| 24 grammar jump-forward | partly | f 0.44 to 0.61, speedup 1.1x to 1.5x (conservative) up to 2.3x (favourable), claim 3.3x |
| 25 2-bit base + 4-bit tiles | partly (8 GB-only) | 1.4x to 1.66x serial at 8 GB, 1.0x at 16/32 GB, tiles every token lose (0.86x) |


## Corrections from the deep paper check (papers/DEEP_SUMMARY.md)
- Family 24: "DOMINO/Formatron report 1.5-2x" is void for Formatron (it measures mask speed); DOMINO itself reports 1.66-1.91x (GPU, speculative tokens). llguidance reports 12.7% forced tokens on its benchmark, far below our synthetic 0.44-0.61, so the ideal bound is about 1.15x until real planner/verifier outputs are measured; naive forced-token insertion is non-canonical in 97% of cases without token healing.
- Family 23: ik_llama.cpp PR 229 does give a CPU number (+3-4% prompt processing, +1-2% token generation); ASAP measures +6% on NPU. Both agree with our 1-7% dispatch share.
- Family 18: the RPC-guide line "2 to 5 tok/s with 10GbE minimum" has no source; the Arm guide reports 1.75 tok/s (Llama-3.1-70B on 3 nodes) and no link figure.
- Family 19: drop the DIMoE-Adapters citation (it is a CLIP LoRA paper). Our probe tested gate matrices only without neuron alignment, so the result is "no byte-level cross-layer residual"; functional cross-layer redundancy is published (CS-MoE, Meta Expert).
- Family 22: OpenAI now documents prompt_cache_options.prewarm (GPT-5.6+) for a known fixed prefix; Anthropic's cookbook describes typing-time cache warming for fixed context.
