# Master idea register

80 canonical ideas from 349 raw ideas: 252 brainstorm families (1,502 engine runs over 7 rounds), 56 angle-sweep ideas, and 41 ideas from the working sessions. Status is judged against the vault, the measurements, the trace simulations and the paper checks. Raw-to-canonical mapping: `IDEA_REGISTER.csv` (column ids = ids in `raw_ideas.json`). Status counts: measured_positive 9, measured_neutral 5, measured_refuted 10, simulated_positive 7, needs_quality_eval 7, untested 18, needs_hardware 3, published_only 2, simulated_negative 19.

'applies' says where the idea would run: any_pc, os_specific, windows_port_needed, needs_second_machine, needs_gpu, needs_special_hardware, needs_training.


## measured positive

| idea | category | applies | raw | headline result | next step |
|---|---|---|---|---|---|
| **overlap io with compute** - Overlap expert reads with compute using concurrent I/O in the pool. | prefetch_overlap | any_pc | 14 | About 1.15-1.16x over the synchronous pool on decode (Mac); prefill overlap untested. | Measure prefill TTFT overlap and port the async reader to Linux/Windows. |
| **explicit read pool** - Large concurrent explicit preads (O_DIRECT/F_NOCACHE) with a byte-capacity LRU instead of mmap. | io_read_path | any_pc | 8 | 4.4-5.2x over plain mmap on the Mac with Qwen3-30B and 9 GB ballast. | Port and verify on Linux and Windows PCs, with auto-sized pool. |
| **thread count and barrier tuning** - Use physical cores and tune barrier spinning. | os_memory_settings | any_pc | 4 | Hyperthread oversubscription collapses ggml barrier spinning 11x; physical cores only. | Auto-detect physical cores in the tool. |
| **compute versus I/O bound measurement** - Measure whether decode and prefill are compute or I/O bound. | measurement_method | any_pc | 1 | About 55 GFLOP/s measured on 30B; 4.6 tok/s ceiling for the 125B. | Build as an auto-calibration step in the tool for other PCs. |
| **madvise read-ahead hints** - Use madvise hints with mmap. | io_read_path | os_specific | 1 | 1.65x on throttled Linux VPS, 1.08x on Mac; far below explicit pool. | Use only as Linux fallback. |
| **wired ballast test method** - Wire memory to emulate a small-RAM PC. | measurement_method | windows_port_needed | 1 | Verified 96-100% emulation without sudo. | Reuse on every new PC. |
| **disable repack** - Pass --repack 0 to keep the pool path usable. | os_memory_settings | any_pc | 1 | Required on small RAM. | Make default. |
| **free ram for page cache** - More free RAM helps mmap. | os_memory_settings | any_pc | 1 | Helps plain mmap strongly; pool still wins at same RAM. | Document. |
| **validity-gated ab method** - A/B with validity gates. | measurement_method | any_pc | 1 | Caught invalid cached runs and noise-level claims. | Package in the portable tool. |

## measured neutral

| idea | category | applies | raw | headline result | next step |
|---|---|---|---|---|---|
| **cross-layer expert prefetch** - Predict next-layer experts from earlier signals and prefetch them. | prefetch_overlap | any_pc | 14 | At most 1.0-1.075x; 0.55 precision is in-sample and held-out is lower. | Stop; do not invest further. |
| **multi-drive striping** - Stripe reads across drives or NAND dies. | hardware_tier | needs_special_hardware | 5 | Single drive saturates at about 3.1-3.4 GB/s with 4 threads; second drive expected 1.1-1.2x serial. | Test only if a second NVMe is present. |
| **hand-tuned kernel variants** - Hand-tuned int4, fused, nibble-skipping or cache-tiled kernels. | compute_kernel | any_pc | 4 | Decode shows no kernel gain; prefill 950 GFLOP/s claim unmeasured. | Profile prefill only if proven compute-bound. |
| **dense weight residency** - Lock dense weights in RAM. | os_memory_settings | any_pc | 4 | No effect on tok/s with the pool. | Drop. |
| **dram bandwidth ceiling** - Measure DRAM bandwidth ceiling. | compute_kernel | any_pc | 1 | Microbenchmark done; kernel route gave no decode gain. | None. |

## measured refuted

| idea | category | applies | raw | headline result | next step |
|---|---|---|---|---|---|
| **reduced top-k experts** - Activate fewer experts per token (top-6, top-4, adaptive k). | expert_count_routing | any_pc | 17 | Top-4 costs +30.9% ppl; top-6 gives 1.12-1.24x for +4.7% ppl. | Ship top-6 only as an opt-in flag after a perplexity run on the target model. |
| **shared low-rank expert basis** - Factor experts into a shared low-rank basis plus small residuals to cut bytes or FLOPs. | quantisation_bytes | any_pc | 14 | No shared low-rank structure in raw expert weights; flat spectrum, so the 4.3x byte cut is unsupported. | Drop. |
| **idle-time prefill and prefill while typing** - Prefill the typed prefix or warm the cache while the user is still typing. | prefill_ttft | any_pc | 14 | VNNI premise removed; gain only when typing is slow and disk idle; decode 1.0x. | One confirming run limited to exact causal prefix with KV reuse; otherwise drop. |
| **vector-quantised expert codebook lookup** - Replace expert weights with product-VQ codebook indices and a lookup/gather kernel. | quantisation_bytes | needs_training | 8 | Compute half refuted by microbenchmark; byte cut gives about 1.5x serial; 1.3 bpw quality error too high. | Drop the kernel claim; only revisit as a plain byte cut with a perplexity eval at 2 bpw or more. |
| **VNNI int8 kernel** - AVX512-VNNI int8 dot kernels for decode. | compute_kernel | any_pc | 8 | Output identical, no end-to-end speed-up (claimed 1.55x); decode is DRAM-bound. | Drop. |
| **static hot expert pinning** - Pin a prompt-derived or hot set of experts in RAM. | cache_policy | any_pc | 7 | No better than LRU; cross-prompt hit 0.05-0.21; pin by prompt share 0.6-0.75x of LRU. | Drop. |
| **lossless expert compression** - Entropy-code expert weights. | quantisation_bytes | any_pc | 4 | Saves only 2.2-2.6% of bytes. | Drop. |
| **inter-expert dedup** - Deduplicate near-identical experts. | quantisation_bytes | any_pc | 2 | Refuted by null test; claimed 1.00. | Drop. |
| **pre-unpacked int8 weights** - Store weights unpacked as int8 to speed compute. | quantisation_bytes | any_pc | 1 | Slower than Q4_K in RAM-resident decode and doubles bytes read. | Drop. |
| **core split io threads** - Dedicate cores to I/O threads. | os_memory_settings | any_pc | 1 | 2 vs 4 threads shows only 8%. | Drop. |

## simulated positive

| idea | category | applies | raw | headline result | next step |
|---|---|---|---|---|---|
| **remote expert compute over LAN** - A peer PC holds experts in RAM and computes them, shipping small activations over the LAN. | distributed | needs_second_machine | 10 | Best distributed variant on paper (KB activations per layer) but needs a 64 GB+ peer and low-latency link; no run. | Two-PC prototype measuring per-layer round-trip latency. |
| **contiguous per-expert layout** - Repack each expert into one contiguous extent (bundled gate/up/down, kernel-interleaved). | layout_file | any_pc | 10 | Reorder gain at most 1.07x; micro read-rate about 1.15x; most claimed gain is the already-measured large-read effect. | Build the repacked file and measure end-to-end on 8 GB. |
| **kv prefix cache on disk** - Persist KV for repeated prefixes and restore from disk to skip prefill. | kv_context | any_pc | 6 | Lossless TTFT win only on repeated prefixes; no gain on a miss and decode stays 1.0x; not measured. | Implement on-disk KV restore and time TTFT on a repeated system prompt with the pool active. |
| **cache-aware expert routing** - Bias routing toward cached experts. | expert_count_routing | any_pc | 4 | Trace-simulated 1.38x at 8 GB; quality unmeasured. | Perplexity run with logit bias. |
| **grammar jump-forward** - Skip decoding for forced tokens in structured output. | speculation_batching | any_pc | 3 | Small, workload-dependent; claim of 1.5 not supported. | Try on tool-call/JSON workloads. |
| **GPU hot expert tier** - Keep hot experts in VRAM. | hardware_tier | needs_gpu | 2 | Modelled gain only; depends on PCIe and hit rate at 20 GB cache. | Test on a PC with a discrete GPU. |
| **layer-streamed prefill** - Stream layers through prefill so weights are read once per layer for the whole prompt. | prefill_ttft | any_pc | 2 | Positive for TTFT on paper; pool measured on decode only. | Measure TTFT end-to-end. |

## needs quality eval

| idea | category | applies | raw | headline result | next step |
|---|---|---|---|---|---|
| **layer skipping** - Skip or exit early from transformer layers using a gate, router or mixture-of-depths. | expert_count_routing | needs_training | 11 | Only lever reaching about 3x at 67% skip in the model, but no perplexity measured and gate needs training. | Run a training-free layer-ablation perplexity sweep on the 30B. |
| **ultra low-bit experts** - Ternary or 2-bit experts. | quantisation_bytes | any_pc | 4 | Compute cut unproven since low-bit kernels did not speed up; quality unmeasured. | Perplexity eval at 2-bit on Qwen3-30B. |
| **mixed-precision requantisation map** - Requantise down projections or per-layer bit maps. | quantisation_bytes | any_pc | 3 | Down tensors about 45% of an expert; tool built but no perplexity result. | Run tiered perplexity with the existing tool. |
| **cold expert low-bit tier** - Read cold or low-weight experts at lower precision. | quantisation_bytes | any_pc | 2 | Byte gain small once I/O overlaps; ranking unstable across prompts. | Quality eval of mixed-precision miss path. |
| **deferred missed experts** - Defer missed experts to the next layer. | speculation_batching | any_pc | 2 | Lossy and unmeasured. | Skip. |
| **low-rank expert tail substitution** - Replace low-weight experts with a shared rank-64 approximation. | compute_kernel | needs_training | 1 | Quality of replacing 7 of 10 experts is unmeasured and likely poor given the flat spectrum. | Skip unless a basis is ever shown to reconstruct well. |
| **draft-gated big-model skipping** - Skip the big model on easy tokens. | speculation_batching | any_pc | 1 | Lossy unless verified. | Skip. |

## untested

| idea | category | applies | raw | headline result | next step |
|---|---|---|---|---|---|
| **expert output prediction skip** - Predict or approximate expert outputs and skip the expert. | compute_kernel | needs_training | 4 | No measurement; needs trained rank-64 predictor and a skip-vs-perplexity curve. | Cheap one-layer test of predicting expert output from previous activation. |
| **NPU and iGPU offload** - Offload GEMV to an integrated GPU or NPU. | hardware_tier | needs_gpu | 3 | iGPU shares DRAM bandwidth; no Gen11 benchmark found. | Skip. |
| **early draft while prefilling** - Emit a draft answer during prefill. | prefill_ttft | any_pc | 3 | Perceived latency only; lossy. | Skip. |
| **quantised kv cache** - Quantise the KV cache. | kv_context | any_pc | 3 | Bound about 1.06x at short context. | Only for long context. |
| **adaptive pool size** - Size the pool by free memory and pressure. | os_memory_settings | os_specific | 3 | Pool 3000 to 6000 MB gave +16%; adaptive not built. | Build pressure-adaptive sizing so the tool self-tunes on other PCs. |
| **storage hardware tweaks** - Exotic drive or CXL tweaks to reduce storage latency. | hardware_tier | needs_special_hardware | 2 | No evidence of a cliff on our disks; needs special hardware. | Skip. |
| **userfaultfd fault path** - Handle expert page faults in userspace with aligned reads. | prefill_ttft | os_specific | 2 | Explicit pread already bypasses the fault path (measured); Linux-only. | Skip. |
| **expert output memoization** - Reuse cached expert outputs. | compute_kernel | any_pc | 2 | Valid only for identical contexts, already covered by prefix KV cache. | Drop. |
| **router shortlist** - Shortlist router candidates before the argmax. | compute_kernel | any_pc | 2 | Routing likely negligible; unmeasured. | Skip. |
| **compile-time kernel scheduling** - Schedule solvers or per-prompt AOT kernels for prefill. | compute_kernel | any_pc | 2 | No evidence 55 GFLOP/s is schedule-limited; decode is bandwidth-bound. | Profile before building. |
| **superpage wired pool** - Use superpages for the wired pool. | os_memory_settings | os_specific | 2 | At most about 7%; allocation risk; Intel macOS only. | Skip. |
| **background io hygiene** - Close background apps and quiet I/O. | os_memory_settings | os_specific | 2 | Mostly benchmark validity. | Add pre-run checks to the tool. |
| **speculative expert preload for prefill** - Preload experts speculatively for prefill. | prefill_ttft | any_pc | 1 | Gain unknown; little added as prompt tokens are already known. | Skip. |
| **screened output head** - Screen the output head vocabulary. | compute_kernel | any_pc | 1 | Bound at most about 5% on decode. | Skip. |
| **two-stream interleaving** - Interleave two concurrent sessions. | speculation_batching | needs_second_machine | 1 | Applies only with two concurrent sessions. | Skip. |
| **token-id expert read-ahead** - Read ahead experts keyed by token id. | prefetch_overlap | any_pc | 1 | At most about 2%. | Skip. |
| **iGPU second bandwidth client** - Use iGPU as an extra memory client. | hardware_tier | needs_gpu | 1 | Shared-DRAM argument suggests little gain. | Skip. |
| **pre-registered kill thresholds** - Set kill thresholds before runs. | measurement_method | any_pc | 1 | Used informally. | Formalise in the harness. |

## needs hardware

| idea | category | applies | raw | headline result | next step |
|---|---|---|---|---|---|
| **distilled dense student model** - Distill the MoE into a small dense model. | training_required | needs_training | 5 | Removes I/O by construction but needs large-scale distillation and changes the model. | Out of scope. |
| **zoned SSD expert layout** - Zone-aligned expert records on ZNS drives. | layout_file | needs_special_hardware | 3 | ZNS rare; 4 MiB reads already reach 2.4 GB/s on ordinary SSDs. | Skip. |
| **cxl dram expert pool** - Use CXL-attached DRAM to hold experts. | hardware_tier | needs_special_hardware | 2 | Equivalent to model in RAM; decode capped by compute at about 4.6 tok/s. | None without hardware. |

## published only

| idea | category | applies | raw | headline result | next step |
|---|---|---|---|---|---|
| **structured sparsity 2:4** - 2:4 structured weight sparsity. | compute_kernel | needs_gpu | 1 | About 2x only and needs a GPU; no CPU speed-up. | Drop. |
| **router stickiness fine-tuning** - Fine-tune the router for cache locality. | training_required | needs_training | 1 | Requires fine-tuning a 125B router. | Skip. |

## simulated negative

| idea | category | applies | raw | headline result | next step |
|---|---|---|---|---|---|
| **expert-major prefill** - Order prefill by expert so each expert is read once and applied to all tokens. | prefill_ttft | any_pc | 15 | Gain is mostly the read-size effect already in the pool (4.4-5.2x vs mmap); pinning adds nothing beyond LRU. | Measure end-to-end prefill TTFT with pool vs expert-major ordering on one model. |
| **computational storage and FPGA expert offload** - Run expert compute inside the drive, an FPGA or PIM device so weights never cross the bus. | hardware_tier | needs_special_hardware | 13 | About 1.0x vs our best path at 8-32 GB in screening; throughput figures unmeasured and hardware not owned. | Park until a dev board is available; no software-only test exists. |
| **remote expert store over LAN** - Fetch expert weights from a peer's RAM over the network instead of local disk. | distributed | needs_second_machine | 9 | 4.2 vs 4.16 tok/s overlapped; link caps below local NVMe unless 4x25GbE or faster. | None unless a 100GbE-class link is available. |
| **layered base plus residual quantisation** - Store a low-bit base plus refinement residual, reading the residual only when needed. | quantisation_bytes | any_pc | 7 | At most 1.4-1.5x serial and about 1.1x overlapped; quality unmeasured. | Perplexity-check a single base+residual tier before any build. |
| **cache eviction and admission policy** - Learned, segmented, ARC, decayed-frequency or per-sequence cache policies. | cache_policy | any_pc | 7 | All policies within about 1.0x of LRU (under 3%) on real traces. | Keep LRU. |
| **grouped decode and union loading** - Group tokens or sequences and load the union of experts once. | speculation_batching | any_pc | 7 | About 1.0x or worse vs pool plus LRU (0.4-1.0x at 8 GB); measured union gain 1.27-1.6x, not 2.2x. | Drop. |
| **partial expert read by activation sparsity** - Read only the active parts of an expert tensor (zone maps, late materialisation, column split). | layout_file | any_pc | 6 | Block-level skipping is lossy and fragments large sequential reads. | Gate on the sparsity measurement above. |
| **warm start expert cache** - Persist or seed the cache across prompts or turns. | cache_policy | any_pc | 5 | No gain over LRU in decode proxies; real prefill profile unmeasured. | Measure the real prefill profile. |
| **prefill accelerator offload** - Offload prefill GEMM to an external accelerator, NPU or FPGA. | hardware_tier | needs_special_hardware | 4 | TTFT about 1.0x vs the explicit-read path; the 69 GB pass at 2.4-3.1 GB/s is about 29 s, so the I/O floor remains. | Drop unless a batched prefill run proves prefill is compute-bound. |
| **activation sparsity compute skipping** - Skip dead FFN blocks or columns inside experts. | compute_kernel | any_pc | 4 | Cache is the bottleneck; 50-60% dead-block assumption is unmeasured. | Measure SwiGLU activation sparsity on Qwen3-30B first. |
| **residual delta coding of experts** - Encode experts as deltas to a reference expert or layer. | quantisation_bytes | any_pc | 4 | No byte-level similarity between layers found; 8.7x claim does not survive the LRU baseline. | Drop. |
| **speculative decoding with draft** - Draft tokens with a small model and verify in batch. | speculation_batching | any_pc | 4 | Verify reads the union of experts; loses at 8 GB (spec2 about 0.93x). | Revisit only with fewer experts per token. |
| **co-activation expert reordering** - Lay out experts by co-routing so related experts are adjacent. | layout_file | any_pc | 4 | About +5%; reduces read count, not bytes. | Drop. |
| **n-gram lookup speculation** - Speculate tokens from n-gram matches in context. | speculation_batching | any_pc | 4 | Reads grow with verified union; acceptance rate unmeasured here. | Test on a highly repetitive workload. |
| **multi-stream batching** - Batch several prompts or streams to amortise expert reads. | speculation_batching | any_pc | 3 | No reuse for single-token batching; cross-prompt expert overlap is small. | None for single-user use. |
| **sketch index and late materialisation** - Use a sketch to predict experts and read lazily. | prefetch_overlap | any_pc | 3 | Router scores already give exact top-10; sketch adds only a predictor. | Drop. |
| **prefill expert count reduction** - Use fewer experts only during prefill. | prefill_ttft | needs_special_hardware | 2 | Prefill reads most experts regardless; top-3 damages the KV cache. | Drop. |
| **two-machine layer split** - Split layers across two PCs as a pipeline. | distributed | needs_second_machine | 2 | Per-token latency sums sequentially; gives memory, not compute scaling. | Drop. |
| **cross-layer successor bundling** - Bundle next-layer partner experts. | layout_file | any_pc | 1 | No signal for partner pairs. | Drop. |