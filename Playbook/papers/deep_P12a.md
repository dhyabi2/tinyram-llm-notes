# deep_P12a: raw-text re-check of P-0255, 256, 257, 258, 260, 261, 262, 263, 264, 265

Counts: 10 rows, 10 raw text read, 0 unread. All raw text from original sources (PDFs via pdftotext, GitHub raw/API JSON, HTML stripped) in the scratch folder only. Ratios are computed by me from the quoted numbers.
Our regime for 'transfers': CPU-only, 8/16/32 GB RAM, experts streamed from NVMe, batch-1 decode, 125B-class 512-expert top-10 MoE.

---
## P-0255: Tutorial: KV cache reuse with llama-server (GitHub discussion #13606, llama.cpp), author smahs, 17 May 2025; maintainer ggerganov replies 19-20 May 2025; later comments to Aug 2026

- Source read: github.com/ggml-org/llama.cpp/discussions/13606 (raw HTML, tags stripped) (raw text read: True)
- Setup: Tutorial model gemma-2-2b-it-Q4_K_M.gguf, llama-server -c 1024 -np 2 (2 slots x 512 ctx). Hardware not stated. 43-token prompt. Later user comment (Aug 2026): Qwen3.6-35B-A3B UD-Q4_K_XL (a MoE), --cpu-moe, 8 threads, parallel 1, with a GPU offload setting, draft-mtp speculative decoding.
- Numbers:
  - Identical repeat of 43-token prompt: tokens re-evaluated: 1 (log: 'need to evaluate at least 1 token to generate logits', kv cache rm [42,end)); baseline: first request 43 tokens (kv cache rm [0,end)); ratio: 43x fewer prompt tokens; where: tutorial log blocks
  - Slightly modified request: tokens evaluated: 12 of 45 (kv cache rm [33,end)); baseline: 45; ratio: 3.75x fewer; where: Partial Context Reuse block
  - Slot save 72 tokens: 1.401 ms, 7,669,224 bytes (=106.5 KB/token, gemma-2-2b); baseline: -; ratio: -; where: Slot Persistence block
  - Slot restore 72 tokens: 0.739 ms; baseline: -; ratio: -; where: same
  - Aug 2026 user log, Qwen3.6-35B-A3B MoE: cache hit: 35-token prompt: cache_n 31, prompt_n 4, prompt_ms 111.464 (35.9 tok/s); baseline: no cache: prompt_n 35, prompt_ms 541.462 (64.6 tok/s, 15.47 ms/token); ratio: 4.86x less prompt time for 8.75x fewer tokens (per-pass fixed cost visible: 27.9 vs 15.5 ms/token); where: comment by felixendres, 27 Aug 2026
  - Same user: slot save 354 tokens: 67.055 ms, 67,912,240 bytes (=191.8 KB/token); restore 28.179 ms; baseline: -; ratio: -; where: same; after restart restore did not give cache hits (cache_n 0) - unresolved bug report
- Catalog / first-pass numbers:
  - Common-prefix check skips prefill for the matching part -> yes. Matches the logs (kv cache rm [n_past,end)). But slot selection is by similarity threshold -sps default 0.5, not a strict LCP rule; an LCP claim appears only in a Mar 2026 user comment ('picks a sufficient match based on LCP similarity rather than the optimal one').
  - Nothing there triggers on keystrokes -> yes. Correct; no trigger or streaming-input feature exists in the thread.
  - First pass: model/hardware not stated -> no. Model IS stated: gemma-2-2b-it-Q4_K_M.gguf (-c 1024 -np 2). Hardware is not stated.
  - First pass: slot save 72 tokens 1.4 ms, restore 0.74 ms -> yes. 1.401 ms / 0.739 ms
  - First pass: 43-token repeat re-evaluates 1 token; -sps default 0.5 -> yes. as above
- Transfers to our regime: yes. Mechanism (exact-prefix KV reuse on the CPU side) transfers fully: it is host-independent and costs no decode time. It does not trigger on typing, and does not change decode tok/s, which is the only thing our regime (NVMe-streamed experts, batch 1) is bound by. The MoE log shows prefill of 4 tokens still costs 111 ms (27.9 ms/token vs 15.5 ms/token for 35), i.e. a small partial pass is not proportionally cheap, which is consistent with our finding that chunk passes re-read expert bytes.
- Quality loss: None (exact reuse). Caveat from maintainer: SWA models (Gemma 2/3) need --swa-full for some of the techniques once context exceeds the window (1024 for Gemma).
- Ablation: none (tutorial, not a benchmark)
- Limitations stated: Author: -sps 'doesn't always behave predictably, inconsistent and unexpected slot switching'; --slots was flagged unsafe by ggerganov (19 May 2025; a Jan 2026 comment says it is now enabled by default). Cyclic N+1 prompts over N slots never hit (ExtReMLapin, 24 May 2025).
- Contradicts our findings: No. Supports F.md family 22: exact-prefix reuse exists; nothing triggers on keystrokes.

---
## P-0256: server : host-memory prompt caching (llama.cpp PR #16391), Georgi Gerganov; created 2025-10-02, merged 2025-10-09 into master (GitHub API: state closed, merged true, +809/-467, base master)

- Source read: api.github.com/repos/ggml-org/llama.cpp/pulls/16391 body + issues/16391/comments (raw text read: True)
- Setup: No benchmark. Description: host-RAM prompt cache acting as 'extra slots'; prefix similarity decides hot-swap into llama_context if it reduces processing. Limits: --cache-ram/-cram (MiB) and max cached tokens (default = --context-size). -cram -1 = no limit, 0 = disabled. Context checkpoints default raised 3 to 8. Works for dense, MoE, SWA, SSM per maintainer comment; incompatible with mtmd (multimodal) at this version.
- Numbers:
  - Example --cache-ram value in body: 8192 MiB (shown as 'use 8192 MiB of host RAM'); a later commenter calls 8 GB the default; baseline: -; ratio: -; where: PR body Usage block / dark-penguin comment
  - User log (AesSedai, 13 Oct 2025): KV size in RAM cache: prompt of 7,745 tokens = 2,783.45 MiB = 0.359 MiB/token; 10 prompts = 18,965 MiB under a 65,536 MiB limit; baseline: -; ratio: -; where: comment; model not named in log, GLM-4.6/R1 class on 2x3090 + 768 GB DDR5
  - User PP/TG on that rig: 150-200 t/s PP, 8-14 t/s TG; baseline: -; ratio: -; where: same comment
- Catalog / first-pass numbers:
  - Hot-swaps cached prompts into the context when they reduce processing -> yes. PR text: 'hot-swap them into the llama_context if it would reduce the processing'
  - Prefix reuse is established practice -> yes. Maintainer: 'game changer for agentic workflows'
  - First pass: --cache-ram default 8192 MiB -> partly. 8192 appears as the usage example in the PR body; the default is not stated there (a commenter asserts 8 GB). Not verified against the merged code.
  - First pass: no benchmarks/numbers -> yes. No benchmark; only anecdotes.
- Transfers to our regime: yes. CPU-side RAM mechanism, model independent. But its cost is host RAM: 0.36 MiB/token for the large model in the log, so a 4k-token conversation is ~1.4 GiB, which competes with the expert cache at 8/16 GB and the default 8 GiB limit would need lowering. Saves prefill only, never decode.
- Quality loss: None.
- Ablation: none
- Limitations stated: mtmd incompatibility (author's note); open requests for disk-backed cache (Mushoz, aviallon, 8-11 Nov 2025). Not benchmarked.
- Contradicts our findings: No.

---
## P-0257: Mini-SGLang chunked prefill with radix cache (mintlify docs page) - page gone; read the repo docs and code instead (sgl-project/mini-sglang, last commit 2026-05-17)

- Source read: mintlify URL returns 'Page gone' (read in raw HTML); raw.githubusercontent docs/features.md and python/minisgl/scheduler/prefill.py read (raw text read: True)
- Setup: Serving engine for Llama-3, Qwen-2.5, Qwen-3 incl. MoE; GPU (FlashAttention/FlashInfer/TRT-LLM attention backends, CUDA graphs, TP). No benchmark in the docs read.
- Numbers:
  - Chunked prefill default state: enabled by default; --max-prefill-length n; n very small (e.g. 128) 'not recommended as it may significantly degrade performance'; baseline: -; ratio: -; where: docs/features.md 'Chunked Prefill'
  - Code: extend_len: req.input_len - cached_len (radix match_req), chunk = min(token_budget, remain_len); baseline: -; ratio: -; where: scheduler/prefill.py _try_allocate_one/_add_one_req
- Catalog / first-pass numbers:
  - Chunked prefill with prefix match: only the uncached tail is computed -> yes. Confirmed by code (match_req gives cached_len; only input_ids[cached_len:] are chunked). The docs state the purpose of chunking as lowering peak memory/OOM, not the 'then the whole sequence is cached' wording.
  - First pass: page HTTP 410, unopened -> partly. Still gone (page says 'Page gone'); but the primary repo docs/code are readable, so the claim is now verified from source.
- Transfers to our regime: partly. Mechanism (prefix match then chunk the tail) is identical to P-0255; GPU-serving engine. The code has a TODO 'consider host cache match case'. Chunk size warning supports the cost of small chunks (we measured 4-6x disk bytes).
- Quality loss: None.
- Ablation: none
- Limitations stated: Docs say supports 'dense model architectures' (Qwen-3 incl. MoE); no numbers.
- Contradicts our findings: No.

---
## P-0258: (a) Tian Pan blog 10 Mar 2026 'LLM latency decomposition: TTFT and throughput' (Chinese edition) and (b) primary: 'Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve', Agrawal, Kedia, Panwar, Mohan, Kwatra, Gulavani, Tumanov, Ramjee, OSDI 2024

- Source read: blog HTML; arXiv 2403.02310 (submitted 4 Mar 2024, v1-v3, v3 17 Jun 2024) full PDF via pdftotext (raw text read: True)
- Setup: Mistral-7B on 1 A100-80GB; Yi-34B on 2 A100 (TP2); LLaMA2-70B on 8 A40 (TP4-PP2); Falcon-180B on 8 A100 (TP4-PP2, 100 Gbps ethernet). Datasets openchat_sharegpt4 and arxiv_summarization. Baselines vLLM and Orca. Token budgets 512 (strict SLO) and 2048 (relaxed). Metric: serving capacity (QPS) under P99 TBT SLO. Many concurrent requests (e.g. 128 requests).
- Numbers:
  - Capacity Mistral-7B: 2.6x; baseline: vLLM; ratio: 2.6; where: Abstract, Fig 10
  - Capacity Yi-34B (strict SLO, openchat_sharegpt4): 3.7x vs vLLM; 4.0x vs Orca; baseline: vLLM; ratio: 3.7; where: Sec 5.1 text, Fig 10
  - Falcon-180B end-to-end capacity: up to 5.6x (4.3x and 3.6x over vLLM TP-only/hybrid under strict SLO); baseline: vLLM/Orca; ratio: 5.6; where: abstract, Fig 13b
  - LLaMA2-70B: up to 6.3x vs Orca, 4.3x vs vLLM; baseline: -; ratio: -; where: Sec 5.1
  - Table 4 (Yi-34B, 2xA100, budget 1024, 128 req) P50 TTFT s / P99 TBT s, openchat: hybrid-only 0.53/0.68; chunked-only 1.04/0.17; Sarathi combined 0.76/0.14; baseline: hybrid-only; ratio: TTFT combined/hybrid 1.43x higher; chunked-only 1.96x higher; TBT 4.9x lower; where: Table 4
  - Table 4 arxiv_summarization P50 TTFT / P99 TBT: hybrid 3.78/1.38; chunked-only 5.38/0.20; combined 3.90/0.17; baseline: hybrid-only; ratio: TTFT 1.03x (combined), 1.42x (chunked-only); where: Table 4
  - Chunking overhead on prefill runtime (Yi-34B): at most ~25% at chunk 512; almost negligible at 2048; baseline: no chunking; ratio: <=1.25x; where: Sec 5.4.1, Fig 14
  - Tile quantisation: chunk 257 vs 256 raises prefill time 32%; baseline: -; ratio: 1.32; where: Sec 4.3
  - Blog's own claims: chunks 256-512, slightly higher average TTFT, capacity 2.6x/3.7x (quoted from paper); baseline: -; ratio: -; where: blog
- Catalog / first-pass numbers:
  - Chunked prefill raises throughput and lowers tail latency, slightly higher average TTFT -> yes. Confirmed by Table 4 (TTFT up 1.4x for combined on openchat) - 'slightly' understates: P50 TTFT 0.53 -> 0.76 s.
  - Does not claim a tok/s gain at decode -> yes. Gain is serving capacity/stall removal; no decode speed claim.
  - First pass: 2.6x Mistral-7B, 3.7x Yi-34B vs vLLM (blog quote) -> yes. Verified in the primary paper abstract.
  - First pass: primary paper not opened -> partly. Now opened: abstract, Table 4, Sec 5.4 read.
- Transfers to our regime: no. GPU serving, many concurrent requests, goal = stop prefill from stalling other users' decodes (generation stalls). Single user batch-1 CPU with NVMe-bound experts has no decode to protect; the paper itself says chunks are 'somewhat slower than full prefills' and raise TTFT. It supports our finding that chunking costs extra passes; on our regime each pass re-reads experts so cost is far above the 25% here.
- Quality loss: None (exact).
- Ablation: Yes: Table 4 separates hybrid batching vs chunked prefill vs both; Fig 14 chunk-size overhead.
- Limitations stated: Authors: smaller chunks reduce efficiency (arithmetic intensity, repeated KV reads); disaggregated prefill/decode compared only qualitatively (left to future work).
- Contradicts our findings: No; consistent with F.md note that chunking gives higher average TTFT.

---
## P-0260: TaxBreak: Unmasking the Hidden Costs of LLM Inference Through Overhead Decomposition, Prabhu Vellaisamy, Shreesh Tripathi, Vignesh Natarajan, Surya Santhan Thenarasu, Shawn Blanton, John P. Shen (CMU NCAL); IEEE ISPASS 2026

- Source read: arXiv 2603.12465v1 (12 Mar 2026), full PDF, 12 pp (raw text read: True)
- Setup: Single GPU. H100 platform: Xeon 8480C (56 cores 2.0/3.8 GHz) + H100 80GB DGX; H200 platform: Xeon Gold 6538Y+ 32 cores + H200 NVL 141GB. 6 CPU cores, 32 GB host RAM allocated. PyTorch 2.10, CUDA 12.6, Python 3.13, nsys. BF16, eager mode. Models Llama-3.2-1B/3B, GPT-2 124M, OLMoE-1B-7B (16 layers, 64 experts top-8, from memory for the structure), Qwen1.5-MoE-A2.7B. 50 warmup + 150 runs. Decode = m=10 output tokens.
- Numbers:
  - Kernel launches per token OLMoE-1B/7B (H100, BS=4, SL=2048, m=10): 9,305.3 (93,053 total); baseline: Llama-3.2-1B 847.5 (8,475 total); ratio: 10.98x; where: Table II
  - Qwen1.5-MoE-A2.7B launches/token: 6,695.1; baseline: Llama-1B 847.5; ratio: 7.90x (vs Llama-3B 1,536.9: 4.36x; OLMoE vs 3B: 6.05x); where: Table II
  - GPU utilisation: OLMoE 15.5%, Qwen-MoE 27.7%; baseline: Llama-1B 58.9%, Llama-3B 67.6%; ratio: 0.26x, 0.47x; where: Table II
  - Prefill (BS=1, SL=512, H100) kernels: OLMoE 13,741; Qwen-MoE 22,558; baseline: dense 850; ratio: 16.2x, 26.5x; where: Sec V.A
  - Prefill latency BS=1 SL=512 H100: OLMoE 260.5 ms; baseline: Llama-1B 22 ms; ratio: 11.7x; where: Sec V.A
  - Decode 10 tokens BS=1 SL=512: OLMoE 2157 ms; baseline: Llama-1B 188 ms; ratio: 11.5x; where: Sec V.A
  - Null-kernel launch floor: 4.707 us (H100), 4.503 us (H200) -> ~44 ms/token launch path for OLMoE; baseline: -; ratio: -; where: Table III, V.A
  - Faster host CPU effect: orchestration -10 to -29%; e2e up to -14% (11-14% in intro); Qwen-MoE on H200 8% slower device-active yet 13-14% faster e2e; baseline: H100 platform; ratio: -; where: Sec VI, conclusion
- Catalog / first-pass numbers:
  - OLMoE 9,305 vs 848 launches per token on H100 -> yes. 9,305.3 vs 847.5 (decode config BS=4/SL=2048, m=10) - ratio 10.98x
  - 8-11x more kernel launches per token than dense -> yes, with caveat. 8x = Qwen-MoE vs Llama-1B, 11x = OLMoE vs Llama-1B; vs Llama-3B the ratios are 4.4-6.1x. The paper's own wording is 'dense models of comparable active parameter count'.
  - That is a GPU result -> yes. All measurements GPU eager PyTorch; CUDA graphs not tested.
  - First pass: faster host CPU cuts orchestration 10-29% -> yes. Sec VI
  - First pass: launches/token measured in prefill -> absent. The 9,305 figure is a decode-config figure (m=10); prefill has 13,741 kernels for a 512-token forward pass, a different quantity.
- Transfers to our regime: no. GPU eager-mode PyTorch launch path (single-threaded Python+CUDA dispatch, ~4.7 us/kernel floor). Does not exist in llama.cpp CPU graph execution. The high kernel count comes from HF-style routing and per-expert loops, an implementation artefact; the paper itself says the main levers are reducing kernel count or amortising submission (e.g. CUDA graphs).
- Quality loss: n/a (diagnostic, no lossy method).
- Ablation: Cross-platform H100 vs H200 CPU comparison; eager vs FlashAttention-2 (Fig 9). No fusion/CUDA-graph ablation.
- Limitations stated: Authors: NVIDIA only, depends on CUDA tracing, replay-based attribution imperfect for dynamic/autotuned kernels; HDBI is a diagnostic ratio; effect weakens as HDBI rises above ~0.3.
- Contradicts our findings: No. It says nothing about CPU inference. It does say additional cores give no benefit for the dispatch path, which is a GPU single-thread issue.

---
## P-0261: Boosting MoE Training Throughput with Advanced Fusion Kernels, Rachit Garg, Matthew Nicely, NVIDIA Technical Blog, 15 Jun 2026

- Source read: developer.nvidia.com blog HTML (stripped) (raw text read: True)
- Setup: CuTe DSL fused GroupGEMM + GLU activation (SwiGLU/GeGLU/sReLU) + MXFP8/NVFP4 quantise/transpose kernels; sync-free MoE enabling full-iteration CUDA graphs. Microbenchmarks on GB200; end-to-end 'internal testing' on DeepSeek-V3 and GPT-OSS pre-training. Training, not inference.
- Numbers:
  - Kernel-level speedup: 1.3x-2x over unfused; baseline: unfused / Transformer Engine optimised kernels (Fig 3 baseline, chart only); ratio: -; where: summary + intro
  - Forward / backward: up to 1.3x / up to 2.1x; baseline: unfused; ratio: -; where: 'From kernel-level gains' section
  - End-to-end: up to 8% DeepSeek-V3; up to 93% GPT-OSS pre-training; baseline: pre-optimisation internal setup (unstated); ratio: 1.08x; 1.93x; where: summary; the 93% is attributed to enabling end-to-end CUDA graphs and communication overlap, not to the fusion alone
- Catalog / first-pass numbers:
  - About 15-20% for Mixtral-8x7B -> absent. Not on the page; NOT verified. Catalog row itself says it came from unopened sources.
  - Fused Triton kernel on OLMoE 0.999x vs eager -> absent. Not on the page; NOT verified.
  - First pass: kernel 1.3x-2x, e2e 8% and 93% -> yes. verified
- Transfers to our regime: no. GPU (GB200) training; fusion removes HBM round trips and CPU sync for GPU kernels. No CPU analogue; the 93% is a graph-capture/overlap effect in a setup where the baseline was CPU-bound.
- Quality loss: Not discussed for inference; MXFP8/NVFP4 training precision stated as 'minimal impact'.
- Ablation: none for CPU; Fig 3 per-activation speedups on GB200 (chart only).
- Limitations stated: 'internal testing', no baseline config given for the 93%, no error bars.
- Contradicts our findings: No.

---
## P-0262: ik_llama.cpp (Iwan Kawrakow), llama.cpp fork; catalog URL is the SourcePulse summary; read also README and PRs #229 and #488

- Source read: sourcepulse page; raw README.md (main); GitHub API for PR 229 and PR 488 bodies and PR 229 comments (none) (raw text read: True)
- Setup: PR 229 (merged 2025-02-23): new ggml op fusing MoE up, gate and activation; enabled with -fmoe. Tested on DeepSeek-Lite (16B total, 2.4B active) on CPU and an RTX 4080. PR 488 (merged 2025-06-05): dequantising GEMM for MoE with Trellis quants (IQ2_KT).
- Numbers:
  - PR 229 fused MoE ffn_up+gate+act on CPU: prompt processing: ~3-4% faster; baseline: unfused; ratio: 1.03-1.04x; where: PR 229 body
  - PR 229 on CPU: token generation: 1-2% faster; baseline: unfused; ratio: 1.01-1.02x; where: PR 229 body
  - PR 229 on CUDA (RTX 4080): ~26% PP, 7% TG; PP-512 4400 t/s vs 3450 t/s; baseline: mainline llama.cpp; ratio: 1.28x; where: PR 229 body
  - PR 488 faster CPU PP for Trellis quants + MoE: ~35% PP (DeepSeek-Lite with only ffn_up/gate as IQ2_KT); baseline: main branch; ratio: 1.35x; where: PR 488 body; a dequantise-GEMM kernel change, not dispatch
- Catalog / first-pass numbers:
  - Fused MoE ops and CPU-optimised quantised GEMM already exist in a llama.cpp fork -> yes. PR 229 (fused ffn) and PRs 482/488/515/531 (CPU GEMM)
  - No speedup figures are given -> no (for the sourcepulse page: correct; overall: wrong). The PR bodies give figures: fused MoE on CPU only +3-4% PP and +1-2% TG on DeepSeek-Lite; +26%/+7% on CUDA. First pass said 'no numbers; wiki not opened' - the PRs contain them.
  - First pass: README says last synced with upstream Aug 2024 -> absent. README: forked June 2024, last synced Aug 2024 (read now)
- Transfers to our regime: yes. CPU, llama.cpp lineage. It is the only direct CPU datapoint for 'fuse the MoE FFN ops': a few percent. PR text cites the removable costs as per-op expert search (shared by 3 ops), repeated quantisation of the shared input, and a barrier after each op - matches the 'op-boundary/barrier' cost class, and measured small. Hardware for CPU numbers is not stated in the PR.
- Quality loss: None for fusion; -fmoe cannot be used when computing an imatrix.
- Ablation: Yes by construction (fused vs unfused op, one switch), CPU and CUDA.
- Limitations stated: Author: Metal missing, so off by default; model only DeepSeek-Lite; CPU model not stated; README says AVX2+/NEON CPU and CUDA are the only fully performant backends.
- Contradicts our findings: Supports F.md (dispatch/fusion worth ~1-7% of decode). F.md's 'ik_llama.cpp ... give no comparable number' should be corrected: PR 229 gives 3-4% PP / 1-2% TG on CPU.

---
## P-0263: KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models, Hongtao Chen, Weiyu Xie, Boxin Zhang, Jingqi Tang, Jiahao Wang, Jianwei Dong, Shaoyuan Chen, Ziwei Yuan, Chen Lin, Chengyu Qiu, Yuening Zhu, Qingliang Ou, Jiaqi Liao, Xianglin Chen, Zhiyuan Ai, Yongwei Wu, Mingxing Zhang; SOSP '25, Seoul, 13-16 Oct 2025, DOI 10.1145/3731569.3764843

- Source read: full 16-page PDF from madsys.cs.tsinghua.edu.cn (SOSP25-chen.pdf); not on arXiv under that name as far as checked (raw text read: True)
- Setup: Dual-socket Xeon Platinum 8452Y (36 cores/socket), 1 TB DDR5, MLC bandwidth 220 GB/s intra-socket, 125 GB/s cross-socket. A100-40GB (BF16/FP16 runs) and RTX 4080-16GB (quantised), PCIe 4.0. Models DS-3 671B (58 MoE layers, 256 experts, top-8), DS-2 236B (59, 160, top-6), QW-2 57B (28, 64, top-8). Batch 1. Prefill 32-8192 token Wikitext prompts; decode 32-token prompt, 512 output. Baselines Fiddler (PyTorch) and llama.cpp that the authors extended with custom expert-level offload code.
- Numbers:
  - Prefill speedup: 4.62-19.74x 'over existing systems'; baseline: Fiddler and llama.cpp (per-baseline values are in Fig 11 chart only); ratio: -; where: Abstract, contributions
  - Decode speedup BF16/FP16, no Expert Deferral: 2.42-4.09x over Fiddler; 1.25-1.76x over llama.cpp; baseline: Fiddler; llama.cpp (FP16); ratio: -; where: Sec 6.2
  - Decode quantised: 1.77-1.93x over llama.cpp; baseline: llama.cpp; ratio: -; where: Sec 6.2
  - With Expert Deferral: additional up to 1.45x; overall 1.66-2.56x over llama.cpp (4.90x overall in contributions); baseline: llama.cpp; ratio: -; where: Sec 6.3, contributions
  - AMX CPU MoE kernel: 21.3 TFLOPS single socket (DS-3), 3.98x over oneDNN PyTorch; PyTorch AMX peaks 5.4 TFLOPS vs AVX-512 1.8, ~7% of 73.7 TFLOPS peak; baseline: PyTorch; ratio: 3.98; where: Sec 2.2/3.2, Fig 3
  - Breakdown prefill 8192 tokens (vs Fiddler): AMX kernel up to 3.14x; AVX-512 kernel is slower than baseline in prefill; dynamic scheduling up to 1.83x; NUMA TP up to 1.22x; CUDA graph slight; baseline: Fiddler; ratio: -; where: Sec 6.4, Fig 14a
  - Breakdown decode: AVX-512 kernel up to 2.22x; NUMA TP up to 1.63x; CUDA graph up to 1.23x; baseline: Fiddler; ratio: -; where: Sec 6.4, Fig 14b
  - Launch overhead on GPU: Fiddler >7,000 launches/token at ~16 us; llama.cpp ~3,000 at ~5 us; GPU invocation overhead >20%; baseline: -; ratio: -; where: Fig 4, Sec 2.3
  - Quality with Expert Deferral (quantised, Table 2): DS-3 HumanEval 83.0 -> 83.0, MBPP 71.2 -> 70.2, GSM8K 94.8 -> 95.2, StrategyQA 83.0 -> 82.9; DS-2 80.5->82.5, 67.6->66.8, 93.3->92.8, 79.7->80.4; QW-2 65.7->67.4, 52.4->53.4, 84.7->83.4, 83.6->82.5; baseline: (8+0); ratio: <=2 points; where: Table 2
- Catalog / first-pass numbers:
  - 4.62-19.74x prefill speedups -> yes (figure), baseline wrong. The range is over both Fiddler and llama.cpp; the catalog 'over a llama.cpp baseline' / first pass 'baseline not named on page' - the paper names Fiddler and an author-extended llama.cpp. Per-baseline split only in Fig 11 (chart).
  - Decode 1.25-4.09x -> yes. 1.25-1.76x is vs llama.cpp, 2.42-4.09x vs Fiddler; the bottom end is the llama.cpp ratio.
  - Gain from kernel arithmetic efficiency and hardware -> yes. AMX tiling-aware layout + kernels, dynamic scheduling, CUDA graph, NUMA TP; fused-op step is merged into the kernel step and is not isolated.
  - First pass: AMX CPU + GPU; transfers partly -> yes. also needs a GPU for attention/shared experts; 1 TB DDR5 and 220 GB/s memory.
- Transfers to our regime: no. Needs Intel AMX server CPU with >200 GB/s DRAM and a GPU, all experts resident in RAM (no NVMe streaming), 1 TB RAM. Prefill gains are arithmetic-efficiency (AMX 3.14x, dynamic work balancing 1.83x), decode gains are CPU/GPU overlap and CUDA graphs. Not dispatch removal. One transferable observation: dynamic scheduling matters because expert load is imbalanced in prefill.
- Quality loss: Only Expert Deferral is lossy: changes in the table above, all within 2 points; 'average accuracy drop no more than 0.5%' (abstract). Core kernels exact (full-accuracy).
- Ablation: Yes, Fig 14 (cumulative, vs Fiddler); fusion not isolated from kernel swap.
- Limitations stated: Not stated as a section; evaluation covers only batch 1 on one machine; the extended llama.cpp baseline is the authors' variant.
- Contradicts our findings: No. Its measured CUDA-graph gain for decode (1.23x) and slight effect for prefill matches the picture that launch removal helps GPU decode, not prefill.

---
## P-0264: MobileMoE: Scaling On-Device Mixture of Experts, Yanbei Chen, Hanxian Huang, Ernie Chang, Jacob Szwejbka, Digant Desai, Zechun Liu, Vikas Chandra, Raghuraman Krishnamoorthi (Meta AI)

- Source read: arXiv 2605.27358v1 (26 May 2026), full PDF text (raw text read: True)
- Setup: MobileMoE-S/M/L: 0.3/0.5/0.9B active (S = 272M active, 1.26B total, 60 routed experts top-4 + 1 shared), INT4 QAT, symmetric group 32, INT8 dynamic activations. Samsung Galaxy S25 (Snapdragon 8 Elite, 4 CPU threads), iPhone 16 Pro (A18 Pro, 2 CPU threads; also Metal GPU via MLX), ExecuTorch + XNNPACK, batch 1, real prompts, 3 runs. Baseline: dense MobileLLM-Pro (0.55 GB INT4). All weights resident in RAM.
- Numbers:
  - S25 prefill TTFT 256 / 512 / 8k tokens (s): MoE-S 0.44 / 0.90 / 50.33; baseline: MobileLLM-Pro 0.78 / 1.73 / 105.21; ratio: 1.77x / 1.92x / 2.09x; where: Table 7
  - S25 decode tok/s 256 / 512 / 8k: MoE-S 138.1 / 130.0 / 25.8; baseline: 61.3 / 56.5 / 10.6; ratio: 2.25x / 2.30x / 2.43x; where: Table 7
  - iPhone CPU prefill 256 / 8k (s): 0.46 / 39.91; baseline: 1.29 / 122.42; ratio: 2.80x / 3.07x; where: Table 7
  - iPhone CPU decode 256 / 8k: 204.6 / 32.2; baseline: 61.0 / 10.6; ratio: 3.35x / 3.04x; where: Table 7
  - iPhone GPU (MLX) prefill 512/1k/2k: 0.16/0.32/0.68; baseline: 0.58/1.20/2.49; ratio: 3.6x/3.8x/3.7x; where: Table 8
  - Accuracy (14 benchmarks, INT4 QAT avg): MoE-S 44.0, M 52.5, L 57.8; baseline: MobileLLM-Pro 45.5; ratio: -1.5 / +7.0 / +12.3 points; where: Table 6/7
  - QAT loss vs BF16 SFT: S 46.7 -> 44.0 (-2.7), M 55.3 -> 52.5, L 60.1 -> 57.8; baseline: BF16; ratio: -2.3 to -2.8; where: Sec 4.3.1
  - Peak RSS S25 real vs dummy prompts: MoE-S 0.93-1.49 GB real vs 0.63-1.11 GB dummy; L 3.66-4.71 vs 1.72-3.41; baseline: dense 0.90-1.91 both; ratio: 1.2-2.1x real/dummy; where: Table 9
- Catalog / first-pass numbers:
  - Fuses the whole expert FFN into one CPU op to amortise launch overhead -> yes. Sec 4.3.2: top-k select, dispatch, gate+up GEMM per expert, SwiGLU, down, weighted scatter in one ExecuTorch op
  - No llama.cpp comparison number -> yes. llama.cpp is not mentioned anywhere in the paper.
  - First pass: S prefill 1.8-3.8x, decode 2.2-3.4x -> yes. 1.8-2.2x/2.2-2.6x on S25, 2.7-3.1x/2.8-3.4x on iPhone CPU, up to 3.8x prefill on iPhone GPU. First pass: 'read first 100k of 162k chars' - now complete.
  - First pass: speedup is vs a different dense model not fused vs unfused -> yes. confirmed; whole-paper search finds no fused-vs-unfused ablation (ablations are scaling-law ablations on H100).
- Transfers to our regime: partly. Shows a fused CPU MoE op exists in a mobile stack, but the speedups come from 3-4x fewer active parameters than the dense baseline (the paper's own explanation: smaller active FLOPs and per-token bandwidth), with all weights in RAM. Says nothing about the dispatch share, nothing on streaming from storage, and the models are trained from scratch at sub-1B size.
- Quality loss: No lossy inference trick other than INT4 QAT (-2.3 to -2.8 average points).
- Ablation: none for fusion.
- Limitations stated: No stated limitations section found; custom kernel only for XNNPACK path.
- Contradicts our findings: No. One mild point: memory under real prompts is 1.2-2.1x dummy-prompt memory for the MoE, i.e. expert footprint depends on input - same direction as our trace-based footprint-per-window measurements.

---
## P-0265: ASAP: A DiSaggregated and Asynchronous Inference System for MoE Prefill, Weiwei Chen, Shuang Chen, Lele Li, Qiang Hu, Han Li, Xin Ye, Ming Yan, Zhibin Yu (Huawei)

- Source read: arXiv 2606.22541v1 (21 Jun 2026), full PDF (15 pp) (raw text read: True)
- Setup: CloudMatrix384 supernode of Ascend 910 NPUs (each die 24 cube cores, 48 vector cores, 64 GB HBM, 400 GB/s bidirectional UB links). DeepSeek-V3.2 (61 layers). Baseline Default synchronous deployment DP8-TP4-EP32 on 32 dies; ASAP DP4-TP4-EP16 (same device count); also ChunkedPrefill baseline. Poisson arrivals, Huawei Cloud production trace lengths <=32k.
- Numbers:
  - SLO-compliant throughput: 20 RPS; baseline: Default 6.8 RPS; ChunkedPrefill 10.5 RPS; ratio: 2.94x (+194%); 1.90x (+90%); where: Sec 5.2 (text)
  - Mean TTFT at RPS=1: 350 ms; baseline: Default 533 ms; ChunkedPrefill 388 ms; ratio: 0.66x (-34.3%); 0.90x (-9.8%); where: Sec 5.2
  - Host kernel dispatch latency per layer: 220 us measured on CloudMatrix384; x61 layers = ~13.4 ms TTFT saved by bubble-free dispatching; baseline: per-layer kernels; ratio: 13.4 ms / 350 ms = 3.8% of TTFT; where: Sec 5.5.3
  - Throughput from bubble-free dispatching (MoE Super Kernel) alone: +6% SLO-compliant throughput; baseline: without super kernel; ratio: 1.06x; where: Sec 5.5.3
  - Triple-stream comm overlap: 17.8 -> 20 RPS (+12.4%); baseline: without overlap; ratio: 1.12x; where: Sec 5.5.2
  - Dispatch overhead statement: 'tens to hundreds of microseconds' CPU dispatch overhead is on the critical path since MoE kernels last sub-ms to a few ms; baseline: -; ratio: -; where: Sec 3.4.2
- Catalog / first-pass numbers:
  - Dispatch overhead of tens to hundreds of microseconds per layer is on the critical path on NPUs -> yes. Sec 3.4.2 states exactly this; measured 220 us per layer in Sec 5.5.3.
  - A layer-oblivious fused super-kernel enables ahead-of-time dispatch -> yes. 'MoE Super Kernel' made layer-oblivious (global pointers to all L layers' weights, layer ID as device-side argument) 'allowing ahead-of-time kernel dispatching'.
  - First pass: abstract says nothing about per-layer dispatch / super-kernel; UNVERIFIED / likely wrong -> no. First pass was WRONG: the full text contains both claims; it only read the abstract page. Abstract alone is indeed about async attention/MoE disaggregation.
  - First pass: +90% over synchronous baselines -> partly. +90% is vs ChunkedPrefill (20 vs 10.5 RPS); vs Default it is +194% (20 vs 6.8).
- Transfers to our regime: no. Datacentre NPU with 32-NPU expert parallelism and a host-launch model; the dispatch bubble (220 us x 61 layers = 13.4 ms) matters because TTFT is ~350 ms. It contributes +6% of throughput, i.e. it is the smallest of ASAP's four optimisations, and even on this NPU it is a few percent. In our regime a layer step costs hundreds of ms of disk time, so 220 us is invisible.
- Quality loss: None (scheduling/kernel change).
- Ablation: Yes: Sec 5.5: triple-stream overlap +12.4%, bubble-free dispatch +6% and 13 ms TTFT; other components in 5.5.1 (not itemised here).
- Limitations stated: Not stated in a dedicated section in the text read; evaluated only on CloudMatrix384 and DeepSeek-V3.2.
- Contradicts our findings: No; it is a second published point (NPU host launch 220 us per layer, +6% throughput) that per-layer dispatch removal is worth single-digit percent.

---
## Family 22: prefill while the user types

Prior art read in this pass: llama.cpp discussion 13606 and PR 16391 (exact-prefix reuse, host-RAM prompt cache), Mini-SGLang repo code (radix match then chunk), Sarathi-Serve primary paper. A web search for an arXiv paper on prefilling a KV cache incrementally while the user types found none; hits were the Anthropic cookbook 'speculative prompt caching' (warms cache with a 1-token request while the user types, fixed context), an OpenAI community request for a prefill-only call, and a Technical Disclosure Commons note about shadow requests on stable partial input (search snippets only, not opened). From the first pass, not re-read here: Stream2LLM (arXiv 2604.16395), Layered prefill (arXiv 2510.08055).
What is new in our variant: nothing in the mechanism (exact-prefix KV reuse and chunked prefill are shipped, and the cookbook warms the cache during typing). What may be specific to us is the measured cost side: chunk passes on an NVMe-streamed MoE re-read experts (4-6x disk bytes in F.md), which the GPU papers do not model. Sarathi-Serve Table 4 confirms chunking raises P50 TTFT (0.53 to 0.76 s combined, 1.04 s chunked-only) and costs up to ~25% prefill time at chunk 512 even on GPU. Closest read in full: Sarathi-Serve.

## Family 23: ahead-of-time kernels to remove dispatch overhead

Read in full this pass: TaxBreak, MobileMoE, ASAP, KTransformers, NVIDIA blog, ik_llama.cpp PRs 229 and 488. Search found no direct fused-vs-unfused CPU expert-FFN benchmark beyond ik_llama.cpp PR 229 (llama.cpp CUDA write-ups, snippet only: fusion PRs each under 10% TG). Datapoints on the size of the dispatch lever: CPU +3-4% PP, +1-2% TG (PR 229); NPU +6% throughput, 13.4 ms of TTFT (ASAP); GPU decode CUDA graphs 1.23x, prefill slight (KTransformers); fused GPU kernels up to 1.3x/2.1x kernel level (NVIDIA, training).
What is new in our variant: nothing established; fusion is shipped in ik_llama.cpp and the CPU datapoint is a few percent. Closest: ik_llama.cpp PR 229 and ASAP's super kernel.

## Corrections to the first pass and consequences

- P-0255: model is stated (gemma-2-2b-it-Q4_K_M); an Aug 2026 MoE log (Qwen3.6-35B-A3B) adds a prefill datapoint: 35 tokens 541 ms vs 4 tokens 111 ms.
- P-0256: 8192 MiB is a usage example in the PR body, not a verified default.
- P-0257: the cited docs page is gone, but repo docs and code verify the prefix-match-then-chunk mechanism.
- P-0258: primary paper now read; Table 4 and chunk overhead numbers added.
- P-0260: the 9,305 figure is a decode-config number; vs the 3B dense model the ratio is 4.4-6.1x not 8-11x; prefill kernel counts are 13,741/22,558 vs 850.
- P-0262: first pass 'no numbers' is wrong; PR 229 gives CPU +3-4% PP, +1-2% TG.
- P-0263: baselines are Fiddler and an author-extended llama.cpp; the 4.62-19.74x range is over both; llama.cpp-only decode ratios are 1.25-1.76x (1.77-1.93x quantised).
- P-0265: first pass was wrong; the full text does contain the tens-to-hundreds of microseconds claim and the layer-oblivious super kernel (220 us per layer, +6%); +90% is vs ChunkedPrefill (+194% vs Default).
- P-0261: 15-20% Mixtral and 0.999x OLMoE remain unverified (not on the page).
Effect on our beliefs: no contradiction. F.md's dispatch share (1-7% of decode, 60% needed) is now supported by two published datapoints (ik_llama.cpp CPU +1-4%, ASAP +6%); F.md's statement that ik_llama.cpp gives no comparable number should be corrected.