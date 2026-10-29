# Deep re-check, batch P11a (raw full text; rows P-0233, P-0234, P-0235, P-0237, P-0238, P-0239, P-0240, P-0241, P-0242 of P11)

Work folder (raw PDFs/text/HTML/JSON): <scratch> All 9 rows read from raw text; 0 unread. Remembered facts are labelled 'from memory'. Ratios are computed by me from the table values quoted.


## P-0233: Run distributed inference with llama.cpp on Arm-based AWS Graviton4 instances (Aryan Bhusari, Joe Stech; Arm Learning Paths, last updated 09 Sep 2026; page is a 5-step tutorial: Introduction, Convert, Configure workers, Configure master, Next steps)

- Read: learn.arm.com/learning-paths/servers-and-cloud-computing/distributed-inference-with-llama-cpp/ (landing page) and sub-pages how-to-1, how-to-2, how-to-3, fetched as raw HTML and stripped; the catalog URL is truncated at 'distributed-inference-wit'; raw text read: True
- Setup: 3x AWS c8g.4xlarge (Graviton4, 1 master + 2 RPC workers; page does not state vCPU/RAM, but the log shows 'RPC[...] - 31491 MiB free' per worker), model Llama-3.1-70B Q4_0 GGUF (37.22 GiB, 4.53 BPW, 80 layers, dense, not MoE), build 6209 (fb22dd07), workers started with 'rpc-server -c -p 50052 -H 0.0.0.0 -t 64', master 'llama-cli -n 128 --rpc $worker_ips -ngl 999'. Network type/bandwidth: not stated (only 'TCP connections between nodes'). Storage: 500 GB EBS (prerequisite). Batch 1, 10-token prompt, 63 generated tokens.
- Numbers:
  - Generation speed in the page's example log: eval time 36101.93 ms / 63 runs = 573.05 ms/token = 1.75 tok/s | baseline: none given on page | ratio: n/a | where: how-to-3 (Configure the master node), llama_perf_context_print
  - Prompt eval: 1854.82 ms / 10 tokens = 5.39 tok/s | baseline: none | ratio: n/a | where: how-to-3
  - Model load time: 489542.78 ms (about 8.2 min); page says inference run 'will take ~10 minutes' | baseline: none | ratio: n/a | where: how-to-3
  - Network bandwidth / minimum link: not stated anywhere on the 4 pages | baseline: n/a | ratio: n/a | where: all pages (grep for Gbps, bandwidth, 10G: nothing)
- Catalog / first-pass numbers checked:
  - guides report 2-5 tok/s for 70B on CPU clusters: no; this page's own log shows 1.75 tok/s (below the 2-5 range); the first pass said 'no tok/s on the page', which is also wrong: the number is on the 'Configure the master node' sub-page, not on the landing page
  - 10GbE as the minimum: absent; no network figure on any page of this learning path; the 10GbE claim must come from other guides
  - tensors held in worker RAM over the network (rpc-server holds weights in remote RAM): yes; 'RPC offloads both the model and the computation over TCP'; master physically hosts the GGUF and ships tensors to workers (-ngl 999); '-c' enables a local file cache on workers
- Transfers to our regime: no. Why: Dense 70B layer sharding over RPC, not an expert store. All weights are resident in the workers' RAM; nothing is streamed per token; only activations cross the wire. Our regime fetches 10 of 512 experts per layer per token from NVMe into a 8/16/32 GB RAM cache, which is the opposite traffic pattern. The one useful datum is that stock llama.cpp RPC on three 16-vCPU Graviton4 boxes reaches only 1.75 tok/s for a 37 GiB dense model.
- Quality loss: none discussed (Q4_0 quantisation only; no benchmark)
- Ablation: none
- Limitations (authors): Tutorial; authors state RPC is insecure by default and supports at most 16 backend workers; the page text flags an 'AI-assisted summary'. No measurement methodology (single run, 63 tokens).
- Contradicts our findings: No contradiction. pubtech F.md line 28 cites 'RPC guides report 2 to 5 tok/s with 10GbE as minimum'; this specific guide gives 1.75 tok/s and no link figure, so it cannot be the source of that sentence.

## P-0234: MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism (Ruidong Zhu, Ziheng Jiang, Chao Jin, Peng Wu, Cesar A. Stuardo, Dongyang Wang, Xinlei Zhang, Huaping Zhou, Haoran Wei, Yang Cheng, Jianzhe Xiao, Xinyi Zhang, Lingjun Liu, Haibin Lin, Li-Wen Chang, Jianxi Ye, Xiao Yu, Xuanzhe Liu, Xin Jin, Xin Liu; ByteDance Seed + Peking University; arXiv v4 26 Jul 2025, cs.DC)

- Read: arXiv:2504.02263v4 (PDF, full text); raw text read: True
- Setup: Models (Table 4, bf16 weights/activations/KV): Mixtral-8x22B (56 layers, 8 experts, top-2, 141B), DBRX (40 layers, 16 experts, top-4, 132B), Scaled-MoE (48 layers, 32 experts, top-4, 317B). Cluster 1: 8 nodes x 8 NVIDIA 80 GB Ampere GPUs, 128 CPUs, 2 TB host RAM, 8x200 Gbps InfiniBand, 400 GB/s NVLink. Cluster 2 (heterogeneous): H20 nodes (900 GB/s NVLink, 4x400 Gbps NICs) + L40S nodes (PCIe, 2x400 Gbps). Workload: production trace, median input 571 / output 159 tokens, TBT limit 150 ms. Baselines vLLM and TensorRT-LLM (min 8 GPUs for Mixtral/DBRX, multi-node for Scaled-MoE). Decode throughput per GPU is the metric; batches are large (aggregated across many attention replicas; roofline discussion uses batch 156 per expert for an A100).
- Numbers:
  - Decode throughput per GPU, Scaled-MoE, homogeneous Ampere: MegaScale-Infer 1.90x vs TensorRT-LLM and 7.11x vs vLLM (absolute tok/s only in Fig 8a bars, not in text) | baseline: TensorRT-LLM / vLLM | ratio: 1.90x / 7.11x (as stated, figure values not tabulated) | where: Sec 7.2 p12, Fig 8a
  - End-to-end throughput incl. prefill (homogeneous): up to 1.18x | baseline: best baseline | ratio: 1.18x | where: Sec 7.2 p12, Fig 8c
  - Abstract headline 'up to 1.90x higher per-GPU throughput': 1.90x | baseline: TensorRT-LLM, Scaled-MoE decode only | ratio: 1.90x (the figure is the best case, not the mean) | where: Abstract; Sec 7.2
  - Heterogeneous H20+L40S, per-cost decode: 1.86x per-cost decode (Intro), 1.80x per unit power decode / 1.72x e2e per unit power | baseline: best baseline | ratio: 1.86x / 1.80x / 1.72x | where: Intro p2; Sec 7.2 Fig 9-10
  - Ablation: disaggregation alone with NCCL: up to 4.66x over colocated (vLLM-like) | baseline: colocated vLLM | ratio: 4.66x | where: Sec 7.4 Fig 13
  - Ablation: optimised M2N lib + ping-pong on top: additional up to 1.53x | baseline: Dist.+NCCL | ratio: 1.53x | where: Sec 7.4 Fig 13
  - Ablation: ping-pong micro-batches m=1 to 2 / m=3: 1.9x; then 1.10x (Mixtral), 1.28x (DBRX), 1.38x (Scaled-MoE) | baseline: m=1 / m=2 | ratio: as stated | where: Sec 7.4 Fig 14
  - M2N vs NCCL micro-benchmark: median latency -80.8%, P99 -96.2%, throughput up to 9.9x (256 KB: -68.2% median, -92.9% tail, 4.2x) | baseline: NCCL | ratio: as stated | where: Sec 7.3 Fig 11
  - Production: cost of serving same traffic 1.5-2.0x lower (heterogeneous), ~10,000 GPUs | baseline: n/a | ratio: 1.5-2.0x | where: Sec 8
- Catalog / first-pass numbers checked:
  - Experts on separate GPUs over RDMA in datacentres: yes, with a correction; attention and FFN/expert modules are disaggregated onto different node groups (attention replicas with DP, experts with EP), tokens shipped M-to-N over RDMA (own M2N library on RDMA write-with-immediate); 200-400 Gbps NICs
  - first-pass: up to 1.90x per-GPU throughput: yes; 1.90x is vs TensorRT-LLM on Scaled-MoE decode; vs vLLM it is 7.11x; end-to-end only 1.18x
  - first-pass: hardware unverified / not on abstract page: wrong in the sense it is now known; 8-GPU Ampere 80 GB nodes with 200 Gbps IB, plus H20/L40S heterogeneous cluster
  - first-pass: 'disaggregates attention and FFN, not experts per se': yes; abstract: 'disaggregates attention and FFN modules within each model layer'
- Transfers to our regime: no. Why: Datacentre GPUs, bf16, decode batches aggregated into the hundreds per expert so FFN becomes compute-bound; the whole point is to raise batch per expert. Our regime is batch 1 where each expert sees one token, nothing to aggregate, CPU-only, experts on NVMe not on remote GPUs. Per-layer cross-node communication also adds latency (paper: TBT only 'comparable' to baselines). The one portable idea (separate the memory-bound expert store from attention) is the conceptual cousin of an expert store on another host, but the paper itself needs batch to make it pay.
- Quality loss: none (lossless serving system)
- Ablation: Yes (Sec 7.4): disaggregation alone 4.66x over colocated; M2N library 1.53x more; ping-pong m=1->2 1.9x. All at large batch on GPUs.
- Limitations (authors): Authors note communication overhead per layer, instability at high percentiles with NCCL, need for micro-batches >=3 for comm overlap; expert-load imbalance handled with on-device redundancy (Sec 8). No limitation section about small batch.
- Contradicts our findings: No. pubtech F.md calls it a datacentre GPU/RDMA system not comparable; confirmed from raw text.

## P-0235: llama.cpp issue #19745: 'llama-server and llama-cli hang/crash during RPC tensor upload for large models, while llama-bench succeeds' (opened by visorcraft 2026-02-19; closed as not planned 2026-04-07 by stale bot)

- Read: GitHub REST API JSON for issue 19745 and its comments (body + 2 comments), read raw; raw text read: True
- Setup: 2x AMD Ryzen AI Max+ 395 (Strix Halo, gfx1151, UMA), 128 GB LPDDR5X each, Fedora 43, kernel 6.18.10, GCC 15.2.1, ROCm 6.4.2 / 7.2 / 7.0 nightlies, llama.cpp b8102 (39e4b1dc), -DGGML_HIP=ON -DGGML_RPC=ON. Link USB4/Thunderbolt direct, ~9.4 Gbps measured. Models: Qwen3-Coder-Next-Q4_K_M ~46 GiB (works in bench/server/cli), MiniMax-M2.5-REAP-139B-A10B-Q8_0 137.77 GiB (139.15 B params), Qwen3.5-397B-A17B-UD-Q4_K_XL ~205 GiB. Split -ts 1/1: ROCm0 68285 MiB local + RPC0 72173 MiB remote, -ngl 99 -fa on --ctx-size 4096.
- Numbers:
  - llama-bench, 138 GiB model over RPC, no -dio: pp64 117.15 +/- 10.13 tok/s; tg16 14.75 +/- 0.09 tok/s; run took 218 s | baseline: n/a | ratio: n/a | where: issue body, llama-bench section
  - llama-server with -dio, ROCm 6.4.2 host build: load ~196 s; tg 15.85 tok/s; pp 75.17 tok/s | baseline: without -dio: hangs forever (30+ min, /health 503) | ratio: n/a | where: issue body, -dio matrix
  - with -dio, ROCm 7.2 container: load ~190 s; tg 15.88; pp 49.05 | baseline: hang without | ratio: n/a | where: same matrix
  - with -dio, ROCm 7.0 nightlies: load ~191 s; tg 15.83; pp 71.71 | baseline: hang without | ratio: n/a | where: same matrix
  - Effective load rate: ~138 GiB in ~190-196 s = 0.70-0.73 GiB/s (~5.9 Gbit/s) for the whole split; remote half 72173 MiB in ~190 s = ~0.37 GiB/s (my arithmetic) | baseline: link ~9.4 Gbps = 1.17 GB/s | ratio: my arithmetic: ~0.3-0.6 of line rate | where: derived from the matrix
- Catalog / first-pass numbers checked:
  - Over 100 GiB across two hosts, llama-server hangs at load_tensors: yes; hangs at 'load_tensors' for the 138 GiB and 205 GiB models; the author's own claim is '>~100 GiB total across the RPC split'; llama-cli instead crashes with an HSA/HIP fault after ~309 s
  - -dio fixes it: yes; -dio (direct IO) resolves it on all three ROCm versions; --no-mmap and --no-warmup alone do not
  - first-pass: tg ~15.8-15.9 tok/s (GPU hosts): yes; 15.83-15.88 with -dio; 14.75 in bench without
  - first-pass: load ~190-196 s: yes; 190, 191, 196 s
- Transfers to our regime: no. Why: Only as a fragility warning: it shows loading of very large models over llama.cpp RPC is broken on a particular UMA GPU path (mmap + HIP). Throughput numbers are GPU (ROCm, Strix Halo) on a 10B-active MoE at Q8_0 with all weights resident after load; no per-token weight transfer, so nothing about streaming experts from NVMe on a CPU-only host.
- Quality loss: none discussed
- Ablation: A testing matrix of ROCm versions x -dio; no ablation of network or storage.
- Limitations (authors): The author admits the root cause is a hypothesis; a maintainer (pwilkin) told the author not to post LLM-generated reports and asked for raw crash dumps; issue closed as 'not planned' after 14 days of inactivity. Single-person report, UMA-specific.
- Contradicts our findings: No.

## P-0237: Delta Decompression for MoE-based LLMs Compression (D2-MoE; Hao Gu*, Wei Li*, Lujun Li*, Qiyuan Zhu, Mark Lee, Shengjie Sun, Wei Xue, Yike Guo; HKUST / Univ. of Birmingham / AISpeech; arXiv v1 24 Feb 2025, cs.LG)

- Read: arXiv:2502.17298v1 (PDF, full text). The text carries no venue statement ('ICML 2025' comes from the catalog; the only ICML mention in the PDF is a reference).; raw text read: True
- Setup: Models: Mixtral-8x7B (8 experts), DeepSeekMoE-16B-Base (fine-grained + shared expert), Phi-3.5-MoE (16 experts), Qwen2-57B-A14B. Experiments on NVIDIA A100 GPUs (count not given). Method (training-free): Fisher-weighted merge of all experts of a layer into one shared base weight W_b, deltas W_i - W_b compressed by truncation-aware SVD, semi-dynamic structured pruning of the base. Compression 20/40/60% (80% max). Calibration data and tasks: WikiText-2, PTB, C4 PPL; 7 zero-shot tasks (OpenbookQA, ARC-e/c, WinoGrande, HellaSwag, PIQA, MathQA). Baselines NAEE, MoE-I2, MC-SMoE, MoE-Compress, SparseGPT 2:4, ASVD, LoSparse. Speed test: Mixtral-8x7B, batch size 64.
- Numbers:
  - Mixtral-8x7B 40% compression, avg zero-shot accuracy: D2-MoE 0.57 (PPL WikiText-2 5.28, PTB 20.54, C4 10.10) | baseline: Original 0.63 (PPL 3.98/12.99/6.78); NAEE 0.48 (6.44); MoE-I2 0.49 (6.74) | ratio: accuracy 0.90x of original; PPL 1.33x original; vs NAEE +9 points (+19% rel.), vs MoE-I2 +8 points (+16%) | where: Table 2 p6
  - Mixtral 60% compression: avg 0.52, WikiText-2 PPL 6.46 | baseline: Original 0.63 / 3.98; NAEE 0.36 / 11.43 | ratio: 0.83x acc; PPL 1.62x original | where: Table 2
  - Mixtral 20% compression: avg 0.60, WikiText-2 4.65 | baseline: Original 0.63 / 3.98 | ratio: 0.95x acc; PPL 1.17x | where: Table 2, Table 3
  - Phi-3.5-MoE 40%: avg 0.60, WikiText-2 6.07 | baseline: Original 0.62 / 3.48 | ratio: PPL 1.74x original | where: Table 2
  - Qwen2-57B-A14B 40%: avg 0.58, WikiText-2 8.19 | baseline: Original 0.59 / 5.12 | ratio: PPL 1.60x | where: Table 2
  - DeepSeekMoE-16B 40%: avg 0.49, WikiText-2 7.93 | baseline: Original 0.56 / 6.38 | ratio: PPL 1.24x | where: Table 2
  - Throughput Mixtral-8x7B, BSZ=64, 60/70/80% compression: 277.72 / 300.33 / 313.29 tok/s (481/386/290 TFLOPs) | baseline: NAEE 271.89 / 272.66 / 278.53; MoE-I2 227.60 / 252.55 / 294.04; LoSparse 158.90 / 191.45 / 198.04 | ratio: vs NAEE 1.02x / 1.10x / 1.125x | where: Table 4 p7 and App. A.3
  - Qwen3-30B-A3B at ~50% (independent, from the LorExperts paper): PPL 23.58, MMLU 45.7 | baseline: uncompressed PPL 8.50, MMLU 77.8 | ratio: 2.77x PPL; -32.1 MMLU points | where: LorExperts Table 4 (arXiv 2608.07814)
  - Within-layer expert similarity: CKA between expert weights 0.3-0.5 (Fig 1); delta V matrices CKA ~0.9, U ~0.3 (Fig 5, App A.6) | baseline: n/a | ratio: n/a | where: Intro p2; App A.6
- Catalog / first-pass numbers checked:
  - Shared base plus per-expert delta, deltas compressed by SVD: yes; Fisher-merged base per layer + truncation-aware SVD of deltas + base pruning
  - 40-60% compression with quality loss: yes; Mixtral: avg acc 0.63 -> 0.57 (40%) -> 0.52 (60%); PPL x1.33 and x1.62
  - base is within a layer (merged experts): yes; base merges all experts of one MoE layer; no cross-layer sharing
  - (ICML 2025): absent; not stated in the arXiv v1 text; keep as catalog's claim, unverified from the paper
  - first-pass: over 13% better than other compressors at 40-60%: yes; abstract wording; consistent with +16-19% rel. accuracy at 40% on Mixtral
  - first-pass: Qwen3-30B-A3B PPL 23.58 vs 8.50: yes; this number is not in the D2-MoE paper; it is in LorExperts Table 4 (D2-MoE re-run by AMD)
  - first-pass: hardware not stated: partly wrong; A100 GPUs are stated; count not
- Transfers to our regime: no. Why: Base is a merge of the experts in the same layer, so compressed deltas still need the base resident in RAM; there is no cross-layer or prediction-based residual. At 128-512 experts per layer a single Fisher-merge base is far from most experts (LorExperts shows D2-MoE degrading sharply on Qwen3-30B: PPL 23.58 vs 8.50). Quality loss is large at the ratios that would matter (40-60% bytes). Throughput table is GPU batch 64 and compares only to other compressors.
- Quality loss: Table 2/3: Mixtral-8x7B avg zero-shot 0.63 -> 0.60 / 0.57 / 0.52 at 20/40/60%; WikiText-2 PPL 3.98 -> 4.65 / 5.28 / 6.46; Phi-3.5 40%: 0.62 -> 0.60; Qwen2-57B 40%: 0.59 -> 0.58. No MMLU/generative benchmark. Training-free, no fine-tuning.
- Ablation: Yes: Tables 5-9: merge method for base (Fisher PPL 5.28 vs frequency 6.42 vs TIES 12.45 on WikiText-2); base/delta compressors (Table 6: truncation-aware SVD 5.28 vs vanilla SVD 6.22); calibration data (Table 9); App A.6 shared-V / merged-U extreme compression (80% ratio PPL 12.95). None isolate cache or layout.
- Limitations (authors): Impact-statement only; paper does not discuss limitations of expert count. Compression time reported (Table 10; delta SVD ~25 min on Mixtral).
- Contradicts our findings: Partly consistent. Their CKA 0.3-0.5 between experts (a function-space statistic, rotation invariant) is not a cosine; our median |cos| ~0.005 between Qwen3 experts and the LorExperts r<=0.21 co-activation/weight correlation agree that raw weights are near-orthogonal. D2-MoE's success on Mixtral (8 experts) is the regime where a single base works; its failure on Qwen3-30B (128 experts) is consistent with our 'no shared structure' result.

## P-0238: DeRS: Towards Extremely Efficient Upcycled Mixture-of-Experts Models (Yongqi Huang*, Peng Ye*, Chenyu Huang, Jianjian Cao, Lin Zhang, Baopu Li, Gang Yu, Tao Chen; Fudan / CUHK / Shanghai AI Lab / Baidu USA / StepFun; arXiv v1 3 Mar 2025, cs.LG, venue not stated in the text)

- Read: arXiv:2503.01359v1 (PDF, full text); raw text read: True
- Setup: Upcycled MoE only (experts initialised as copies of one dense FFN). Three tasks, six architectures: MoE-LLaVA (StableLM-2-1.6B, Qwen-1.8B, Phi-2-2.7B; 4 experts, top-2, every other block) on general multimodal benchmarks; Med-MoE (StableLM-2-1.6B, Phi-2-2.7B) on VQA-RAD/SLAKE/PathVQA; Coder-MoE (DeepSeek-Coder-1.3B) on HumanEval/MBPP. Training on 8x or 4x A100-80G (Tab S19). DeRS compression: expert = shared base + delta; delta sparsified (drop rate up to 0.99) or quantised (16 to 1 bit); DeRS upcycling: train sparse-matrix or low-rank deltas against a trainable shared FFN.
- Numbers:
  - MoE layer parameters, DeRS compression, general multimodal: reduced by 65% without performance drop (4 experts x16 bits -> 16 + 4x2 bits with 2-bit deltas; 90% sparsity gives 1+4x0.1 expert-equivalents) | baseline: vanilla upcycled MoE-LLaVA | ratio: up to 65% fewer params per MoE layer | where: Intro p2; Sec 4.1; Fig 4
  - Extreme settings: sparsification 0.99 drop rate or 1-bit deltas still without significant degradation | baseline: vanilla | ratio: n/a | where: Sec 4.1 p6
  - DeRS upcycling, additional trainable params: 1.11M (DeRS-SM) / 2.42M (DeRS-LM) extra params | baseline: vanilla upcycling | ratio: 2270x / 1041x fewer extra params, +0.3% / +0.2% overall performance | where: Sec 4.1 Table 1
  - Coder-MoE cost analysis (DeRS-LM upcycling): model size -52.7%, training memory -21.2%, inference memory -43.8%, overall performance +0.7% | baseline: vanilla upcycling | ratio: 0.47x size | where: Sec 4.4 Table 6
  - Cosine similarity of trained experts to initial FFN and to each other: > 0.999 (e.g. 0.9990-0.9998) | baseline: n/a | ratio: n/a | where: Fig 1, Intro
- Catalog / first-pass numbers checked:
  - Base plus sparse or quantized delta: yes; DeRS-Sparsification / DeRS-Quantization (compression) and sparse-matrix / low-rank (upcycling)
  - works for upcycled MoE where experts start as copies of one dense FFN, not for from-scratch MoE: yes; Appendix: 'not applicable to compressing MoE models trained from scratch ... impossible to extract redundant delta weights' since random init
  - first-pass: up to 65% fewer parameters per layer without drop: yes; abstract/intro wording
  - first-pass: Coder-MoE size -52.7%: yes but mislabelled; 52.7% is the DeRS-LM upcycling (training-time) size cut, not the post-hoc compression figure
  - first-pass: 2-bit deltas OK: yes; Sec 4.1; even 1-bit shown without significant drop on the 3 general models
- Transfers to our regime: no. Why: Redundancy exists only because experts are fine-tuned copies of one FFN (cosine > 0.999). Real from-scratch MoEs (Qwen3-30B, our 125B-class 512-expert model) have near-orthogonal experts, and the authors state the compression is not applicable. Models are 1.3-2.7B with 4 experts and the evaluated metrics are VQA/code; no throughput or storage-I/O claims.
- Quality loss: Reported as no significant drop on VQA/SQA/POPE etc. for upcycled MoE-LLaVA at 65%; differences in Table 1 are +0.2 to +0.3% for upcycling; no perplexity numbers. Fine-tuning data is the same as for vanilla upcycling (training required for upcycling variant).
- Ablation: Yes (Sec 4.4): freezing the shared base costs 1.3% / 1.6% (SM/LM); sparse rate and rank sweeps (Table 4); rank 64 degrades.
- Limitations (authors): Authors state it needs a shared pre-trained initialisation; from-scratch training with DeRS upcycling may be capacity-limited (App D).
- Contradicts our findings: No; it supports the reading that cross-expert redundancy needs shared initialisation, which our Qwen3 probe lacks.

## P-0239: Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression (Peijun Zhu, Ning Yang, Baoliang Tian, Jiayu Wei, Weihao Zhang, Haijun Zhang, Pin Lv; Guangzhou Univ., CASIA, Tianjin Univ., USTB; arXiv v4 21 Jul 2026, cs.CL; ICML 2026 submission template)

- Read: arXiv:2510.02345v4 (PDF, full text); raw text read: True
- Setup: Trained-from-scratch 12-layer Transformer, d_model=768, E=32 experts, G=8 groups, rank r=16, T=100 recluster interval, alpha=0.7, top-g1=1 group then top-2 experts; trained on 2 A100 GPUs with AdamW; evaluated on GLUE dev and WikiText-103 (plus pretraining Table 7 and long-text Table 8, not examined). Baselines Dense 110M, Switch-Top2 (875M), Expert-Choice, MoE-Lite (295M), D2-MoE (620M). Memory 'peak GPU memory (GB)'. Shared base FP16 + INT4 residuals; inactive groups offloaded to NVMe with EMA prefetch predictor (lookahead L=2 steps). The text never says whether a group spans experts of a single layer or a cross-layer set; compression formula is per group K*d_in*d_out -> d_in*d_out + K*r*(d_in+d_out).
- Numbers:
  - Total params: Ours 188M | baseline: Switch-Top2 875M | ratio: 0.215x (-78.5%) | where: Table 1 p7
  - GLUE avg: Ours 83.5 (offload 83.3) | baseline: Switch-Top2 85.1; Dense 83.8; D2-MoE 84.1 | ratio: -1.6 vs Switch; -0.6 vs D2-MoE | where: Table 1
  - Throughput: Ours 10.2 k tok/s (offload 10.1) | baseline: Switch-Top2 8.9; Dense 10.5 | ratio: 1.146x vs Switch | where: Table 1
  - WikiText-103 PPL: Ours 26.8 (offload 26.9; quant 27.5) | baseline: Switch-Top2 24.5; Dense 29.8; MoE-Lite 25.1 | ratio: +2.3 PPL vs Switch (1.09x) | where: Table 2
  - Peak GPU memory: Ours 19.2 GB (offload 16.5, quant 15.7) | baseline: Switch-Top2 33.1 GB; Dense 15.4 | ratio: 0.58x (-42%) -- text says '50% less' | where: Table 2, Sec 4.2 p7
  - Throughput WikiText: Ours 8.5 (offload 8.2) | baseline: Switch 7.2 | ratio: 1.18x; offload costs 3.5% vs no offload | where: Table 2
  - Offloading bookkeeping overhead: < 0.5% wall-clock (text); Table 6 'Dynamic Offloading' 0.5% | baseline: n/a | ratio: n/a | where: Sec 3.4 p5-6; Table 6
  - Total added overhead: 9.0 +/- 1.5% (residual quantisation alone 6.0%) | baseline: n/a | ratio: n/a | where: Table 6
  - Ablation w/o low-rank compression: GLUE 83.3, throughput 6.9 | baseline: Full 83.5, 10.2 | ratio: -32% throughput | where: Table 3
  - Compression ratio formula example: CR ~6.6x for d=4096, K=8, r=16 | baseline: n/a | ratio: formula d^2*K/(d^2+2Kdr) = K/(1+2Kr/d) = 8/1.0625=7.5 by my arithmetic, not 6.6 | where: Sec 3.2 p4
  - All-to-all volume: 6.5 GB (dynamic) vs 7.8 (static) vs 8.0 (D2-MoE); text says 28-41% reduction | baseline: n/a | ratio: n/a | where: Table 4; Sec 3.3
- Catalog / first-pass numbers checked:
  - Shared base per expert group plus low-rank residuals A_i, B_i: yes; W_i = W_base^g + A_i B_i, r=16
  - storage d_in*d_out + K*r*(d_in+d_out): yes; Sec 3.2 equation
  - within-layer groups: absent; the paper does not state it; groups are regrouped online by fused param+activation similarity and routed hierarchically, layer scope is never specified in the extracted text
  - first-pass: ~80% fewer params, 10-20% throughput: yes; abstract; Table 1: 188M vs 875M (-78.5%), throughput 1.146x (Table 1) and 1.18x (Table 2)
  - first-pass: Table 2 memory 19.2 vs 33.1 GB (~42%): yes; text says 50% less peak GPU memory; 0.58x is -42%
  - first-pass: GLUE 83.5 vs 85.1: yes; Table 1 / Table 5 (3 runs 83.5 +/- 0.2 vs 85.1 +/- 0.2)
  - first-pass: compression formula internal inconsistency (7.5x vs 6.6x): yes; confirmed: K/(1+2Kr/d) with d=4096, K=8, r=16 is 7.53x; paper says ~6.6x
- Transfers to our regime: no. Why: A 188M model trained from scratch with this structure on GLUE/WikiText-103; groups are discovered during training, not post-hoc on an existing 125B model. NVMe offload of idle groups is described in one paragraph and has no isolated ablation (Table 2 'Ours + Offload' vs 'Ours' is the only evidence: peak memory 19.2 -> 16.5 GB, throughput 8.5 -> 8.2 k tok/s, PPL 26.8 -> 26.9; the offload is of groups on a GPU box with 2 A100). Batch size and storage bandwidth not given.
- Quality loss: GLUE avg 83.5 vs 85.1 (Switch) and 84.1 (D2-MoE); WikiText-103 PPL 26.8 vs 24.5 (Switch). Trained from scratch, so 'loss' is relative to a different, bigger model, not to an uncompressed version of the same weights.
- Ablation: Yes (Table 3): w/o online clustering GLUE 81.7, load balance 0.37; w/o low-rank comp. throughput 6.9; w/o hierarchical routing 9.1. Table 4 dynamic vs static clustering. Offload only as a row in Table 1/2.
- Limitations (authors): Authors list future work: theory of clustering stability, scaling to larger pretraining/expert counts, adaptive rank, multitask/multimodal. Not stated: NVMe bandwidth, batch size, layer scope.
- Contradicts our findings: No. The claim that experts in a group have 'correlated transformations' is asserted, not measured against weights; the paper has no cosine/CKA measurement of pretrained experts.

## P-0240: Shape Mutating Expert Compression: LorExperts and BTExperts (Inesh Chakrabarti*, Sourjya Roy*, Bowen Bao, Thiago Crepaldi, Spandan Tiwari, Ashish Sirasao; Advanced Micro Devices; arXiv v1 7 Aug 2026, cs.LG)

- Read: arXiv:2608.07814v1 (PDF, full text; the catalog URL was an alphaxiv .md mirror); raw text read: True
- Setup: Models: Qwen3-30B-A3B (E=128, top-8; uncompressed PPL 8.50 on Wikitext-103, MMLU 77.8), Gemma-4-26B-A4B (uncompressed PPL ~7.15), Mixtral-8x7B for the gate_proj speedup arithmetic. ~50% expert compression: cluster experts per layer (k-medoids by co-activation NPMI or weight Frobenius distance), keep one full-precision dominant per cluster, store other members as neuron-permutation-aligned (Jonker-Volgenant) rank-r (r=64) residuals. Calibration: C4, 64 sequences x 2048 tokens. Eval: Wikitext-103 PPL plus LM-Eval-Harness MMLU/ARC-C/HellaSwag/WinoGrande/PIQA. Hardware: not stated. Baselines: M-SMoE, HC-SMoE, Freq-Prune, REAP, D2-MoE.
- Numbers:
  - Qwen3-30B-A3B ~50% compression, LorExperts (Coact-NPMI): PPL 12.20, MMLU 54.1, ARC-C 37.1, HellaSwag 52.6, WinoG 68.4, PIQA 75.3 | baseline: Uncompressed PPL 8.50, MMLU 77.8, ARC-C 52.6, HellaSwag 59.6, WinoG 71.0, PIQA 79.4 | ratio: PPL 1.435x; MMLU -23.7 points | where: Table 4 p9
  - same, best variant (WS-Frob+protK8): PPL 11.79, MMLU 53.7 | baseline: uncompressed | ratio: PPL 1.39x; MMLU -24.1 | where: Table 4
  - same, Msoft @40.6%: PPL 11.99, MMLU 54.9 | baseline: uncompressed | ratio: PPL 1.41x; MMLU -22.9 | where: Table 4
  - same, Msoft @51%: PPL 15.37, MMLU 46.0 | baseline: uncompressed | ratio: 1.81x PPL | where: Table 4
  - Baselines at ~50%: D2-MoE PPL 23.58 / MMLU 45.7; Freq-Prune 29.64 / 44.6; REAP 34.14 / 32.1; HC-SMoE 39.50 / 43.3; M-SMoE 68.61 / 24.7 | baseline: uncompressed 8.50 / 77.8 | ratio: D2-MoE 2.77x PPL | where: Table 4
  - Gemma-4-26B-A4B ~50%, LorExperts (WS-Frob+protK8) @49.9%: PPL 14.66, MMLU 51.1 | baseline: uncompressed PPL ~7.15 | ratio: 2.05x PPL | where: Table 5 p9
  - Co-activation (NPMI) vs weight cosine correlation, Qwen3-30B: gate 0.085-0.208, up -0.005..0.017, down -0.026..0.020; layers 5/11/23/35/47; text: r about 0.08-0.21 (gate), effectively zero for up/down | baseline: n/a | ratio: n/a | where: Table 1 p3, Sec 2.2
  - Rank-64 truncation of an expert matrix (I=768, H=2048): retains 26.0% of squared Frobenius energy (r=200 56%, r=311 72%, r=450 86%) | baseline: flat spectrum would give 8.3% at r=64 | ratio: 3.1x a flat spectrum | where: Table 2 p4
  - Routing concentration, 64 busiest of E=128: top-64 share 0.945 mean (WikiText-2); 24.6 'dead' experts per layer on average | baseline: uniform 0.500 | ratio: n/a | where: Table 3
  - Speedup claim: ~1.9x FLOPs/HBM-read cut for Mixtral gate_proj k=2, I=14336, H=4096, r=64, 'kernel-independent' | baseline: uncompressed | ratio: analytic only, no measured wall-clock; authors say 'make no straw-man speedup claims' | where: Sec 5/Fig 4 p8, Limitations 2
- Catalog / first-pass numbers checked:
  - a single global anchor is far from most experts and expert weights are near-orthogonal, so a low-rank delta leaves a large residual: yes; Intro and Sec 2 say exactly this; verified from raw text, no longer 'snippet only'
  - first-pass: Qwen3-30B PPL 12.20, MMLU 54.1 vs uncompressed 8.50: yes; Table 4 (Coact-NPMI row); uncompressed MMLU is 77.8, so the loss is 23.7 MMLU points, which the first pass did not state
  - first-pass: weight-coactivation corr r~0.08-0.21: yes; gate projection only; up/down are ~0; and this is within-layer expert pairs, not cross-layer
  - first-pass: PPL still 43% above uncompressed: yes; 12.20/8.50 = 1.435
  - first-pass: 1.9x speedup theoretical: yes; analytic for Mixtral gate_proj; no measurement
  - first-pass: D2-MoE PPL 23.58 on Qwen3-30B (cited for P-0237): yes; Table 4
- Transfers to our regime: partly. Why: Supports our measurement that expert weights are near-orthogonal (within layer, here), and quantifies what a rank-64 + aligned-dominant scheme costs on the same Qwen3-30B-A3B we probed: ~50% fewer expert parameters at PPL +3.7 and MMLU -23.7. That is a lossy 2x byte cut with a quality price our lossless-byte plan avoids. It is within-layer and post-hoc, so it does not touch the cross-layer predicted-residual idea. Assumptions that break: GPU/HBM speedup argument (not NVMe), calibration needs activation data, quality loss not recoverable without more rank, and a dominant per cluster still needs to be read in full.
- Quality loss: Qwen3-30B-A3B: PPL 8.50 -> 12.20, MMLU 77.8 -> 54.1, ARC-C 52.6 -> 37.1, HellaSwag 59.6 -> 52.6, WinoG 71.0 -> 68.4, PIQA 79.4 -> 75.3 (Coact-NPMI, ~50%). Gemma-4: PPL ~7.15 -> 14.66, MMLU 51.1 (uncompressed MMLU not in the extracted table row). Sweep Fig 5: accuracy drops mainly near 50%.
- Ablation: Yes (Appendix A, Table 6): tree (BTExperts) vs flat per-cluster decomposition (sharing corrections across a hierarchy does not beat flat on near-orthogonal experts); clustering metrics WS-Frob / Coact-NPMI / Msoft; protecting top-8 firing experts raises Qwen3 MMLU 42.4 -> 53.7. No ablation of alignment alone found in the extracted text.
- Limitations (authors): Authors: calibration cost (per-member SVD + reconstruction fine-tuning ~500 steps), inference amortisation bounded and routing-dependent, closed-form Hessian-weighted fit is ongoing work; utilisation measured on WikiText-2 only, domain shifts change busy sets. Hardware not stated.
- Contradicts our findings: No contradiction; strengthens it. Caution for our probe: LorExperts aligns neuron permutations before forming residuals ('experts may be similar only up to a neuron permutation'); our raw element-wise cosine and residual test on Qwen3-30B (no permutation alignment, gate matrices only, 16 rows) does not test that. Their reported outcome with alignment (rank-64 keeps 26% energy, PPL +43%) is still far from lossless, so it would not rescue a ~1.0x byte bound, but the alignment case is untested by us.

## P-0241: Improving Parameter Utilization by Sharing Neural Experts Across Layers in Transformers (CS-MoE; Dian Jiao*, Jiaxin Duan*, Shuai Zhao, Jiabing Leng, Yiran Zhang, Feng Huang; China Electronics Cloud Technology Co.; arXiv v1 31 Aug 2026, cs.LG)

- Read: arXiv:2609.22199v1 (PDF, full text); raw text read: True
- Setup: Pilot (App. B): 'pre-trained Qwen3-MoE-4B' with 36 layers, 4 experts/layer, top-1; held-out validation 50,000 tokens from DCLM; linear CKA computed on Gram matrices of expert activations (not on weights); greedy zero-shot swap: each expert in deep layers 24-32 replaced by the shallow-layer (1-8) expert with highest CKA (>0.85 as 'redundancy indicator'), no fine-tuning, compared with random swaps at expert sharing ratio ESR 5% and 20%. CS-MoE proper: trained from scratch, 0.6B-12B (Qwen3 style), Megatron-LM on 8x NVIDIA H200 (up to 32 nodes for a 4.5T-token run); each layer has independent experts plus a global shared pool (M=324 for d=4096), per-token routing.
- Numbers:
  - Zero-shot targeted CKA swap Shallow->Deep, +PPL: +1.22 (ESR 5%), +3.45 (ESR 20%) | baseline: random swap +4.20 / +12.80; random arbitrary-layer swap +4.15 / +15.30 | ratio: 0.29x / 0.27x of random damage | where: Table 4 p14
  - Deep->Shallow targeted: +1.54 / +3.98 | baseline: random as above | ratio: n/a | where: Table 4
  - Base PPL of the pilot model: not stated in text (Fig 1 y-axis 10-80; authors call < 15 'acceptable') | baseline: n/a | ratio: n/a | where: Fig 1; App B.1
  - 8B CS-MoE activating 55% of physical params: PPL 9.55 | baseline: fully activated 8B dense 9.75 | ratio: -0.20 PPL | where: Sec 3 / Table 7
  - Ablation: no private experts (N_indep=0): PPL 10.32 | baseline: default 10.08 | ratio: +0.24 | where: Table 3
  - Decode latency at batch size 1, 8x H200, 8B scale: Dense 14.2 ms/tok; standard sparse MoE 16.5; CS-MoE default 15.8; CS-MoE with cache optimisation 14.9 | baseline: standard MoE 16.5 | ratio: 0.90x of MoE latency | where: App D.3 Table 11
  - Memory traffic claim: '45% reduction in unique memory traffic per token' (adjacent layers re-use shared experts in L2/SRAM) | baseline: n/a | ratio: n/a | where: App D.3 text
  - Adjacent-layer overlap of top-10% shared experts: 68.4% (layer l vs l+1) falling to 42.1% (l vs l+5); Gini 0.28 | baseline: n/a | ratio: n/a | where: App D.4
- Catalog / first-pass numbers checked:
  - Cross-layer sharing found by CKA above 0.85 on Qwen3-MoE-4B: yes with a correction; the 0.85 threshold and the 4-expert 36-layer model are in App B.1; it is a post-hoc zero-shot swap test on a pretrained model, not a training-time design (the training-time design is CS-MoE itself, which is trained from scratch)
  - a training-time design: partly wrong; the CKA finding motivates the training-time CS-MoE architecture, but the numbers in the pilot are zero-shot swaps
  - CKA is rotation-invariant so it does not imply element-wise similarity: yes; App B.1 itself says CKA 'is invariant to orthogonal transformations' and chooses it to avoid permutation invariance of weights; moreover it is computed on activations, not weights
  - first-pass: PPL +1.22 (5%), +3.45 (20%) vs random +4.20/+12.80: yes; Table 4
  - first-pass: 55% active params of dense at lower PPL: yes; 8B: 9.55 vs 9.75
- Transfers to our regime: partly. Why: Two separable pieces. (1) The CKA swap evidence is functional (activation-based), 4 experts per layer, 36 layers, +1.2 PPL at 5% swapped: it says deep experts can sometimes be replaced by shallow ones, not that one is a byte-level residual of the other. For our 512-expert model it is unmeasured. (2) CS-MoE is a from-scratch architecture (retraining required) whose point is that the same expert is reused by multiple layers, so a cache of a shared pool would hit across layers; the only latency datum is a GPU batch-1 decode table (14.9 vs 16.5 ms/tok) with weights resident in HBM. Does not apply to already-trained 125B weights.
- Quality loss: Pilot: PPL increase +1.22 to +3.98 for targeted swaps, base PPL not given. CS-MoE: PPL lower than dense at 55% active params; downstream Table 1/2 accuracy also reported (not itemised here).
- Ablation: Yes (Table 3): removing independent experts PPL 10.32 vs 10.08; pool size, top-K, load-balance coefficient alpha=0.01. No ablation of the cache effect separate from architecture.
- Limitations (authors): Authors: validated only up to 12B; no tailored distributed infrastructure for a cross-layer pool; effects under RLHF/DPO/PEFT unknown. Pilot model and its pretraining are not described.
- Contradicts our findings: Not a contradiction of the cosine ~0 result (CKA on activations vs raw weights), but it is the first read paper showing functional cross-layer interchangeability on a pretrained MoE; our element-wise residual test cannot detect that, and our result must be worded as 'no byte-level cross-layer residual', not 'no cross-layer redundancy'. Worth knowing the pretrained 'Qwen3-MoE-4B' with 4 experts is not a public model I can verify.

## P-0242: DIMoE-Adapters: Dynamic Expert Evolution for Continual Learning in Vision-Language Models (Mengxin Qin, Xiang Zhang, Xi Wang, Kun Wei, Xu Yang, Cheng Deng; Xidian University; arXiv v1 8 May 2026, cs.CV)

- Read: arXiv:2605.07494v1 (PDF, full text; Figure 6 is an image, only its caption and text are readable); raw text read: True
- Setup: Continual-learning (MTIL) on CLIP ViT-B/16: each expert is a LoRA module, router is a single-layer MLP with Top-p gating (p0=0.6), experts grown/pruned by SCEE during training; single NVIDIA A6000. Appendix E: for each incremental task one randomly selected newly introduced expert is taken as the task representative; cosine similarity of expert parameter vectors is computed after training between experts of different tasks, shown as per-layer matrices (Fig 6, MTIL Order I).
- Numbers:
  - Off-diagonal cosine similarity of task-specific LoRA experts: 'close to zero' across all (selected) MoE layers; no numeric value in the text | baseline: diagonal = 1 by definition | ratio: n/a | where: App E, Fig 6 (qualitative, p18-19)
- Catalog / first-pass numbers checked:
  - cosine similarity between expert parameter vectors across MoE layers is near zero: wrong object; the matrices are per layer, between experts of different TASKS (one random expert per task), not between experts of different layers; the experts are LoRA adapters on CLIP ViT-B/16, not pretrained LLM expert FFNs
  - so no cross-layer redundancy: no; the paper makes no cross-layer claim; the conclusion drawn is that task-specific experts are 'nearly orthogonal rather than redundant'
  - first-pass: wrong object (task adapters, not pretrained LLM expert weights): yes; confirmed; additionally it is within-layer across tasks, not cross-layer
- Transfers to our regime: no. Why: LoRA experts trained on different tasks from a shared init, in a CLIP vision encoder; near-zero cosine there is expected (independently trained low-rank updates in a high-dimensional space) and says nothing about pretrained LLM expert FFNs or about layer-to-layer similarity.
- Quality loss: not applicable (analysis of a continual-learning method)
- Ablation: none for the similarity claim
- Limitations (authors): Qualitative figure only; one random expert per task is sampled.
- Contradicts our findings: No. But pubtech F.md (family 19 'Published methods vs measured') cites DIMoE-Adapters as literature for 'near-zero cross-layer cosine'; that citation should be dropped. The real supporting literature is LorExperts Table 1/Sec 2.2 (within-layer weight cosine vs co-activation r<=0.21; 'near-orthogonal') and Meta Expert App A.2 ('naive cosine similarities collapse toward zero' in high dimensions).

# Per-family prior art and novelty

## Family s2 #18 (experts / weights served from another host over the network; distributed MoE decode)

Read in full for this batch: Arm Graviton4 RPC learning path (4 sub-pages), MegaScale-Infer (2504.02263v4), llama.cpp issue 19745 (body + comments). Additional prior art found and read at table/abstract level from raw text: prima.cpp (Li et al., arXiv 2504.08791v3, ICLR 2026): pipelined-ring parallelism with disk prefetch for 30-70B dense models on 4-6 home devices (CPU/GPU, Wi-Fi, slow disks); 70B TPOT 674 ms (without its Halda scheduler 20848 ms, without prefetch 755 ms; llama.cpp 10120 ms, exo and dllama OOM), Table 4. It is dense models, but it is the closest published CPU-cluster-with-disk-offload system and states sparse/MoE systems were excluded. Chen et al. 'Towards Building Private LLMs: Exploring Multi-Node Expert Parallelism on Apple Silicon for MoE' (arXiv 2506.23635, RACS 2024): DBRX (132B, 16 experts, top-4) unquantised bf16 on a Mac Studio M2 Ultra cluster over 10 Gb Ethernet, 6.1 tok/s, experts partitioned over nodes with fork-join per layer, memory resident (a performance model for node count and network). Also listed, not read in full: D2MoE dual routing (2504.15299) and CoMoE collaborative aggregation/offloading at edge (2508.09208), EdgeMoE-style systems (from memory). Not found by arXiv listing search: a paper that keeps the expert store passive (RAM/NVMe) on a second host and ships expert slices to a thin client per token at batch 1. Closest conceptual: MegaScale-Infer (disaggregated expert hosts, but active GPUs and large batches), the Mac Studio expert-parallel paper (experts on nodes, activations cross the network, memory resident), prima.cpp (disk offload on home clusters, dense).
What is new in our variant, if anything: a passive weight store whose per-token traffic is the selected expert bytes (not activations), at batch 1 on CPU; I found no paper that measures this. Every network-based system found ships activations to nodes that own the weights, which is the opposite direction of traffic. Novelty: open/known-to-variant (the architecture of remote expert hosts is known; the passive-store variant is not published in what I could read). Note the Arm tutorial's own measured figure (1.75 tok/s for a 37 GiB dense model over 3 Graviton4 nodes) and the Mac cluster's 6.1 tok/s over 10 GbE show what activation-shipping RPC clusters reach.

## Family s2 #19 (quantised residual against the predicted previous-layer expert; cross-layer/within-layer expert redundancy)

Read in full for this batch: D2-MoE, DeRS, Breaking the MoE Trilemma, LorExperts/BTExperts, CS-MoE, DIMoE-Adapters. Further prior art read from raw text:
- Meta Expert: Cross-Layer Expert Sharing for Storage- and Retrieval-Efficient MoE (Hua et al., Tsinghua/Shanghai AI Lab, arXiv 2610.27598v1, 20 Oct 2026): training-time cross-layer shared experts in a Qwen3.5-style ladder trained from scratch on H200s (12-24 layers, 128 routed experts per layer); matches a vanilla MoE with 70-80% of the experts at fixed activation and granularity. Redundancy is measured by linear CKA on expert responses to a probe set, explicitly because 'naive cosine similarities collapse toward zero' (App A.2). It is the closest published statement of cross-layer redundancy, but at the functional level and only by retraining. Not post-hoc.
- Shared Low-rank Basis Factorization (SLBF, arXiv 2610.09342v1, 7 Oct 2026): data-free weight reconstruction with per-(layer, projection) shared rank-k bases, 'no cross-layer coupling'; Qwen3-30B-A3B-2507 at 24% compression avg 84.3 vs 84.9 original (MoBE 82.8, REAP 72.5, HC-SMoE 74.2, MoLAE 76.0), at 36% 82.4; Moonlight-16B at 15%: 62.5 vs 64.3; Qwen3.5-122B at 34%: MoBE 82.8 vs 86.3 (SLBF row not captured). Shows a within-layer shared basis can reach ~24-36% parameter cut with 0.6-2.5 points of loss on the same Qwen3-30B we probed, much better than D2-MoE's 50% PPL 23.58. So 'no shared structure' in our probe applies to a flat shared SVD basis across experts, not to a learned per-layer basis that is allowed to be non-orthogonal-aligned (MoBE/SLBF).
- MoBE (Mixture-of-Basis-Experts, arXiv 2508.05257v1): W=AB, B a linear combination of per-layer shared basis matrices; Qwen3-235B-A22B-2507, DeepSeek-V3-0324 (671B), Kimi-K2 (1T) parameters cut 24-30% with 1-2% accuracy drop (abstract, read from raw text; tables not itemised). MoLAE (arXiv 2503.23100v2): shared latent projection + expert-specific transforms. Sub-MoE (arXiv 2506.23266, listed, not read). Routing geometry across layers (arXiv 2609.02404): router control states share a canonical subspace across depth after Procrustes alignment, one linear transition R2 0.39-0.71 (about routing, not weights; relevant to cross-layer expert prediction not to residual bytes). SPICE (arXiv 2608.21240v2): low-rank expert surrogates for prefetch misses, up to 3.12x TPOT on GPU PCIe offload (abstract read). From the catalog and pubtech files: DeltaMoE, REAP, MC-SMoE, HC-SMoE (from memory, not re-read).
What is new in our variant: the specific test of a quantised residual against the previous layer's predicted expert (cross-layer, keyed by the router's prediction) as a byte lever for streaming from NVMe is not found in anything read: every delta/base paper uses a within-layer base or an upcycled-copy base, and the only cross-layer papers (CS-MoE, Meta Expert) are functional/CKA and training-time. Closest papers read in full: LorExperts (within-layer, aligned, rank-64: ~50% bytes at PPL +43%), CS-MoE and Meta Expert (cross-layer, functional, retraining or zero-shot swaps), SLBF/MoBE (within-layer basis, lossy). Our negative result (median |cos| ~0.005 for same expert ID in neighbouring layers of Qwen3-30B-A3B, residual ratio 0.99999) stands as a raw element-wise result but should not be generalised to function-level or permutation-aligned redundancy, which none of the read papers rule out and CS-MoE partly supports.

