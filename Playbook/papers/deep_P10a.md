# deep_P10a: raw-text re-check of P-0211, 0212, 0213, 0214, 0216, 0217, 0218, 0219, 0221

Read-only research, 2026-10-29. Raw text obtained with curl + pdftotext (arXiv PDFs, AAAI PDF) or curl + tag stripping (blogs). Geerling charts were downloaded as images and read directly. Scratch: `<scratch>`. Our measurements: `Playbook/pubtech/E.md` (families 14, 15, 16).

Counts: 9 rows. Raw text read: 8 rows fully (P-0211, 0212, 0213, 0216, 0217 via the AAAI PDF plus OD-MoE for its intended paper, 0218, 0219, 0221, and 0214 whole page). Unread: 0. Partial: none, but P-0211 is the 11-page ICML-style v3 (appendix read), and the Geerling post has no raw benchmark tables beyond the three bar-chart images.

## Corrections to the first pass (P10.md / P10.json)

1. P-0216: the first pass said the DeepSeek numbers (21.1/27.8/32.5) were not confirmable and that Kimi is "about 30, not 28". Both wrong. The chart images show DeepSeek V3.1 671B Exo(RDMA) 21.1 / 27.8 / 32.5 t/s on 1 / 2 / 4 nodes (llama.cpp TCP 14.8 / 16.0 / 14.6), and Kimi K2 Thinking Exo 21.6 (2 nodes) and 28.3 (4 nodes) (llama.cpp 18.5 / 16.4). The catalog is right on both; the post text says "around 30" loosely.
2. P-0211: first pass said hardware not on the page. It is: RTX A5000 24 GB, PCIe 4.0 (Table 1 says 24 GB/s, Section 5 says 32 GB/s), host DRAM 32-1024 GB depending on model. Headline 3.1-16.7x is DeepSeek-V2-Lite average TPOT, Table 1, vs vLLM and Ollama; the paper's own Conclusion says 2.7-13.7x (internal inconsistency).
3. P-0221: latency is wall-clock of the MoE block forward pass (not the whole forward pass) on one NVIDIA H200, batch/sequence not stated. First pass said hardware unstated.
4. P-0217: the first pass was right that the URL is FIRM-MoE; now read in full: RTX 3090 24 GB, 32-core CPU, 64 GB, PCIe 4.0 32 GB/s, Qwen1.5-MoE-A2.7B, Qwen3-30B-A3B, DeepSeek-MoE-16B, DeepSeek-V2-Lite, batch 1; "up to 1.5x / 2.8x" is the abstract; body gives 1.8x vs baseline BB at a 128-expert cache and 1.31x average throughput gain over Fiddler/llama.cpp.
5. P-0212: first pass omitted the quality loss. DAOP is lossy (Table VI): Mixtral GSM8K 58.91 (official) -> 51.48 / 48.07 / 41.77 / 33.51 at ECR 62.5 / 50 / 37.5 / 25%; Phi-3.5 GSM8K 86.88 -> 74.07 at 25%. Also DAOP v3 on arXiv is dated 21 Aug 2026.
6. P-0213: first pass fine. Added: the paper states in footnote 1 that it does not use SSD/NVMe because of large access delay; the shadow node costs 45 GB GPU memory (total 60 GB); worker DRAM is 192 GB.
7. P-0219: first pass correct that the page has no 200B MoE number and no 16 ms sync figure. Confirmed from raw text: the only llama.cpp RPC figures are "2-5 tok/s for 70B CPU-only" and "10-20 tok/s mixed CPU/GPU" and "405B at 2 tok/s", all unsourced. E.md's line "llama.cpp RPC (4-8 tok/s for 200B+ MoE on 10GbE ...)" has no source on this page.
8. P-0218: correct as written. Added exact numbers below. WDMoE itself is a different pair of papers (2405.03131, 2411.06681), both read at abstract level only (see family 15).

## Result that changes what we should believe about our own measurements

- Family 14 (ARC): our "ARC is within +0.005-0.012 of LRU" is now backed by outside evidence. FlashMoE (arXiv 2601.17063, read) evaluated ARC and LeCaR directly on MoE expert traces on a PCIe 5.0 NVMe desktop and found LRU the best of the heuristics for token rate; their learned policy beat LRU by +22% (OLMoE) and +7% (Qwen3-30B-A3B) tok/s. So (a) ARC for MoE experts is not novel (it is a published baseline), (b) the Belady-LRU gap is real and partly reachable by a learned policy (their LRU 73% vs Belady 86% hit in Figure 2, similar to our +0.17-0.22 gap), so "predictor-driven policies have headroom" is supported with a measured 7-22% speed gain, not zero.
- Family 16 (top-k 10 to 4): Edge0 (arXiv 2609.18063, read) is a measured Mac + SSD result with K=8 to K=4 on a 256-expert model: decode 3.3 -> 6.4 tok/s (+94%) at the same cache budget on a 16 GB MacBook M2 (Table 3), with a trained recovery LoRA and trained prerouter. That agrees with our "8 GB serial gain 1.6-2.4x comes mostly from cache footprint" but it required training; our k override was training-free and failed the perplexity gate (+45.6% at top-4 on OLMoE).
- Family 15: the first pass was too pessimistic about RDMA evidence; the chart numbers are real. They still do not transfer (weights resident across nodes), so our arithmetic (RDMA 1.21x serial, 1.0x overlap vs local NVMe) is unaffected.
- Nothing found contradicts the measured ARC, network-tier or top-k numbers.

---

## P-0211: MoE-Infinity

- Paper: "MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache", Leyang Xue, Yao Fu, Zhan Lu, Chuanhao Sun, Luo Mai, Mahesh Marina (Univ. of Edinburgh). arXiv 2401.14361v3, 12 Mar 2025 (11 pages incl. appendix; no venue printed). Read from PDF.
- Setup: batch 1; one RTX A5000 24 GB over dedicated PCIe 4.0 (Table 1 caption 24 GB/s, Section 5 text 32 GB/s); all experts in host DRAM (64 GB Switch, 256 GB NLLB, 32 GB DeepSeek, 128 GB Mixtral, 1 TB Arctic), no SSD tier. Models: DeepSeek-V2-Lite (64 experts x 2.4B, 31 GB), Mixtral-8x7B (120 GB), Switch-128x0.2B, NLLB-128x0.4B, Arctic-128x4B. Prompt 512, decode 32; 290 tasks (BigBench, FLAN, MMLU). Baselines: DeepSpeed-Inference, llama.cpp (Ollama), Mixtral-Offloading, vLLM, BrainStorm (re-implemented, not open source). PyTorch/HF checkpoints.
- Numbers:
  - Table 1 (p1), DeepSeek-V2-Lite TPOT avg/tail ms: DeepSpeed 737/803, Mixtral-Offloading 1250/1530, Ollama 2590/2599, vLLM 485/493, MoE-Infinity 155/169. Ratios (mine, avg TPOT): 4.75x, 8.06x, 16.7x, 3.13x. GPU idle time: 513/754/2073/254 vs 51 ms.
  - Section 5.1 text (p7): Mixtral latency "as low as 836 ms"; 1.4x vs BrainStorm and Mixtral-Offloading. Switch 155 ms, NLLB 531 ms, DeepSeek 155 ms. BrainStorm 934 ms vs vLLM 485 ms.
  - Long context (Fig 8): works to 2^16 tokens, then KV cache squeezes the expert buffer; at 2^17 falls to on-demand fetch (+137 ms).
  - Conclusion: "2.7-13.7x" (differs from abstract's 3.1-16.7x).
- Catalog row numbers: "3.1-16.7x latency versus mostly GPU-offload baselines (not ARC, not SSD)": correct (avg TPOT, DeepSeek-V2-Lite). "per-request working-set expert cache": correct. First pass "tables not read": now read.
- Conditions: GPU with experts cached in VRAM, PCIe host DRAM tier, batch 1, 64-128 experts (Mixtral, the 8-expert case, gains only 1.4x). Cache is guided by an Expert Activation Matrix Collection (request-level traces matched by cosine distance; Algorithm 1 evicts lowest predicted activation x layer-decay), plus prefetch.
- Transfers to us: no. GPU cache + DRAM tier, baselines are GPU offloaders (vLLM, DeepSpeed), no SSD, all weights fit in host DRAM (32 GB-1 TB). Cache is a share of VRAM, not 2-4 footprints of a 512-expert model. Useful content: it states LRU is what llama.cpp/vLLM/DeepSpeed implement (Figure 6 caption) and that expert reuse skew exists only within a request and flattens over 1000 requests (Fig 2), which supports per-sequence rather than global popularity.
- Quality loss: none (lossless caching and prefetch).
- Ablation: EAMC capacity (Fig 9: 1 to 120 entries, 3% of requests suffices), workload-shift recovery (Table 3, ~50 requests within a dataset). No ablation of cache vs prefetch separately, and no policy ablation against LFU/ARC in numbers (only the Fig 6 toy example).
- Limitations stated: only batch 1 personal use; multi-server left to future work; clustering of EAMs not scaled; long context shrinks cache.
- Contradicts our findings: no. It supports that an LRU-like cache is the common baseline; its gain over LRU alone is not isolated in the paper.

## P-0212: DAOP

- Paper: "DAOP: Data-Aware Offloading and Predictive Pre-Calculation for Efficient MoE Inference", Yujie Zhang, Shivam Aggarwal, Tulika Mitra (NUS). DATE 2025 (Lyon); arXiv 2501.10375v3, dated 21 Aug 2026 on the stamp. Read from PDF.
- Setup: Mixtral 8x7B (32 blocks, 8 experts, top-2, 46.6B total) and Phi-3.5-MoE (16 experts, top-2, 41.7B). Intel i9-10980XE (18 cores, 3.0 GHz, 130 GB DDR) + RTX A6000 48 GB (768 GB/s), PCIe 4.0 64 GB/s (Section V.b; Table I uses A100 + Xeon Gold 6326 for microbenchmarks). Batch 1. HF Transformers. ShareGPT used to calibrate the initial cache. Baselines: MoE-OnDemand, DeepSpeed-MII, Mixtral-Offloading, Fiddler. ECR (expert cache ratio on GPU) 46.9% for Fig 9.
- Numbers:
  - Table I (A100, decode, 256 tokens): block exec CPU 8.02 ms, GPU 1.24 ms; expert migration CPU to GPU 39.87 ms; activation transfer 0.02 ms. Migration is 32x a full GPU block (ratio 39.87/1.24 = 32.2, mine).
  - Fig 9 / Section V.B: [256,512] tokens, DAOP 4.52 tok/s Mixtral and 8.21 tok/s Phi-3.5, +40.4% vs Fiddler (implied Fiddler 3.22 tok/s, mine). MoE-OnDemand, DeepSpeed-MII, Mixtral-Offloading below 1 tok/s on Mixtral.
  - Fig 10: avg +35.4% over Fiddler across ECR 15-50%; at ECR 25%: 3.23 tok/s Mixtral, 5.03 tok/s Phi.
  - Abstract: 8.20x vs caching/prefetching, 1.35x vs offloading. 8.20x appears only as a figure bar, not in tables; not independently derivable (4.52/8.20 = 0.55 tok/s implied baseline, consistent with "<1 tok/s").
  - Table IV energy: 14.37 tokens/kJ DAOP vs 10.06 Fiddler (Mixtral), 1.50x average.
  - Prefill-vs-decode activation similarity 90.72% (Table II); next-layer prediction accuracy 84.11% (Fig 5).
- Catalog row numbers: "per-sequence expert allocation already published": true but incomplete: placement is decided from the prefill token counts, swapped once (threshold 1.05) and frozen through decode.
- Conditions: GPU is faster than CPU, CPU-GPU transfer is slower than CPU expert compute (their stated assumptions 1-3, Section VI.A), top-2 models with 8-16 experts, 130 GB DRAM so no disk.
- Transfers to us: no. Their key trick is computing missing experts on the CPU instead of moving weights. In our regime the CPU is the only processor and the cost is reading weights from NVMe; the activation-vs-weight size ratio (about 1:10,000) argument does not apply. The per-sequence prefill-to-decode similarity (0.907 for Mixtral top-2/8) is for 8 experts; for 512 experts and short prompts it is untested.
- Quality loss: yes, lossy (graceful degradation replaces a CPU-resident predicted expert by a GPU-resident next-best one). Table V (prefill-only tasks, ECR 25%): Mixtral HellaS 66.96 -> 66.80, MMLU 70.60 -> 70.47; Phi MMLU 78.78 -> 78.69. Table VI (whole generation): Mixtral TriviaQA 71.59 -> 69.08 (25%), BBH 49.36 -> 46.61, GSM8K 58.91 -> 33.51 (25%) / 51.48 (62.5%); Phi GSM8K 86.88 -> 74.07 (25%) / 82.79 (62.5%). The paper says "minimal impact" except GSM8K.
- Ablation: no separate ablation of allocation vs pre-calculation (only cache-size sweep). Cannot isolate the per-sequence swap.
- Limitations stated: GSM8K degradation (diverse experts within a sequence); needs GPU faster than CPU.
- Contradicts ours: no. Our "in-sequence LRU approximates the per-sequence allocation for free" is not tested by the paper; it never compares against LRU.

## P-0213: OD-MoE

- Paper: "OD-MoE: On-Demand Expert Loading for Cacheless Edge-Distributed MoE Inference", Liujianfu Wang, Yuyang Du, Yuchen Pan, Soung Chang Liew, Jiacheng Liu, Kexin Chen (CUHK). arXiv 2512.03927v1, 3 Dec 2025. Read from PDF (code link is "github.com/Anonymous/DoubleBlind").
- Setup: Mixtral-8x7B, greedy decoding, batch 1. Ten nodes: main (1x RTX 3090) + shadow (2x RTX 3090, quantized INT8 model) + 8 workers (1x RTX 3090 each, 192 GB DRAM), AMD R7-7700, 1 Gbps Ethernet. Baselines run on an 8x RTX 3090 server with EPYC 7K62 and 512 GB DRAM; offloading baselines use one GPU. Test: 60 Alpaca prompts (30 with 16 input tokens, 30 with 128), outputs 64 or 256. Footnote 1: "We do not use SSD/NVMe storage due to their large access delays."
- Numbers (Table 2):
  - Decode throughput avg tok/s: Transformers 4.8900, OD-MoE 3.6925, AdapMoE 3.1300, Mixtral-Offloading 2.2375, llama.cpp (CPU, all experts in DRAM) 0.8225, HOBBIT 0.7850, MoE-Infinity 0.6875. Ratios (mine): OD-MoE / Transformers = 0.755; / llama.cpp = 4.49x; / Mixtral-Offloading = 1.65x; / MoE-Infinity = 5.37x; / HOBBIT = 4.70x; / AdapMoE = 1.18x.
  - TTFT avg ms: OD-MoE 2244.3, Transformers 417.0, llama.cpp 4288.4, AdapMoE 1387.0, Mixtral-Offloading 1845.0, MoE-Infinity 6118.7, HOBBIT 5679.7.
  - GPU memory: OD-MoE 60 GB (7 main + 45 shadow + 1 per worker x 8), Transformers 180 GB.
  - Recall (Fig 3, Eq 3): 99.94% / 97.34% / 95.67% for FP16 / INT8 / NF4 shadow; Table 1 prior: AdapMoE 0.86, DAOP 0.84, HOBBIT 0.91 recall; Mixtral-Offloading ~0.80 and fMoE <0.85 cache hit.
  - Ablation (Fig 8, 9): token and KV alignment every iteration is best; no-shadow random prefetch ~25% recall.
- Catalog row numbers: no numbers in the row; "network-distributed expert loading is published": accurate. First-pass numbers: all correct.
- Conditions: Mixtral top-2/8 (8 experts, 340 MB each), full precision, 1 GbE LAN, experts in worker DRAM (not disk), eight GPU workers loading experts in parallel to multiply PCIe bandwidth, a second quantized model as predictor.
- Transfers to us: no. GPU workers, DRAM tier, 8-expert model, shadow predictor needs another full model. The one transferable idea (prediction several layers ahead with a quantized shadow model) needs the extra compute our CPU does not have. Their own footnote dismisses SSD, which our regime depends on.
- Quality loss: none for OD-MoE (full precision); baselines lose quality (their Table 2 part iii, e.g. GSM8K 64.14 OD-MoE vs 56.00/60.00/35.00/22.00 offloading baselines; MMLU 70.34 vs 49.82-55.26).
- Limitations stated: shadow node GPU memory, network delay absent in single-server baselines.
- Contradicts ours: no. Their reproduction of MoE-Infinity at 0.6875 tok/s is far below the 3.1-16.7x claims of P-0211 (different model, hardware, batch of reproduction); worth knowing when quoting either.

## P-0214: "Understanding MoE Offloading" (SomeOddCodeGuy)

- Source: DEV Community post, 13 Oct 2025 (cross-post of someoddcodeguy.dev). Read whole page.
- Setup: explanatory; gpt-oss-120b (about 120B total, 5.1B active; 36 layers, 128 experts, top-4; dense about 1.5B + active experts about 3.5B; expert about 24M params; expert pool about 114.7B), llama.cpp `-ngl 99 --n-cpu-moe N` on a "24 GB GPU" gaming PC. No hardware measured, no benchmark, no tok/s.
- Content: with `--n-cpu-moe 20` the first 20 layers' experts stay in system RAM and the CPU computes them; activation vectors cross PCIe, experts are not copied to VRAM. Author corrects himself: first N layers, not last N (from code). Claims the bottleneck is PCIe latency (assertion, no measurement).
- Catalog numbers: the row says "llama.cpp RPC backend / mac-tensor (experts over network among Macs)": wrong. Nothing on RPC, mac-tensor, networks or Macs. First pass correct.
- Transfers: no (CPU-GPU split, DRAM resident). It does confirm that stock llama.cpp MoE offload computes CPU-side rather than streaming weights, which is what we assume.
- Quality loss: n/a. Ablation: none. Limitations: author says finding the right N is trial and error.
- Contradicts ours: no. Replace the source (llama.cpp RPC README, or Geerling's llama.cpp TCP numbers in P-0216).

## P-0216: Geerling, "1.5 TB of VRAM on Mac Studio - RDMA over Thunderbolt 5"

- Source: jeffgeerling.com blog, 18 Dec 2025 (the catalog URL slug "15-tb" is the real URL). Text read; the three result charts (Qwen3 235B, DeepSeek V3.1, Kimi K2 Thinking) downloaded and read as images.
- Setup: four M3 Ultra Mac Studios (two 512 GB, 32 CPU cores, 8 TB SSD, $11,699 each; two 256 GB, 4 TB SSD, $8,099 each; about $40,000 total, loaned by Apple), full-mesh Thunderbolt 5 (post: "50-60 Gbps real-world"), macOS 26.2 RDMA (latency 300 us to under 50 us, a statement in the post), Exo 1.0 (RDMA) vs llama.cpp RPC (TCP). Prerelease software, stability problems noted. Batch size, quantisation and context lengths not stated. A llama.cpp 2.5 GbE vs TB5 comparison exists but is not tabulated in the post.
- Numbers (bar charts), tok/s, Exo(RDMA) vs llama.cpp(TCP), nodes 1 / 2 / 4:
  - DeepSeek V3.1 671B: Exo 21.1 / 27.8 / 32.5; llama.cpp 14.8 / 16.0 / 14.6. Exo 4 nodes / 1 node = 1.54x; Exo / llama.cpp at 4 nodes = 2.23x.
  - Qwen3 235B A22B: Exo 19.5 / 26.2 / 31.9; llama.cpp 20.4 / 17.2 / 15.2. Exo 4 / 1 = 1.64x; llama.cpp gets slower with nodes (0.75x).
  - Kimi K2 Thinking 1T A32B: Exo 21.6 (2) / 28.3 (4); llama.cpp 18.5 (2) / 16.4 (4); no single-node bar (does not fit). Exo / llama.cpp at 4 nodes = 1.73x.
  - Text: "32 tokens per second on the full cluster" for Qwen; "around 30" for Kimi.
- Catalog row numbers: DeepSeek 21.1 / 27.8 / 32.5 on 1 / 2 / 4 nodes: correct. Kimi about 28 on four nodes: correct (28.3). First pass marked both wrong or unconfirmable.
- Conditions: model resident in the union of node RAMs (each node holds a slice), tensor/pipeline split across nodes with RDMA, 819 GB/s class unified memory per node, GPU-class Apple compute. Scaling is bandwidth-per-token aggregation (21.1 tok/s of 671B A37B on one node).
- Transfers to us: no. No thin client pulls weights per token; our client has 8-32 GB and weights are streamed. The only portable fact is link capability: TB5 50-60 Gbps (about 6-7.5 GB/s) which is in the 2.8-5 GB/s range our E.md assumed for "TB-class / RDMA", so our row assumptions are plausible but still unmeasured by us.
- Quality loss: none discussed. Ablation: Exo vs llama.cpp across node counts only. Limitations stated: prerelease software, crashes with RDMA, 4 nodes max (no TB5 switch), macOS management pain, HPL over TB crashes.
- Contradicts ours: no. Supports the E.md remark that the cluster works because compute and bandwidth are large.

## P-0217: row says OD-MoE; URL is FIRM-MoE

- Paper at the URL: "FIRM-MoE: Fine-Grained Expert Decomposition for Resource-Adaptive MoE Inference", Keyu Chen, Qihang Zhou, Bin Qian, Zhenyu Wen, Wenchao Meng, Shibo He (Zhejiang University / ZJUT). AAAI-26, vol 40 no 24, pp 20190-20198, published 2026-03-14, DOI 10.1609/aaai.v40i24.39106. Read from the AAAI PDF. The row's described paper (OD-MoE, ten-node testbed) is P-0213 (arXiv 2512.03927); read there.
- FIRM-MoE setup: one RTX 3090 24 GB, 32-core CPU, 64 GB DRAM, PCIe 4.0 32 GB/s; Qwen1.5-MoE-A2.7B, Qwen3-30B-A3B, DeepSeek-MoE-16B-Base, DeepSeek-V2-Lite; batch 1; expert cache limits 128-896 experts; baselines AdapMoE (AP), MoE-Infinity (MI), DeepSpeed (ZeRO-Infinity), Fiddler, llama.cpp, and a Mixtral-offloading style baseline (BB). Method: splits each expert into sub-experts (prefetch only the first linear layer, n=1 or 2), multi-layer expert prediction, resource-adaptive preload search.
- Numbers: abstract "up to 1.5x speedup and 2.8x memory savings in decoding" (the body: up to 1.8x over BB at a 128-expert cache and 1.6x at 768 on Qwen1.5-MoE; under 30% latency increase when cutting cache from 768 to 128, a 2.85x memory reduction; Table 1 fine-grained prefetch 1.24-1.41x; average 1.31x throughput gain over Fiddler/llama.cpp in Fig 5). The abstract's "up to 1.5x" is thus not the same as the body's "up to 1.8x" (different baselines).
- Catalog row numbers: no numbers. Row's description (ten-node OD-MoE) does not match this URL: wrong source. First pass: correct.
- Transfers: no. Single GPU with PCIe 4 DRAM tier, 64 GB host RAM, small models (14-30B), batch 1, lossless (orthogonal to quantisation). No SSD, no network.
- Quality loss: none claimed (lossless prefetching). Ablation: fine-grained vs coarse (Table 1), granularity n (Table 2), prediction strategies (Fig 7). Limitations: prediction accuracy sensitivity, search cost.
- Contradicts ours: no.

## P-0218: Wi-Fi broadcast paper (catalog says WDMoE)

- Paper: "Broadcast Rate Limits in Wi-Fi: A Forgotten Bottleneck for Collaborative Edge LLM Inference", Liujianfu Wang, Yuyang Du, Shiqi Xu, Soung Chang Liew (CUHK). arXiv 2608.02341v1, 3 Aug 2026. Read in full. It is not WDMoE.
- Setup: Qwen3-30B-A3B (8 of 128 experts), main node + 8 worker nodes + 1 prediction node, each node one RTX 3090, 10 Gbps LAN (wired); wireless test with an ASUS RT-BE96U Wi-Fi 7 router at 6 GHz 320 MHz, Qualcomm QCNCM865 NICs at about 1 m; NS-3 simulation at 1, 2, 5 m; rates emulated on the wired testbed with traffic control. Metric: expert-layer time at the main node; embeddings are broadcast, results are gathered.
- Numbers: wired expert layer time UDP 2.56 ms (Shadow-e4n16), 72% of NCCL (Fig 6a), about 1.4x vs NCCL and TCP for all three predictors; theoretical max 16/9 = 1.78x. Wireless (Fig 8): TCP 7.20 ms, UDP 32.01 ms, NCCL 471.73 ms (NCCL about 100x its wired time; UDP 12.5x its wired time, 32.01/2.56 = 12.5). NS-3 optimum (Fig 9): MCS 8 at 1 m = 3459 Mbps, MCS 5 at 2 m = 2306 Mbps, MCS 4 at 5 m = 1729 Mbps; ratios to 54 Mbps (mine): 64.1x, 42.7x, 32.0x; layer times 4.65, 5.32, 8.72 ms (the figure labels).
- Catalog row numbers: "Wi-Fi broadcast rates around 54 Mbps are the bottleneck": correct for broadcast frames. "Experts live on worker nodes and activations travel, not weights": correct. Row names WDMoE: wrong paper.
- Conditions: embeddings (small) are broadcast to GPU workers that already hold experts; inactive experts are kept in GPU/CPU memory "or SSD" (a one-line mention citing HOBBIT/AdapMoE).
- Transfers to us: no. Activation traffic, not weight traffic; 54 Mbps broadcast cap is irrelevant for a unicast weight stream, though 54 Mbps (6.75 MB/s) is also a reminder that Wi-Fi unicast would be far below the 0.117 GB/s 1 GbE row in E.md.
- Quality loss: none. Ablation: three predictors (little difference), UDP vs TCP vs NCCL. Limitations: wireless results via simulation and emulation; firmware cannot be changed.
- Contradicts ours: no.

## P-0219: fungies.io multi-node guide

- Source: "How to Set Up Multi-Node Local LLM Inference: The Complete 2026 Guide", Dawid Wozniak, fungies.io, 1 July 2026 (a payments vendor's blog). Read all text.
- Content: vLLM+Ray, llama.cpp RPC, NVIDIA Dynamo; 10GbE minimum; Llama 3.1 70B benchmark table (one card 45 tok/s, 2x4090 over 10GbE 38 tok/s, 4x4090 52 tok/s; unsourced); "llama.cpp RPC: CPU-only 2-5 tok/s for 70B, mixed CPU/GPU 10-20 tok/s, 405B at 2 tok/s". Mentions MoE only in the Dynamo section ("MoE models like Mixtral and DeepSeek") and a VRAM table (Mixtral 8x22B 141B, DeepSeek-R1 671B, Qwen3-235B). Table error: Llama 3.1 405B FP16 810 GB called "ten RTX 4090s" (10x24 GB = 240 GB).
- Catalog numbers: "200B+ MoE at Q4 about 4 to 8 tok/s across 10 Gbps by shipping activations; per-token sync about 16 ms": absent from the page. First pass correct.
- Transfers: no usable data. No quality, no ablation, no limitations section.
- Contradicts ours: no. Replace by the llama.cpp RPC README and Geerling's llama.cpp TCP bars (P-0216).

## P-0221: Elbow-based MoE routing

- Paper: "Elbow-Based MoE Routing: A Training-Free Inference Time Plugin for Expert Selection", Robin Pan, Raymond Liu, Daniel Fang, Adelina Andrei, Rosa Wu (Harvard). ICLR 2026 Test-Time Updates (TTU) workshop; arXiv 2608.04401v1, 5 Aug 2026. Read in full (appendix too).
- Setup: OLMoE-1B-7B-0924-Instruct (64 experts, top-8, 16 layers), cap K=8, Kneedle-style elbow on sorted router probabilities, one NVIDIA H200, latency = wall-clock around the MoE block forward pass with cuda synchronize (batch and sequence length not stated; FLOPs formula uses N = batch x seq). Six multiple-choice benchmarks.
- Numbers (Table 2): mean k 7.576-7.655 (overall 7.615), FLOPs 5.37e8 to 4.81-4.92e8; latency base -> elbow ms: ARC-Easy 11.203 -> 10.497, ARC-Challenge 11.531 -> 10.974, MMLU 13.287 -> 12.571, HellaSwag 14.630 -> 13.960, PIQA 13.398 -> 12.944, WinoGrande 9.659 -> 8.948. Per-benchmark reduction (mine): 6.3, 4.8, 5.4, 4.6, 3.4, 7.4%; mean 5.31% (matches 5.3%). Accuracy base -> elbow: 77.82 -> 78.24, 61.86 -> 62.29, 51.74 -> 51.75, 47.02 -> 46.86, 73.18 -> 72.85, 50.36 -> 50.28 (max drop 0.33). 99.7% of 2M router curves have sharp elbows (angle at most 135 degrees; Fig 3). Load balance: mean l1 change 0.034, top-1 share -2.16%, CV -2.83% (Table 1; note the text says CV increases 2.83%, Table header shows negative, an inconsistency).
- Catalog row numbers: "per-token adaptive k, only 5.3% latency reduction at no accuracy loss": correct (accuracy within 0.33 points, some higher).
- Conditions: tiny change (k from 8 to 7.6), GPU, MoE-block time, one small model.
- Transfers: no. It removes 5% of expert activations; our proposal is 10 to 4 (60%) with 2-bit. Elbow can only prune, never below its detected elbow; tail randomisation (Fig 2) shows only that post-elbow experts do not matter. Nothing about k=4.
- Quality loss: measured, none beyond 0.33 points at k=7.6.
- Ablation: tail randomisation vs pre-elbow randomisation on MMLU (Fig 2). Limitations: none explicit; single model, workshop paper.
- Contradicts ours: no (our +45.6% perplexity at top-4 concerns a far larger cut; the paper does not test it).

---

# Per-family prior art (read, not just snippets)

Searches: arXiv API queries on expert caching + SSD/NVMe, cache replacement, remote/RDMA, wireless distributed MoE, dynamic routing/skipping, plus WebFetch-free reading of every hit listed. "Read" = PDF text pulled and read for setup and tables as described.

## Family 14: per-sequence (ARC) expert cache

- FlashMoE, Kim, Lee, Han, Yoo, Kim (KAIST/Chung-Ang/Yonsei), arXiv 2601.17063v2 (read, results and setup). Experts on SSD, ML-based replacement combining recency and frequency (a tiny per-layer FFN, 113 KB, trained from Belady-labelled traces); platform Ryzen 5 9600X, RTX 5070 Ti 16 GB, 4x16 GB DDR5-6000, SK hynix P51 PCIe 5.0 NVMe (7.4 GB/s read), RAM limited to about 1 GB free; OLMoE-1B-7B and Qwen3-30B-A3B. Hit rate: vs LRU +21%, vs LFU +51%, up to +28% vs ARC and +21% vs LeCaR (Fig 6); Fig 2: Belady 86% vs LRU 73% hit; evicted-expert reuse 34.2% for LRU vs 0.1% Belady. Tok/s: +22% (OLMoE) and +7% (Qwen3-30B-A3B) over LRU, "the best-performing heuristic" (Fig 7b,e); 2.6x over llama.cpp/Fiddler/DAOP/HF in the abstract; prefill 2.5x over llama.cpp. GPU cache above SSD, not CPU-only. This is the closest prior art to an SSD expert cache and it tests ARC.
- Edge0 (arXiv 2609.18063, below under family 16) also uses an LRU hot set over SSD on a Mac and reports adjacent tokens agree on only about a quarter of a layer's experts.
- SpecMD, Hoang, Jaiswal, Samragh, Cho (Apple), arXiv 2602.03921 (read, abstract/setup/tables located): A100 80 GB emulation at 5 GB/s measured bandwidth, int4/int8 experts; Least-Stale eviction cuts collision misses up to 85x vs LRU at 1% capacity, 88-92% hit rates, TTFT improvements 10.7-34.7% on OLMoE at 5% cache (0.6 GB), 14.8-44.1% across baselines (Table 2). Prefill/TTFT focus, not decode; no ARC.
- In-depth Analysis on Caching and Pre-fetching in MoE Offloading, Lin et al. (CMU), arXiv 2511.05814: downloaded; LFU optimisation vs LRU on GPUs; I did not re-verify the first-pass tok/s figures (only first pass numbers exist).
- ProMoE, arXiv 2410.22134 (downloaded; text confirms LRU baseline and static baseline; first-pass numbers 1.66x / 1.09x not re-verified by me).
- Mixture of Cache-Conditional Experts (Skliar et al., TMLR June 2025, arXiv 2412.00099; abstract read): cache-aware routing, over 50% fewer cache misses, perplexity +0.1-3%, 2x speedup on mobile hardware; changes the model's routing (lossy, training-free).
- Prefetch-Steered Routing (arXiv 2610.34721, 26 Oct 2026; abstract and intro read): OLMoE with 25-50% experts resident on RTX 4090, 2.2-2.3x faster than exact routing with LRU, +0.1 to +0.5 mean benchmark points; lossy routing.
- ReMoE (arXiv 2605.27081; abstract read): router fine-tuning for expert reuse, +26% reuse, TPOT -43.6 to -49.8% under llama.cpp on Jetson Orin NX (UFS storage), 1.77-1.99x decode, needs training.
- Cache-Aware Joint Router Adaptation (arXiv 2609.04895; abstract read): post-training temporal/spatio router for residency; hit rate +1.15-18.03 points, traffic -4.6 to -53.3% vs strongest prefetch baseline (Qwen3).
- DALI (arXiv 2602.03495; abstract read): GPU cache policy using workload dynamics, CPU-GPU assignment, residual-based prefetch for local PCs.
- SSD Offloading for LLM MoE Weights Considered Harmful in Energy Efficiency, Kyung, Yun, Ahn, arXiv 2508.06978 (read): SSD decode energy up to about 12x vs HBM, 4.9x vs HBM and 3.1x vs DDR for DeepSeek-R1; SSD reads are 80% of per-token energy. Relevant to a laptop-battery story; not a speed paper.
- From memory (not re-read): Eliseev and Mazur 2312.17238, Fiddler 2402.07033, Pre-gated MoE, ExpertFlow (arXiv 2510.26730 appeared in search), AdapMoE, HOBBIT (2411.01433, downloaded), EdgeMoE.

What is new in our variant (ARC on per-sequence traces): ARC itself as an expert-cache policy is not new (FlashMoE evaluated it as a baseline against LRU, LFU, LeCaR); neither is per-request caching (MoE-Infinity) or learned/predictive replacement (FlashMoE, SpecMD, Cache-Aware Joint Router). What is not in any paper found: a CPU-only llama.cpp-style engine, a 512-expert top-10 model, and caches of 2-4 expert footprints (8-32 GB) over NVMe, with decode measured at batch 1. Closest papers: FlashMoE (SSD tier, GPU cache) and Edge0 (SSD, Mac, LRU). Our measured null result for ARC is consistent with FlashMoE's note that LRU is the best heuristic.

## Family 15: all experts in a second machine's RAM

- WDMoE, Xue, Sun, Chen, Tao, et al. (SJTU / CUHK-SZ), arXiv 2405.03131 (May 2024) and journal version 2411.06681 (Nov 2024), abstracts read: gate and preceding layers at a base station, experts on mobile devices (Jetson testbed), expert selection by performance and latency; activations travel, weights stay. Same pattern as P-0218 and OD-MoE. Not read beyond abstract/intro.
- AirMoE (2608.22932), SiftMoE (2603.23888), Logit-Aware MIMO AirComp (2610.03741): wireless distributed MoE inference; titles/abstract hits only, not read; same activation-travel pattern.
- OD-MoE (P-0213) and FIRM-MoE (P-0217): read; local DRAM to GPU, not remote weight pull. The OD-MoE group's follow-up (P-0218) concerns broadcast of activations.
- From the first pass (not re-read here): MLX issue #3208 proposing demand-paged expert fetch over TB5 RDMA at about 5 GB/s, unbuilt; TrimMoE 2608.00573, OrderMoE 2607.17154.
- From memory: Petals, exo, llama.cpp RPC, DeepSpeed-MoE expert parallelism.
- No paper found (arXiv API: "remote memory expert RDMA", "disaggregated memory CXL MoE", "decentralised consumer experts") implementing a thin client that pulls expert weights per token from a second machine's RAM.

What is new in our variant: weight-pull from a second host's RAM by a thin client at batch 1. Everything published moves activations to nodes that hold the weights; Geerling's cluster keeps the model resident across nodes. Closest: MLX issue proposal (unbuilt) and OD-MoE (parallel PCIe loads from local DRAM). Our arithmetic (E.md) says it only beats local NVMe at RDMA-class bandwidth.

## Family 16: active experts 10 to 4 plus 2-bit

- Edge0, "The Other Half of the Memory Wall: Serving 35B MoEs from SSD with Trained Routing Prediction", Yu Lin et al. (AutoArk), arXiv 2609.18063v2, 17 Sep 2026 (read results, limitations). Qwen3.6-35B-A3B (40 layers x 256 experts + shared, native K=8, released at K=4) and Ling 3.0 tiny (24 x 128, K=8); int4 affine g64; MLX on Mac mini M4 Pro 24 GB and MacBook M2 16 GB; SSD-streamed experts, LRU hot set; trained per-layer prerouter predicting next token's routing; recovery LoRA (r=16). Results: 20.4 tok/s decode in 2.9 GiB active memory (K=4 profile) vs 3.9 tok/s for vanilla mlx-lm with weights resident (18.2 GiB); Table 3 (16 GB M2, 18.4 GiB checkpoint): K=2/4/8 on-demand 4.8/3.5/1.8 tok/s vs prerouter 8.6/6.4/3.3 (+80/82/84%); narrowing K=8 to K=4 at the same cache nearly doubles decode (3.3 to 6.4). Quality vs fp16 (Table 2): average 79.2 vs 83.2 (35B), 69.9 vs 72.7 (8B); AIME 86.6 vs 92.7. Limitations: batch 1, CPU-side graph building floor 44 ms/step, loss concentrated in long reasoning. Closest published work to our regime (Mac, SSD, batch 1, k reduction, tiny cache), but uses trained prerouter + LoRA and a 256-expert top-8 model, not 512 experts top-10 and not CPU-only llama.cpp.
- Distribution-Consistent Inference for Dynamic Sparse MoE, Kim, Soro, Hwang (KAIST), arXiv 2609.09241 (read table lines): Qwen3-30B-A3B k=8 -> 4: average 74.44 -> 66.49 (72.17 with their fix); vLLM on one RTX PRO 6000: 910.71 -> 1156.26 tok/s (1.27x, mine) at k=4.
- Don't Skip the Baseline, Steimle et al. (TUM/Tensordyne), arXiv 2610.28369 (abstract read): static reduced k is hard to beat by four dynamic skipping methods on Gemma4-26B-A4B and Qwen3-30B-A3B at matched compute; so Elbow-style dynamic k is not clearly better than a plain static k.
- ACE, Xu et al., arXiv 2609.05228 (abstract read): training-free expert skipping; at 50% skipping on Qwen3.6-35B-A3B, perplexity -7.96% and +4.15 points vs strongest competitor; up to 2.25x prefill and 1.41x decode speedups.
- From the first pass (not re-read): ZEDA 2605.18643 (1.20x end to end), OEA, SERE (2602.07616), HookMoE, MC-MoE.
- From memory: NAEE, dynamic top-p routing (Huang et al., NeurIPS 2024).

What is new in our variant (10 to 4 plus 2-bit, in a disk-bound regime): reducing k at inference is well known and its measured gain is about 1.3x on GPUs (Distribution-Consistent: 1.27x) with clear quality cost; the SSD case with large benefit comes from cache footprint (Edge0 1.94x at fixed cache). New would be a 512-expert top-10 model on a CPU-only engine at 8-32 GB; nothing found covers it. Closest: Edge0 (read in full detail above) and Distribution-Consistent Inference.
