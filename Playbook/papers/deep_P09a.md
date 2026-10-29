# deep_P09a: raw-text re-check of rows P-0186..P-0198 (selected 11 rows)

Source of text: arXiv PDFs via pdftotext (full text), GitHub raw README, HTML pages stripped with python3 -I. Counts: raw text read 10, partly/UNREAD 1 (P-0194, body blocked). Where I used search-result summaries instead of raw pages it is said in the entry.


---

## P-0186: OD-MoE: On-Demand Expert Loading for Cacheless Edge-Distributed MoE Inference. Liujianfu Wang, Yuyang Du, Yuchen Pan, Soung Chang Liew, Jiacheng Liu, Kexin Chen (CUHK). arXiv:2512.03927v1 [cs.DC], 3 Dec 2025

- **Version/text read:** arXiv 2512.03927v1 (PDF, pdftotext, full text) (raw_text_read=True)
- **Setup:** Mixtral-8x7B (8 experts/layer, top-2, 32 layers), full-precision experts. Ten nodes on a 1 Gbps Ethernet LAN: main node (1x RTX 3090, 128 GB DRAM, non-experts), shadow node (2x RTX 3090, 128 GB, INT8 copy of the whole model, 45 GB), 8 worker nodes (1x RTX 3090, 192 GB DRAM each; each worker holds <1 GB of GPU memory, experts are fetched CPU DRAM -> GPU just in time over PCIe). All nodes AMD R7-7700. Baselines run on a separate server: 8x RTX 3090, EPYC 7K62, 512 GB DRAM (Transformers needs 8 GPUs; offloaders use one GPU). Batch 1 greedy decode; HOBBIT test set: 60 Alpaca prompts, input 16 or 128 tokens, output 64 or 256. Software: PyTorch/Transformers-based; llama.cpp CPU baseline. No disk/NVMe involved: experts come from DRAM.
- **Numbers (table/figure, baseline, ratio):**
  - OD-MoE decode throughput (avg of 4 configs): 3.6925 tok/s | baseline: Transformers all-experts-cached 4.8900 | ratio: 0.755 (paper: 75.51%) | where: Table 2(i), p.11
  - OD-MoE vs llama.cpp (CPU, all experts in DRAM): 3.6925 | baseline: llama.cpp 0.8225 | ratio: 4.49x | where: Table 2(i), p.11; text Sec 4.4
  - OD-MoE vs Mixtral-Offloading: 3.6925 | baseline: 2.2375 | ratio: 1.65x | where: Table 2(i)
  - OD-MoE vs MoE-Infinity: 3.6925 | baseline: 0.6875 | ratio: 5.37x | where: Table 2(i)
  - OD-MoE vs HOBBIT: 3.6925 | baseline: 0.7850 | ratio: 4.70x | where: Table 2(i)
  - OD-MoE vs AdapMoE: 3.6925 | baseline: 3.1300 | ratio: 1.18x | where: Table 2(i)
  - TTFT (prefill) OD-MoE avg: 2244 ms | baseline: Transformers 417 ms; Mixtral-Offloading 1845; MoE-Infinity 6119; HOBBIT 5680; AdapMoE 1387; llama.cpp 4288 | ratio: 5.4x slower than Transformers; slower than Mixtral-Offloading and AdapMoE | where: Table 2(i)
  - Expert-activation recall of the shadow-model predictor (SEP): FP16 99.94%, INT8 97.34%, NF4 95.67% | baseline: AdapMoE/DAOP 86%, HOBBIT ~84% (Table 1 estimates) | ratio: n/a | where: Fig 3, Table 1, Sec 4.3
  - Recall without alignment (INT8 shadow, no token/KV alignment): ~100% falling to ~30% by token 256, mean ~45%; random prefetch ~25% | baseline: with per-iteration alignment 97% | ratio: n/a | where: Sec 4.1 text on Fig 8 (Case 4/5), Fig 3b
  - GPU memory: 60 GB total (7 main + 45 shadow + 8x1 workers) | baseline: full model on 8 GPUs (~180 GB listed for Transformers) | ratio: 1/3 of full-precision (paper) | where: Table 2(ii)
  - Output throughput incl. prefill: 3.3700 tok/s | baseline: Mixtral-Offloading 2.1725, MoE-Infinity 0.6675, HOBBIT 0.7575, AdapMoE 3.0350, Transformers 4.8425, llama.cpp 0.7975 | ratio: 1.55x / 5.05x / 4.45x / 1.11x vs the four offloaders | where: Table 2(i)
- **Catalog / first-pass numbers checked:**
  - catalog one-liner: many low-cost edge nodes, under 1 GB GPU memory each -> yes (workers only). Each worker <1 GB, but the shadow node uses 45 GB and main 7 GB; total 60 GB
  - catalog one-liner: shadow model predicts experts -> yes. INT8/NF4/FP16 emulative shadow copy ("SEP"), aligned every iteration
  - catalog one-liner: "Reaches about 7..." (text cut off at 204 chars in the CSV) -> absent. Cannot be checked; paper reports 75.5% of the all-GPU-cached speed, 3.69 tok/s. If the cut text was "about 75%" it is right.
  - first pass: 3.69 tok/s = 75.5% of 4.89; 4.49x llama.cpp 0.82; recall 99.94% FP16 -> yes. 3.6925 / 4.8900 / 0.8225 / 99.94%
  - first pass: HOBBIT 0.79, MoE-Infinity 0.69, AdapMoE 3.13, Mixtral-Offloading 2.24 -> yes. 0.7850 / 0.6875 / 3.1300 / 2.2375
  - first pass: INT8 recall collapses to ~30% by token 256 without alignment -> yes. Text says ~30% by token 256 (mean ~45%) for the no-alignment INT8 case
  - first pass: "experts from DRAM not disk" -> yes. Workers have 192 GB DRAM; experts loaded CPU DRAM -> GPU
- **Transfers to our regime (CPU-only, 8-32 GB, NVMe, batch 1, 125B 512-expert top-10):** no. Assumes GPU nodes (RTX 3090 each), batch 1 but 8 experts/layer with top-2 so one expert fits one worker; experts sit in 128-192 GB DRAM per node, never on disk; the LAN carries only activations (16 KB/token/layer) and alignment KV (256 KB/token). The predictor needs a second INT8 copy of the whole model on a 2-GPU node plus per-token alignment traffic, impossible on an 8-32 GB CPU machine; with 512 experts, top-10 the "one expert per worker" parallelism does not apply. The baseline llama.cpp number (0.82) is a CPU run on another server with all experts resident, so 4.49x is a GPU-vs-CPU gap, not a method gain.
- **Quality loss:** None for OD-MoE itself (full-precision experts; Table 2(iii) shows it matches Transformers/llama.cpp on six benchmark groups). Quantised offloading baselines lose quality (their benchmark rows drop, e.g. BigBenchHard and Big Code in Table 2(iii)).
- **Ablation:** Yes, Fig 8 (six cases: shadow node with/without token and KV alignment, random prefetch, load-after-routing) and Fig 9/10 (alignment periods). Token alignment matters more than KV alignment; without the shadow model the speed falls to the I/O-bound regime. No cache-vs-prefetch ablation because there is no cache.
- **Limitations (stated):** Authors state: predictor accuracy decays unless aligned; alignment trades accuracy against cost; best period depends on GPU (RTX 3080 optimum differs). They do not discuss disk, non-8-expert models, or batch>1 (decoding is batch 1 only).
- **Contradicts our findings (pubtech/E.md):** No contradiction. Its high-recall predictor (99.9%) needs a shadow model; our measured predictor precision (0.55) is a cheap-predictor figure and OD-MoE itself shows recall ~25-45% when the shadow is not aligned, consistent with cheap predictors being weak. pubtech E.md wording "about 75% of a GPU-cached Mixtral speed" is correct.

---

## P-0187: Running DeepSeek V3 671B on M4 Mac Mini Cluster, "Day 2 - 12 days of EXO" (exo labs blog, no author, no date visible on the page)

- **Version/text read:** https://blog.exolabs.net/day-2/ (HTML, stripped with python3 -I; full page text) (raw_text_read=True)
- **Setup:** 8x Mac mini M4 Pro, 64 GB each (512 GB total unified memory), 4-bit models. DeepSeek V3 671B (37B active, MoE). All weights resident in the cluster RAM. Batch 1 implied (the post argues about batch_size=1). No context length, no network, no software version, no parallelism method stated. Page text discusses memory bandwidth vs FLOPS and why MoE is cheap per token if all parameters stay "hot" in memory.
- **Numbers (table/figure, baseline, ratio):**
  - DeepSeek V3 671B 4-bit: TTFT 2.91 s, 5.37 tok/s | baseline: Llama 3.3 70B 4-bit: 3.14 s, 3.89 tok/s | ratio: 1.38x faster decode than the dense 70B on the same cluster | where: results table at top of the page
  - Llama 3.1 405B 4-bit: TTFT 29.71 s, 0.88 tok/s | baseline: DeepSeek V3 5.37 | ratio: 6.1x slower than DeepSeek V3 | where: results table
- **Catalog / first-pass numbers checked:**
  - 5.37 tok/s for 671B at 4-bit across 8 nodes -> yes. 5.37 tok/s, TTFT 2.91 s, 8x M4 Pro 64GB
  - catalog: "uses pipeline or tensor splits, not an expert-server design" -> absent. The page never names the parallelism strategy; this statement is not from the page (from memory exo uses ring/pipeline sharding of layers at this date).
  - first pass: TTFT 2.91 s, Llama 405B 0.88, 70B 3.89 -> yes. as above
  - first pass: "all 671B weights resident in 512 GB" -> yes. Page: "load all of the 671B parameters across a cluster of Macs"
- **Transfers to our regime (CPU-only, 8-32 GB, NVMe, batch 1, 125B 512-expert top-10):** no. All weights live in unified memory (512 GB); nothing is streamed from disk, so it only shows that a MoE with every expert resident decodes at dense-30B-like speed. Our regime is 8-32 GB RAM with experts on NVMe. The cluster costs 8 machines.
- **Quality loss:** 4-bit quantisation only; no evaluation shown.
- **Ablation:** none
- **Limitations (stated):** None stated; marketing blog, no methodology, no repeats.
- **Contradicts our findings (pubtech/E.md):** No.

---

## P-0188: Accelerating Edge Inference for Distributed MoE Models with Latency-Optimized Expert Placement ("Prism" in v4). Tian Wu, Liming Wang, Zijian Wen, Xiaoxi Zhang, Xu Chen, Jingpu Duan, Xianwei Zhang, Jinhang Zuo (SYSU, CityU HK, Peng Cheng Lab). arXiv:2508.12851v4 [cs.DC], 15 Apr 2026

- **Version/text read:** arXiv 2508.12851 latest (v4) PDF, full text. A search-result snippet shows the v1 abstract page titled the system "DanceMoE"; v1 not read. (raw_text_read=True)
- **Setup:** Testbed is ONE machine: 4x A100-40GB, 256 GB RAM, split into 3 simulated edge servers (1, 1, 2 GPUs) joined by a Docker network limited to 500 Mbps with tc. Models: Mixtral-8x7B (32 layers, top-2 of 8) and DeepSeek-V2-Lite (26 layers, top-8 of 64); GPU memory artificially capped to 70% (Mixtral) / 30% (DeepSeek). Datasets BIG-bench, MMLU-Pro, WikiText, TACO; output length 20 tokens or answer length; requests arrive as Poisson (10 s / 20 s mean inter-arrival). Baselines: Uniform, Redundance (own heuristic), SmartMoE placement module, EPLB (re-implemented by the authors for heterogeneity). Migration every 5 min. Plus an event-driven simulator for 4-256 GPUs and 100-1000 Mbps.
- **Numbers (table/figure, baseline, ratio):**
  - Serve latency, DeepSeek-V2-Lite, BigBench, total avg: 6.63 s (Ours) | baseline: EPLB 9.56 s; SmartMoE 19.39; Redundance 13.08; Uniform 21.66 | ratio: 1.44x vs EPLB (30.6% lower); 3.3x vs Uniform | where: Table II, p.8
  - Same, MultiData: 13.29 s | baseline: EPLB 14.93 | ratio: 1.12x (10.9% lower) | where: Table II
  - Mixtral-8x7B BigBench: 2.09 s | baseline: EPLB 2.16 | ratio: 1.03x (3.2% lower) | where: Table II
  - Mixtral-8x7B MultiData: 9.17 s | baseline: EPLB 9.36 | ratio: 1.02x (2.0% lower) | where: Table II
  - Table I motivation: Mixtral on 3 servers: Naive collaboration (random expert split, remote calls) avg 4.11 s | baseline: MoE-Infinity 5.19; MoE-Infinity with load balancing 5.03 | ratio: 1.26x vs MoE-Infinity | where: Table I, p.3
  - Migration effect, DeepSeek-V2-Lite: 6.73 s | baseline: static placement 7.48 s | ratio: 1.11x (10%) | where: Sec IV-B, Fig 7
  - Simulation: bandwidth: higher bandwidth lowers per-prompt time >55% at 4 GPUs, 35% at 256 GPUs | baseline: lowest bandwidth | ratio: n/a | where: Fig 8b, Sec IV-C
- **Catalog / first-pass numbers checked:**
  - catalog: places experts across edge devices to minimise network hops (same architecture class) -> yes (partly). Placement is activation-aware and entropy-driven to maximise local compute ratio under memory limits, plus runtime migration; it is about request locality across heterogeneous servers, not hop count
  - first pass: up to 30.6% lower latency (6.63 vs 9.56 s) -> yes. DeepSeek-V2-Lite BigBench only; Mixtral 2-3%
  - first pass: Mixtral 2.09 vs 2.16 s; ~2% on MultiData -> yes. 2.09 vs 2.16 (3.2%) on BigBench; 9.17 vs 9.36 (2.0%) on MultiData
  - first pass: "latency"  -> partly. Table II is serve latency under Poisson arrivals and includes queueing (values up to 48 s), not per-token decode latency
- **Transfers to our regime (CPU-only, 8-32 GB, NVMe, batch 1, 125B 512-expert top-10):** no. Multi-user serving across 3 GPU "servers" with queueing; experts live in GPU memory partitions, no disk; placement uses per-server request distributions (input locality across servers) which does not exist for one user on one laptop. Edge testbed is a single 4-GPU box with throttled network.
- **Quality loss:** None (placement only, exact computation).
- **Ablation:** Migration on/off (Fig 7), baselines Uniform/Redundance/SmartMoE/EPLB; no isolation of cache vs layout vs prefetch.
- **Limitations (stated):** Gains shrink with few experts (Mixtral ~2-3%); depends on workload locality; heterogeneity assumptions; edge environment simulated.
- **Contradicts our findings (pubtech/E.md):** No. Its small Mixtral gain (2-3%) and our held-out balanced split (1.266 -> 1.250 straggler) both say placement refinements recover little when routing is near uniform.

---

## P-0189: Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference (ExFlow). Jinghan Yao, Quentin Anthony, Aamir Shafi, Hari Subramoni, Dhabaleswar K. Panda (Ohio State). arXiv:2401.08383v2 [cs.LG], 17 Jan 2024 (IEEE conference style; venue not printed in the text)

- **Version/text read:** arXiv 2401.08383v2 PDF, full text (raw_text_read=True)
- **Setup:** GPT-style MoE models, top-1 gating: GPT-M 350M with 8/16/32/64 experts per layer (24 layers), also 32/40-layer variants and a GPT-XL (1.3B, 16 experts). Hardware: Wilkes3 cluster, nodes with 2x AMD EPYC 7763 and 4x A100-80GB (NVLink), dual-rail HDR200 InfiniBand; 1-8 nodes (4-32+ GPUs). Baseline: DeepSpeed-MoE expert parallelism (two Alltoalls per layer). Method: context-coherent expert parallelism (one Alltoall instead of two) plus an integer program for expert-to-GPU placement from routing traces on Pile (a few thousand tokens suffice); evaluated on Pile eval, affinity re-tested on C4, Dolma, Yelp (Table III).
- **Numbers (table/figure, baseline, ratio):**
  - Throughput speedup, GPT MoE-16, 2 experts per GPU: up to 2.2x | baseline: DeepSpeed-MoE | ratio: 2.2x | where: Fig 10, Sec V-E (p.9)
  - Speedup MoE-32 on 8 and 16 GPUs: 1.6x | baseline: DeepSpeed-MoE | ratio: 1.6x | where: Sec V-E
  - Speedup MoE-8 on 8 GPUs: 10% | baseline: DeepSpeed-MoE | ratio: 1.10x | where: Sec V-E
  - MoE-16 on 16 GPUs (1 expert per GPU): ~20% | baseline: DeepSpeed-MoE | ratio: 1.2x | where: Sec V-E
  - Cross-GPU routing / Alltoall latency reduction: up to 67% (56/65/67% over three hardware setups) | baseline: vanilla placement | ratio: 3.0x lower | where: Abstract and contributions list (p.2)
  - Share of runtime in Alltoall, vanilla: ~15% (1 node), ~63% (2 nodes), 76% (8 nodes) | baseline: n/a | ratio: n/a | where: Sec V-E, Fig 9
  - Affinity transfer: placement solved on Pile gives almost identical affinity on C4/Dolma/Yelp (normalised ~0.99-1.00) | baseline: n/a | ratio: n/a | where: Table III
- **Catalog / first-pass numbers checked:**
  - catalog: expert-parallel placement keeping the next layer's experts local; all-to-all is the bottleneck -> yes (incomplete). The method has two parts: single-Alltoall context-coherent EP AND affinity placement; most of the gain at 2.2x needs both
  - first pass: up to 2.2x throughput and up to 67% lower routing latency -> yes. matches abstract; 2.2x is the best single configuration (MoE-16, 2 experts/GPU)
  - first pass: Alltoall ~15% on 1 node, 63% on 2, 76% on 8 -> yes. as above
  - first pass: "GPT MoE 350M-1.3B top-1, A100 NVLink+IB" -> yes. Table II models 350M/470M/590M/1.3B, top-1 at inference
- **Transfers to our regime (CPU-only, 8-32 GB, NVMe, batch 1, 125B 512-expert top-10):** no. Gain is communication volume between GPUs in expert parallelism for batched serving; top-1 gating; tiny models. In our regime (one node pair at best, batch 1, top-10 of 512, bandwidth bound by NVMe) there is no Alltoall to remove. Useful signal: inter-layer routing conditional probabilities are strongly peaked and stable across datasets (Table III) - supports next-layer expert prediction for prefetch, which is the part that could transfer.
- **Quality loss:** None (exact; "without accuracy degradation").
- **Ablation:** ExFlow with vs without affinity placement (Fig 10: "ExFlow w/o affinity" vs "w. affinity") separates the single-Alltoall change from the placement; I did not extract the per-bar numbers (bar chart), so the size of the placement-only share is not quantified here.
- **Limitations (stated):** Requires pre-trained model profile (1000-3000 tokens); gain shrinks when each GPU holds one expert per layer or on 4 GPUs of one node (NVLink Alltoall cheap); trains on GPT-style top-1 only.
- **Contradicts our findings (pubtech/E.md):** Consistent: our held-out balanced split gained little (1.266 -> 1.250). Affinity helps when whole Alltoall volume is the cost; with top-10 of 512 per token and two nodes the straggler dominates, not communication.

---

## P-0190: Reddit post "Distributed inference worth the time?" by u/Karnitine (mirror: reddit.sentinel-team.org, snapshot 2026-09-04 21:20 UTC, post ~58 days old at snapshot)

- **Version/text read:** HTML snapshot, full text including the 7 captured comments (of 10 shown) (raw_text_read=True)
- **Setup:** Poster asks whether llama.cpp RPC over a Thunderbolt 4 (40 Gb/s) link between two PCs (9950X3D, 96 GB DDR5-6000, RTX 5090 32 GB; and Ryzen AI 395+ Max, 128 GB LPDDR) can run a Q4 MoE ("GLM-5.3-Flash" in one place, "GPT-5.3-Flash" in another) at 5+ tok/s rather than below 1 tok/s. He has tried nothing. No benchmark anywhere in the thread.
- **Numbers (table/figure, baseline, ratio):**
  - any measured throughput: none | baseline: n/a | ratio: n/a | where: whole page
- **Catalog / first-pass numbers checked:**
  - catalog: llama.cpp RPC places layers or experts on remote devices; no measured MoE numbers -> partly. No measured numbers: right. The placement claim is from ONE commenter (dionysio211): llama.cpp lets you "place" experts on different devices but "a syntactical trick rather than true expert sharding"; not from the poster.
  - first pass: two PCs over Thunderbolt untested -> yes. Thunderbolt 4, untested
  - thread pointer -> new. A commenter links arXiv 2608.19147 (Pipeline shards for Intel AI PC fleets); I read it: it is pipeline parallel on dense Llama, not MoE
- **Transfers to our regime (CPU-only, 8-32 GB, NVMe, batch 1, 125B 512-expert top-10):** no. Anecdote with zero data. Comments disagree: "network traffic will kill the gain" vs "even over 1G ethernet the loss can be negligible depending on how you do it" - both unsupported.
- **Quality loss:** n/a
- **Ablation:** none
- **Limitations (stated):** n/a (forum thread)
- **Contradicts our findings (pubtech/E.md):** No.

---

## P-0192: RDMA over Thunderbolt Lets Mac Studios Run Huge Local LLMs Faster Than Expected. Allan Witt, Hardware Corner, 19 Dec 2025 (reports Jeff Geerling and jakkuh tests)

- **Version/text read:** https://www.hardware-corner.net/mac-rdma-over-thunderbolt-llm/ (HTML, full text); numbers cross-checked in a raw read of technews.tw (25 Dec 2025) and in search-result summaries (not raw-read) of other write-ups (raw_text_read=True)
- **Setup:** 4x Mac Studio M3 Ultra (80-core GPU; two with 512 GB, two with 256 GB), Thunderbolt 5 mesh, macOS 26.2 RDMA, exo 1.0 (MLX, tensor sharding) versus llama.cpp over TCP (layer split). Models: Qwen3 235B-A22B 8-bit MLX (242 GB), DeepSeek V3.1 671B 4-bit MLX (378 GB), Kimi K2 Thinking 1T-A32B 4-bit (658 GB). Steady-state generation only; prefill/large context not benchmarked. All weights resident in unified memory. About $40k hardware.
- **Numbers (table/figure, baseline, ratio):**
  - Qwen3 235B exo+RDMA, 1 node -> 4 nodes: 19.5 -> 31.9 tok/s | baseline: llama.cpp TCP 20.4 -> 15.2 | ratio: exo 1.64x scaling; exo vs llama.cpp at 4 nodes 2.10x; llama.cpp 0.75x | where: Hardware Corner page: graphs only (no text numbers). 31.9 confirmed raw in technews.tw; 19.5, 20.4, 15.2 only from search-result summaries of secondary articles (not raw-read)
  - DeepSeek V3.1 671B exo+RDMA, 1 / 2 / 4 nodes: 21.1 / 27.8 / 32.5 tok/s | baseline: llama.cpp 2 nodes 18.5, 4 nodes 14.6 (ITHome only, via search summary) | ratio: 1.54x at 4 nodes | where: 32.5 confirmed raw in technews.tw; others from search summaries
  - Kimi K2 Thinking 4 nodes: ~28.3 tok/s (one report), ~30 (another) | baseline: n/a | ratio: n/a | where: search summaries only
  - Latency TCP vs RDMA: ~300 us -> <50 us | baseline: n/a | ratio: 6x | where: technews.tw (raw)
- **Catalog / first-pass numbers checked:**
  - Qwen3 235B: llama.cpp 20.4 falls to 15.2 on 4 nodes -> yes (secondary, not on the opened page). Not on the Hardware Corner page (graphs only); found in secondary write-ups per search summary; I could not reach Geerling's own post
  - exo+RDMA 19.5 -> 31.9 -> yes (31.9 raw confirmed in technews.tw; 19.5 from search summary). 19.5 -> 31.9
  - DeepSeek V3.1 21.1 -> 32.5 on 4 nodes -> yes. 21.1 / 27.8 / 32.5 on 1/2/4; 32.5 raw-confirmed
  - first pass: "catalog Qwen3 numbers not found; secondary: DeepSeek 21.1/27.8/32.5" -> wrong (partly). The Qwen3 numbers do exist in secondary coverage (20.4/15.2/19.5/31.9); first pass only looked at the one page
- **Transfers to our regime (CPU-only, 8-32 GB, NVMe, batch 1, 125B 512-expert top-10):** no. RDMA carries activations for tensor parallelism between machines that each hold part of the weights in RAM. Nothing is streamed from disk; the speed-up is from adding memory bandwidth (4 x ~800 GB/s unified memory) and a 50 us interconnect, not from caching. Needs TB5 Macs ($40k).
- **Quality loss:** None stated (8-bit/4-bit MLX models as is).
- **Ablation:** TCP vs RDMA at 1/2/4 nodes is the only isolation (transport), no expert-placement ablation.
- **Limitations (stated):** Article: early software, crashes, naming limits, only 4 nodes, no large-context test; MoE scales worse than dense.
- **Contradicts our findings (pubtech/E.md):** No. It supports our model that a cluster helps only when compute/bandwidth is pooled; note the same article says llama.cpp TCP gets slower with more nodes (decode latency per hop), matching P-0195.

---

## P-0193: Harvest: Opportunistic Peer-to-Peer GPU Caching for LLM Inference. Nikhil Gopal, Kostis Kaffes (Columbia). arXiv:2602.00328v1 [cs.LG], 30 Jan 2026

- **Version/text read:** arXiv 2602.00328v1 PDF, full text (raw_text_read=True)
- **Setup:** Azure NC80adis H100 v5: 80 vCPUs, 640 GB RAM, 2x H100 with 12 NVLink links, PCIe 5.0. Models Mixtral-8x7B-Instruct, Qwen2-MoE (Fig 6 says Qwen1.5-MoE-A2.7B), Phi-3.5-MoE, Phi-tiny-MoE. Built on MoE-Lightning CGOPipe: --max-new-tokens=32, micro-batch 324 tokens x 14 micro-batches = 4,536 tokens per batch, MTBench prompts, 5 trials. Experts forced offloaded: a fraction lives in host DRAM, the rest in peer-GPU HBM reached over NVLink; 50% offloaded for headline. Also a KV-cache offload path in vLLM.
- **Numbers (table/figure, baseline, ratio):**
  - Expert transfer latency, peer NVLink vs PCIe host fetch: 7.5x faster (Tiny Phi) to 9.5x (Mixtral 8x7B) | baseline: PCIe host fetch | ratio: 7.5-9.5x | where: Fig 3, Sec 2.2
  - Decode throughput, 50% experts offloaded: +48% to over +110% | baseline: CGOPipe with CPU-DRAM offload | ratio: 1.48x to >2.1x | where: Fig 5, Sec 4.5
  - Qwen MoE throughput vs offload fraction: ~975 tok/s flat from 0% to 100% offloaded (Harvest) | baseline: CPU offload falls to ~810 tok/s at 100% | ratio: 1.20x at full offload | where: Fig 6, Sec 4.5
  - Mixtral throughput vs offload fraction: ~740 tok/s (Harvest) | baseline: CPU offload <600 tok/s at 100% | ratio: >1.23x | where: Fig 6
  - End-to-end claim: 1.5-2.0x for Qwen2-MoE and Phi-3.5-MoE | baseline: CPU offload | ratio: 1.5-2.0x | where: Conclusion; abstract says "more than 2x"
- **Catalog / first-pass numbers checked:**
  - catalog: peer transfers replace slow PCIe host-DRAM fetches of offloaded experts; same concept on GPU interconnect -> yes. Peer = unused HBM of a second GPU in the same server over NVLink
  - first pass: transfer 7.5-9.5x; decode +48% to >110%; e2e 1.5-2.0x -> yes. as above; abstract (">2x") and body (1.5-2.0x; 48-110%) disagree
  - first pass: 2x H100 NVLink -> yes. yes
- **Transfers to our regime (CPU-only, 8-32 GB, NVMe, batch 1, 125B 512-expert top-10):** no. NVLink (hundreds of GB/s, same box) versus PCIe5 host fetch; large-batch throughput serving (4,536 tokens/batch), where expert fetch is amortised across many tokens; not batch 1 decode. A peer laptop over a LAN is 3-4 orders of magnitude slower than NVLink. The 1.2x at 100% offload for Qwen shows the real marginal gain of the faster tier is small once pipelining hides PCIe.
- **Quality loss:** None (exact weights).
- **Ablation:** Offload fraction sweep (Fig 6) isolates peer vs host; no cache-size or policy ablation; no prefetch ablation.
- **Limitations (stated):** Authors: single two-GPU NVLink system; not evaluated on NVSwitch fabrics, contention with model-parallel collectives, or tenants.
- **Contradicts our findings (pubtech/E.md):** No.

---

## P-0194: RMAI: Rethinking memory for AI (inference): In-kernel remote shared memory as a software alternative to CXL. Amir Noohi, Mostafa Derispour, Antonio Barbalace (Univ. of Edinburgh). EuroMLSys '25 (5th Workshop on ML and Systems, Rotterdam, 31 Mar 2025), ACM, pp. 122-131, published 1 Apr 2025, DOI 10.1145/3721146.3721954

- **Version/text read:** Edinburgh Research Explorer record (abstract, metadata, BibTeX) read raw. Full PDF (8.6 MB, CC BY-NC-SA) NOT readable: the Edinburgh file server returned a Cloudflare "Just a moment" challenge page and the ACM DL returned HTTP 403; I did not try to bypass the bot check. A web search found no other copy. (raw_text_read=False)
- **Setup:** UNREAD beyond the abstract: RDMA-based in-kernel remote shared memory framework with dynamic virtual memory regions to reduce page faults and user-kernel transitions; focuses on MoE; "initial evaluation". Models, hardware, link speed, disk type and batch size are not on the record page.
- **Numbers (table/figure, baseline, ratio):**
  - expert switching: "up to 10x faster expert switching" | baseline: "disk-based solutions" (disk type unknown) | ratio: up to 10x (abstract only) | where: abstract on the record page; table/figure not seen
- **Catalog / first-pass numbers checked:**
  - catalog: abstract claims up to 10x faster expert switching than disk -> yes (abstract). Abstract: "up to 10x faster expert switching and reduced memory management overhead ... compared to disk-based solutions"; also "comparable to CXL-like architectures"
  - catalog: RDMA remote shared memory for AI inference -> yes. abstract
  - catalog: "closest to the exact idea" -> unverifiable. The abstract frames it as a data-center CXL substitute; cannot confirm what is moved or at what bandwidth without the body
  - first pass: EuroMLSys '25, pp. 122-131, 1 Apr 2025 -> yes. confirmed
- **Transfers to our regime (CPU-only, 8-32 GB, NVMe, batch 1, 125B 512-expert top-10):** unclear. Cannot be determined without the evaluation section. Abstract positions it for data-center memory sharing, with an unspecified disk baseline; "10x vs disk" is also consistent with the generic gap between a fast network/RAM and a slow disk, which our own combo_model already covers (peer link rate as input).
- **Quality loss:** Not stated in the abstract (a memory system; presumably exact).
- **Ablation:** unknown
- **Limitations (stated):** Abstract calls the evaluation "initial".
- **Contradicts our findings (pubtech/E.md):** Unknown. Our modelled "10x" case is a peer at 2.5 GB/s versus a local NVMe at 3.11 GB/s: ~1x, so the paper's 10x is consistent only if its disk is far slower than a modern NVMe.
- **UNREAD:** Full text blocked (Cloudflare challenge on research.ed.ac.uk file URL, HTTP 403 on dl.acm.org). Only abstract and metadata were read. UNREAD for body, numbers and setup.

---

## P-0195: kjaiswal/llama-cpp-distributed-benchmarks, "Distributed LLM Inference: llama.cpp RPC across Metal + CUDA over 10GbE" (GitHub README, author handle kjaiswal; no date in the README text)

- **Version/text read:** raw README.md from raw.githubusercontent.com (full file) (raw_text_read=True)
- **Setup:** Mac Studio M2 Ultra 128 GB (Metal, llama-server host, local layers) + DGX Spark GB10 Blackwell 120 GB (CUDA, rpc-server), point-to-point 10 GbE, iperf3 9.41 Gbps. Qwen2.5-7B-Instruct Q4_K_M (4.4 GB) and Qwen2.5-72B-Instruct Q4_K_M (44.2 GB). Dense models, no MoE tested. llama-bench / benchmark.sh; prompt 512, gen 128 for llama-bench. A "projected" table (not measured) lists Qwen3-235B-A22B and MiniMax M2.5 at ~5-8 tok/s expected decode.
- **Numbers (table/figure, baseline, ratio):**
  - 7B decode: 52.7 tok/s (RPC Metal+CUDA) | baseline: local Metal only 91.8 | ratio: 0.57x | where: Results table
  - 7B prefill: 317.7 tok/s (RPC) | baseline: local 76.1 | ratio: 4.17x | where: Results table
  - 72B decode: 5.9 tok/s (RPC, 30.7 GB Metal + 13.8 GB CUDA) | baseline: local 11.1 | ratio: 0.53x (README: ~47% slower) | where: Results table
  - 72B prefill: 29.5 tok/s | baseline: local 28.2 | ratio: 1.05x | where: Results table
  - Per-token network round trip: ~0.17 ms on 10GbE | baseline: n/a | ratio: n/a | where: Analysis section
- **Catalog / first-pass numbers checked:**
  - catalog: RPC splits layers across machines and ships activations, not weights -> partly. README: layer split, activations per token; rpc-server needs no model file, so weights are uploaded once at load (from memory llama.cpp RPC does send tensors at load); per token only activations move
  - catalog: 10GbE recommended, 1GbE works but is slow -> yes. README Prerequisites: "10GbE recommended, 1GbE works but slower". The first pass said this is NOT on the page: that was wrong.
  - first pass: 72B decode 11.1 local vs 5.9 RPC; 7B prefill 76->318, decode 91.8->52.7 -> yes. as above
  - first pass: no MoE tested -> yes. projection table only, labelled projected
- **Transfers to our regime (CPU-only, 8-32 GB, NVMe, batch 1, 125B 512-expert top-10):** partly. Confirms that an extra network hop per token costs decode speed (batch 1 decode halves) and that RPC value is capacity not speed; consistent with our 1.0x / sub-1x modelled results for a peer fetch. It does not test expert streaming, MoE, CPU-only machines or disk.
- **Quality loss:** None (same weights).
- **Ablation:** local vs RPC only; no layer-split ratio sweep.
- **Limitations (stated):** Author: RPC is for capacity, not speed; decode is latency bound.
- **Contradicts our findings (pubtech/E.md):** No; supports our finding that a LAN hop does not help decode when the model fits locally.

---

## P-0196: (a) FlashMoE: Reducing SSD I/O Bottlenecks via ML-Based Cache Replacement for Mixture-of-Experts Inference on Edge Devices. Byeongju Kim, Jungwan Lee, Donghyeon Han, Hoi-Jun Yoo, Sangyeob Kim (KAIST, Chung-Ang, Yonsei). arXiv:2601.17063v2, 5 Oct 2026. (b) MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for MoE. Wenfeng Wang, Jiacheng Liu, Xiaofeng Hou, Xinfeng Xia, Peng Tang, Mingxuan Zhang, Chao Li, Minyi Guo. arXiv:2511.14102v1, 18 Nov 2025; v3 (22 Oct 2026) retitled "Specter: Fast MoE Inference via Quantization-Enabled Multi-Step Speculative Offloading"

- **Version/text read:** arXiv 2601.17063v2 (full PDF text); arXiv 2511.14102v1 and v3 (both PDFs, full text) (raw_text_read=True)
- **Setup:** (a) OLMoE-1B-7B (64 experts/layer) and Qwen3-30B-A3B (128 experts/layer, top-8), Ryzen 5 9600X, RTX 5070 Ti 16 GB, 4x16 GB DDR5-6000 with all but ~1 GB locked, SK hynix P51 PCIe5 NVMe 7.4 GB/s, Ubuntu 24.04, PyTorch 2.7; experts stored as per-expert .pt files; decode-time expert cache lives in VRAM with 12/16/20/24 experts per layer (3.4-14.4 GB), cache policy trained offline on TriviaQA routing traces (A100 batch 32 to collect); expert SSD load ~3 ms vs FFN 158 us; expert loading >70% of decode time (Fig 5). Baselines: Fiddler, DAOP, llama.cpp, Transformers; policies LRU/LFU/ARC/LeCaR. Batch 1 decode. (b) v1: Phi-3.5-MoE, Qwen1.5-MoE-A2.7B, DeepSeek-V2-Lite on A100-40GB, 24-core Xeon, 256 GB RAM, PCIe 4.0 x16 (~32 GB/s); target FP16, INT4 GPTQ draft, experts in host DRAM (not SSD); baselines Transformers, Mixtral-Offloading-SC and -SM; low and high GPU-memory settings (Fig 13). v3: primary baseline SpecMoEOff, batch size 1 and B=32 results.
- **Numbers (table/figure, baseline, ratio):**
  - FlashMoE cache hit rate vs LRU / LFU (OLMoE-1B-7B): +21% vs LRU, +51% vs LFU (translates to 22% / 35% fewer I/O ops) | baseline: LRU, LFU; also up to +28% vs ARC and +21% vs LeCaR | ratio: n/a | where: Sec 4.3, Fig 6, p.6
  - FlashMoE decode throughput vs LRU cache: +22% (OLMoE) and +7% (Qwen3-30B-A3B) | baseline: LRU policy | ratio: 1.22x / 1.07x | where: Sec 4.4, Fig 7b,e
  - FlashMoE "up to 2.6x speedup compared to existing MoE inference systems": 2.6x | baseline: unspecified; body shows Fig 7c,f bars (Fiddler, DAOP, llama.cpp) with no numbers in the text | ratio: cannot be recomputed (bar chart only) | where: Abstract only
  - FlashMoE load / prefill: load 4x faster than llama.cpp, 6.8x faster than Fiddler/DAOP; prefill (incl. expert load) 2.5x vs llama.cpp, 4.1x vs Fiddler/DAOP | baseline: llama.cpp, Fiddler, DAOP | ratio: 2.5x-6.8x (prefill/load, not decode) | where: Sec 4.2, Fig 7a,d
  - FlashMoE motivation: LRU vs Belady: LRU 73% hit vs Belady 86% (1.9x more I/O) | baseline: Belady optimal | ratio: 0.85 | where: Sec 2 (Fig 2)
  - MoE-SpeQ v1 abstract: "at most 2.34x speedup over the state-of-the-art offloading framework" (Phi-MoE) | baseline: unnamed in abstract | ratio: 2.34x | where: Abstract, Sec 1 contributions
  - MoE-SpeQ v1 body, high-memory setting: Phi-MoE 163.1 ms/token | baseline: Mixtral-Offloading-SC 536.7; -SM 351.8; Transformers 960.5 | ratio: 3.3x / 2.16x / 5.9x | where: Sec 4.2 text, Fig 13
  - MoE-SpeQ v1 Qwen2-MoE / DeepSeek-V2-Lite: 74.1 ms/token | baseline: Mixtral-Offloading-SC (2.5x); Transformers 3.5x; DeepSeek 4.8x vs Transformers | ratio: 2.5x / 3.5x / 4.8x | where: Sec 4.2
  - Specter v3: avg 1.50x (up to 1.88x) TPOT speedup, single request | baseline: SpecMoEOff | ratio: 1.50x | where: Abstract v3, Table 3
  - Specter v3 prediction-error sensitivity: prefetch accuracy 1.00 -> 0.41: TPOT 63.0 -> 84.2 ms | baseline: perfect prediction | ratio: 1.34x slowdown | where: Table 6a
  - Specter v3 ablation (cumulative): without PIO 83.5, +no SIC 94.4, +no SEE 130.4 ms/token | baseline: full 69.2 | ratio: 1.21x / 1.36x / 1.88x | where: Table 5b
- **Catalog / first-pass numbers checked:**
  - catalog: FlashMoE single-machine SSD expert offload reports 2.6x over existing systems -> partly. The 2.6x is only in the abstract, baseline unspecified, not reproduced in any body number I can read; measured body decode gain over the best policy (LRU) is 7-22%
  - catalog: MoE-SpeQ reports 2.34x from speculative decoding plus prefetch -> partly. v1 abstract says "at most 2.34x"; v1 body contradicts it with 3.3x over Mixtral-Offloading-SC (Phi-MoE high memory); v3 (Specter) reports 1.50x avg / 1.88x max vs SpecMoEOff. The 2.34x is therefore a stale v1 figure
  - catalog: "This is the local-disk lever" (applies to both) -> partly. FlashMoE reads experts from NVMe; MoE-SpeQ/Specter offload to HOST DRAM over PCIe (no SSD)
  - first pass: FlashMoE up to 2.6x (+7-22% vs LRU); MoE-SpeQ v1 up to 2.34x; Specter v3 avg 1.50x -> yes. as above
  - first pass: cache hit "up to 51% above LRU/LFU" -> partly. 51% is vs LFU only; vs LRU it is 21% (the abstract itself lumps them)
  - first pass: Specter v3 "ASPLOS '27" -> absent. Not stated in the paper text I read; only a repo name "specter_asplos" appears
  - first pass: FlashMoE hardware "64 GB DDR5 (about 1 GB left free)" -> yes. 4x16 GB DDR5-6000 with ~1 GB left free
- **Transfers to our regime (CPU-only, 8-32 GB, NVMe, batch 1, 125B 512-expert top-10):** partly. FlashMoE is the closest single-machine NVMe expert-streaming paper: same expert-granular files, batch 1, ~3 ms per expert read, RAM squeezed to 1 GB free. But the expert cache is in 16 GB of VRAM (9-38% of experts per layer), compute on GPU, and Qwen3-30B has 128 experts (not 512). Its relevant message is that cache policy beats LRU by 7-22% decode (hit rate 73% LRU vs 86% Belady) at equal cache. MoE-SpeQ/Specter needs a GPU, a quantised draft model resident on device, speculative multi-token verification (batch>1 verification) and host DRAM; none is available CPU-only in 8-32 GB.
- **Quality loss:** FlashMoE: none (exact). MoE-SpeQ/Specter: lossless w.r.t. the target (INT4 draft, FP16 target verifies); speed falls with draft accuracy.
- **Ablation:** FlashMoE: policy comparison LRU/LFU/ARC/LeCaR/ML at several cache sizes (Fig 6, 7b,e) isolates the replacement policy; no prefetch ablation. Specter: Table 5b (PIO/SIC/SEE) and Table 6 isolate prefetch accuracy.
- **Limitations (stated):** FlashMoE: policy trained offline on TriviaQA traces; no stated limitations section; routing-trace generalisation to other domains not tested. Specter: depends on draft quality; A100 only.
- **Contradicts our findings (pubtech/E.md):** Mild: FlashMoE shows LRU is 13 points below Belady in hit rate (73% vs 86%), so our "LRU already hits 0.567 at 8 GB" best path is not at the policy ceiling; a learned recency+frequency policy might add 7-22% decode speed. Their gain is relative to LRU, so it does not contradict our numbers but is a missed lever in our simulations. Also their SSD read 3 ms/expert vs our 2.7 MB reads at 3.1 GB/s is consistent.

---

## P-0198: Training-Free Activation Sparsity in Large Language Models (TEAL). James Liu, Pragaash Ponnusamy, Tianle Cai, Han Guo, Yoon Kim, Ben Athiwaratkun (MIT, Together AI, Princeton). "Published as a conference paper at ICLR 2025", arXiv:2408.14690v3, 25 Feb 2025

- **Version/text read:** arXiv 2408.14690v3 PDF, full text (raw_text_read=True)
- **Setup:** Dense models only: Llama-2 7B/13B/70B, Llama-3 8B/70B, Mistral-7B. Magnitude thresholding of the inputs to every linear layer (calibrated per layer, greedy per-layer sparsity). Speed: GPT-Fast with CUDA graphs + torch.compile, FP16, batch 1, ~5 input tokens, up to 200 output tokens, A6000 (300 W) and A100 (500 W); Llama-3-70B uses tensor parallel (TP2 A100, TP4 A6000). Custom sparse GEMV kernel. LM head not sparsified. Quantisation compatibility only shown as perplexity (Fig 4: 8-bit RTN, 4-bit AWQ, 2/3-bit QuIP#); quantised-sparse speed explicitly left to future work. No MoE experiments.
- **Numbers (table/figure, baseline, ratio):**
  - Decode speed A6000, Llama-2-7B: 77.30 tok/s (1.53x) at 40%; 89.91 (1.78x) at 50% | baseline: dense 50.54 | ratio: 1.53x / 1.78x | where: Table 3, p.7
  - Decode speed A6000, Llama-2-13B: 40.20 (1.52x) at 40%; 47.60 (1.80x) at 50% | baseline: dense 26.43 | ratio: 1.52x / 1.80x | where: Table 3
  - Decode speed A6000, Llama-3-8B: 64.15 (1.42x) at 40%; 73.94 (1.63x) at 50% | baseline: dense 45.32 | ratio: 1.42x / 1.63x | where: Table 3
  - Decode speed A6000, Llama-3-70B (TP4): 20.86 (1.31x) / 23.77 (1.49x) | baseline: dense 15.93 | ratio: 1.31x / 1.49x | where: Table 3
  - Decode speed A100, Llama-2-13B: 81.90 (1.34x) / 88.38 (1.45x) | baseline: dense 61.01 | ratio: 1.34x / 1.45x | where: Table 3
  - Kernel at 0% sparsity on A100: 0.91-0.93x | baseline: torch.compile dense | ratio: n/a | where: Table 3 (paper says kernel slower than baseline at 0% on A100)
  - Perplexity Llama-3-8B (WikiText): 5.87 dense; 5.94 / 6.21 / 6.67 / 9.06 at 25/40/50/65% | baseline: dense 5.87 | ratio: +0.34 at 40%, +0.80 at 50%, +3.19 at 65% | where: Table 1
  - Llama-2-7B perplexity: 5.07 dense; 5.09 / 5.22 / 5.43 / 6.62 | baseline: dense | ratio: +0.15 at 40%, +0.36 at 50% | where: Table 1
  - Downstream average of six tasks, Llama-3-8B: 68.07 dense; 67.73 / 66.21 / 63.42 / 52.59 | baseline: dense | ratio: -1.86 at 40%, -4.65 at 50% | where: Table 2
  - Downstream, Llama-3-70B: 80.41 dense; 79.29 at 40%, 78.26 at 50% | baseline: dense | ratio: -1.12 / -2.15 | where: Table 2
  - Baseline CATS 40% (perplexity Llama-3-8B): 7.6e4 (degenerate) | baseline: TEAL 40% 6.21 | ratio: n/a | where: Table 1
- **Catalog / first-pass numbers checked:**
  - catalog: 40-50% model-wide sparsity with minimal degradation -> yes. abstract; but "minimal" is perplexity +0.34 (Llama-3-8B, 40%) and -1.9 / -4.7 downstream points at 40 / 50% for the smallest Llama-3
  - catalog: 1.53x / 1.8x decode speedup at 40% / 50% on dense models -> yes (best cases). 1.53x is Llama-2-7B A6000 at 40%; 1.80x is Llama-2-13B A6000 at 50%; Llama-3-8B 1.42x / 1.63x; A100 1.25-1.45x
  - catalog: "ICLR 2025" -> yes. printed in the PDF header "Published as a conference paper at ICLR 2025" (first pass: not named on page, based on the digest)
  - first pass: Llama-3-8B ppl 5.87/6.21/6.67/9.06, downstream 68.07/66.21/63.42 -> yes. matches Tables 1-2
  - first pass: Table 3 A6000 7B 1.53/1.78 and 13B 1.52/1.80 -> yes. matches
- **Transfers to our regime (CPU-only, 8-32 GB, NVMe, batch 1, 125B 512-expert top-10):** partly. Real batch-1 speedups, but from a GPU memory-bound dense FP16 GEMV with CUDA graphs: skipping weight columns for zeroed activations. In our regime (4.3-bit experts streamed from NVMe at 3 GB/s) the saving only materialises if unneeded weight columns are not read from disk, which needs column-contiguous (strided) layout and small reads; the gate/up inputs are the same for every row so the sparsity pattern is per token, and the union over a draft/prefetch window fills in. Speed with quantisation is NOT measured. Batch scaling is poor (paper: Wdown sparsity at 50% model-wide drops from ~60% at batch 1 to ~38% at batch 4). MoE: the paper only speculates that MoE low-batch is "less difficult"; no results.
- **Quality loss:** Table 1-2: Llama-3-8B +0.34 ppl and -1.86 downstream points at 40%; +0.80 ppl and -4.65 points at 50%; larger models (70B) lose 1.1-2.2 points. 65% degrades significantly.
- **Ablation:** Yes: per-layer greedy allocation vs uniform, input sparsity vs output sparsity for Wup (Sec 5.4.1), prefill sparsification (Sec 5.4.3), batch size (Sec 5.4.4). No ablation on memory-transfer (disk) vs compute.
- **Limitations (stated):** Authors: does not scale to higher batch sizes; LM head not sparsified; sparse+quantised kernels left to future work; MoE note is speculation.
- **Contradicts our findings (pubtech/E.md):** No. Their 1.5-1.8x ceiling equals our closed-form gate-first byte gain at d=0.5 (1.50x) and TEAL is the dense case without our expert-union and 57% dense-compute limits. Our reading in E.md (family 12) is consistent; note the 1.53x/1.8x are the best two entries of Table 3, not a typical value (1.25-1.63x).

---

## Per-family sections

### Family 10 (second LAN machine as expert compute server; rows P-0186..P-0190)

Further prior art found and read (same depth where text was reachable):
- OrderMoE, arXiv 2607.17154 (Yuan, Li, Chen, Xu, Guo; IEEE Trans.-style, 2026), read in full text: 16 physical edge servers (GPUs 20-110 TFLOPS, 12-48 GB each, 1-10 Gbps links, 0.2-20 ms latency), Switch-Base-8E / Qwen-MoE-A2.7B / Mixtral-8x7B. On Mixtral: average latency 187 ms, 51.0% below EdgeShard and 40.1% below MoE^2; 1.5 GB cross-server traffic per 1000 tokens; 1380 tok/s; 83% of expert calls local (52% exact, 31% substitute). Quality: WikiText-103 ppl +0.3, SQuAD F1 -0.6, GSM8K -0.7. It substitutes a similar local expert for a remote one (lossy). Serving many requests per second (40 req/s) on GPU servers.
- SlimCaching, arXiv 2507.06567 (Chen, Chen, Huang; v3 1 Mar 2026): expert caching across edge servers as submodular knapsack, simulation only (abstract and intro read).
- Prism/DanceMoE (this batch, P-0188), OD-MoE (P-0186), ExFlow (P-0189) as above. Also named by search: vExpert (NeurIPS 2026 poster, host-DRAM expert storage for distributed MoE, abstract only), EdgeShard and MoE^2 (baselines in OrderMoE; not read), Pre-compiled pipeline shards for Intel AI PCs arXiv 2608.19147 (Aug 2026; I read the abstract and intro: pipeline-parallel dense Llama 3.1 8B/70B across AI PCs, 1.79x aggregate throughput for two concurrent users on two nodes; not MoE, not expert streaming).
- Memory check (from memory, not read now): DeepSpeed-MoE / Tutel / Lina / Janus (datacenter expert parallelism), Petals, distributed-llama (tensor-parallel on Raspberry Pi/PC clusters), llama.cpp RPC.

What is new in OUR variant: none of the papers I could read reports a two-machine expert split for one user at batch 1 where the second machine only adds RAM-cache capacity and compute while the first keeps the dense part; the closest (OD-MoE) distributes expert loading over GPU nodes with experts in DRAM and needs a full shadow model; OrderMoE/Prism/ExFlow all target multi-request GPU serving. The measured numbers in pubtech/E.md (straggler 1.25-1.28 for random halves, dense-part Amdahl 1.19x) have no published counterpart I found. The idea itself (shard experts over edge servers) is known (OD-MoE, SlimCaching, Prism, OrderMoE).

### Family 11 (layer-split expert store, peer RAM instead of NVMe; rows P-0192..P-0196)

Further prior art read:
- SGLang SSD Expert Pack (LMSYS blog, 29 Aug 2026, WiCi AI/SGLang), read raw: consumer box (Intel Ultra5 230F, 32 GB RAM, 2 TB NVMe, RTX 5090 32 GB), batch 1 sequential requests, 200-token completions, experts packed contiguous per (layer, expert), O_DIRECT, pinned staging, GPU expert cache. DeepSeek-V4-Flash (155 GB MXFP4): decode 1.992 / 1.846 tok/s (Alpaca / MMLU) vs 0.288 / 0.282 for Ollama baseline = 6.9x / 6.5x; prefill 2.28x / 3.39x; VRAM cache hit 54.2% / 46.4%, 1.66 / 2.04 GB SSD traffic per token. Kimi-K3 (985 GB Q2_K, top-16, 92 layers) vs llama.cpp: decode 3.30x / 3.52x, hit rate 17.0% / 22.7% with a 5 GiB cache, 22-28 GB SSD traffic per token. Exact (no expert substitution). This is the closest published regime to ours (batch 1, 32 GB RAM, NVMe) but with an RTX 5090 and VRAM cache; absolute decode is ~2 tok/s.
- FlashMoE (P-0196) closest academic equivalent.
- Speculating Experts, arXiv 2603.19289 (Madan et al., Mar 2026), read: predicts next-layer experts from quasi-hidden states and prefetches; 5-14% TPOT reduction (Qwen3-30B-A3B 9-14%, also GPT-OSS-120B, Qwen3-235B-A22B on GH200); experts in CPU DRAM, GPU compute.
- DALI, arXiv 2602.03495 (local-PC offloading): skimmed only; reports average decode speedups 7.62x / 3.80x / 2.45x / 2.00x over llama.cpp / KTransformers / MoE-Lightning / HybriMoE (abstract-level, GPU+DRAM). Context-aware MoE on CXL GPU-NDP, arXiv 2512.04476: moves activations to near-data compute instead of weights; abstract-level only. SSD offloading energy study, arXiv 2508.06978: SSD offload costs up to ~12x energy per token vs HBM (4.9x for DeepSeek-R1 in a table), prefetch hides latency but not energy; read the abstract and key numbers.
- Memory check (from memory, not re-read): MoE-Infinity, ProMoE, HOBBIT, Fiddler, PowerInfer-2 (phone, flash), EdgeMoE, Pre-gated MoE, ExpertFlow, Mixtral-Offloading.

What is new in OUR variant: using a peer machine\'s RAM as the cold expert tier for a laptop (instead of NVMe) was not found in any paper I could read; RMAI (EuroMLSys 2025) is the only one that appears to do remote-memory expert switching but its body is UNREAD, so novelty versus RMAI is undecided. Published numbers say a network tier only wins when compared against a much slower tier (RMAI "10x vs disk", Harvest 7.5-9.5x vs PCIe); against a 3 GB/s NVMe the peer link needs >3 GB/s to tie (our model: ~1.0x).

### Family 12 (activation/block sparsity inside experts; row P-0198)

Further prior art: TEAL itself cites CATS, Deja Vu, ReLU-fication; search surfaced TurboSparse (arXiv 2406.05955, dReLU, MoE experts, claims 2-5x decode but needs continued training; abstract-level) and found no training-free TEAL-style method evaluated on MoE experts with CPU/SSD streaming. Memory check (from memory): PowerInfer/PowerInfer-2, LLM in a Flash (windowing, row-column bundling on flash), CATS, R-Sparse, WINA.

What is new in OUR variant: TEAL contains no MoE or quantised-speed evidence, so applying it to streamed expert weights is open; its own batch result (sparsity falls with batch/union) and the closed-form cap in E.md make >1.5x unlikely.


## Summary of corrections to the first pass (P09)
1. P-0195: the first pass said "10GbE recommended, 1GbE works" is NOT on the page. It IS (README Prerequisites). Catalog line accurate on that point.
2. P-0192: first pass said the catalog Qwen3 235B numbers were not found. They exist in secondary coverage (31.9 and 32.5 raw-confirmed in technews.tw; 20.4, 15.2, 19.5, 21.1, 27.8 from search summaries). The opened Hardware Corner page itself still has graphs only.
3. P-0196: "cache hit up to 51% above LRU/LFU": 51% is vs LFU, vs LRU 21%. The 2.6x appears only in the abstract with no baseline; MoE-SpeQ v1 abstract (2.34x) conflicts with its own body (3.3x), and MoE-SpeQ/Specter keeps experts in host DRAM, not SSD. "ASPLOS \'27" for Specter is not in the paper text.
4. P-0198: venue ICLR 2025 is printed in the paper header (first pass said unnamed). 1.53x/1.8x are the best two cells of Table 3; Llama-3-8B gives 1.42x/1.63x; A100 1.25-1.45x.
5. P-0188: "latency" is Poisson-arrival serve latency with queueing, not decode latency; v1 of the arXiv page names the system DanceMoE (search snippet), v4 Prism.
6. P-0194: the first pass labelled it "abstract only" correctly; I confirm the body is not obtainable (Cloudflare challenge, ACM 403).
7. P-0186 catalog line is truncated at "about 7" in the CSV (204-char cell), so that number cannot be checked.
