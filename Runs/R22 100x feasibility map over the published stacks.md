---
id: R22
type: run
status: done
links: [P03, R17, R19, M19]
date: 2026-10-29
---
# 100x feasibility map over the published stacks

tools/hundredx_map.py (tinyram-llm, committed): 100x means F/B >= 100, so each stack gets B* = F/100, the baseline speed below which it is 100x faster. Published stacks added in the order the literature introduced them (explicit reads+LRU = Flash-MoE/BigMoeOnEdge/SSD-LLaMA; +overlap = ProMoE/ExpertFlow; +cache-aware routing = Skliar; +exact speculation = MoE-Spec/EcoSpec; +3-bit = HOBBIT/MC-MoE; +lookup kernels = T-MAC; +top-6/top-4 = NAEE/LExI; +layer skip = LayerSkip; +GPU tier = KTransformers/Fiddler). 8 GB, likely parameters: F = 2.49 (base path, quality-safe), 3.13 (+overlap), 3.59 (+cache routing), 2.90 (+spec), 4.09 (+3-bit), 4.71 (+lookup), 6.47 (+top-6), 7.76 (+top-4), 10.6 (+skip 25%), 81.8 (+12 GB GPU), 100.2 (+24 GB GPU). B* = F/100. Against the modelled plain-mmap baseline (0.53 tok/s at 8 GB) 100x needs F >= 52.6 and only the GPU tiers reach it (hardware, not a small PC). Against a thrashing baseline of 0.10 tok/s (F >= 10) only layer-skip and the GPU tiers qualify; against 0.05 (F >= 5) five stacks qualify, the first at risk 7; against swap-class baselines 0.02 / 0.01 even the quality-safe base path is 124x / 249x. So whether 100x holds is decided by the true naive baseline at the real model-to-RAM ratio (~8.6x), which had never been measured; measuring how the baseline collapses with the ratio is in progress (VPS, strict cgroup cap).
- see [[P03 The 100x claim is unproven]]
- see [[R17 Loop-until-dry brainstorm rounds 3-6 with an exclusion list]]
- see [[R19 Evaluation of all 217 investigation items]]
