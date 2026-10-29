---
id: R16
type: run
status: done
links: [R14, M15, P01]
date: 2026-10-29
---
# Screen of the top 22 brainstorm idea families

22 screeners (14 speed families + 8 problem families) recomputed each multiplier from our traces against OUR BEST PATH (explicit large reads + LRU), searched prior art (4+ searches each), and proposed the cheapest test; a ranker merged them (Playbook/idea_screen_raw.json). Verdicts: 10 drop, 12 combine-with-other, 0 pursue-now. Novelty: 21 known, 1 partly known, 0 not found. Headline: none qualifies as a >=3x, never-done-before result. Recomputed: (1) expert-major k=8 speculative union = KNOWN (MoE-Spec 2602.16052, DraftExpert, SpecMoEOff, SP-MoE, EcoSpec) and 0.4-1.0x vs our best path: simulated bytes per accepted token are 1.3-2.7x WORSE than LRU at 2.1-4 footprints because the cache already captures the same locality and rejected drafts still load experts; (2) GPU tier (dense + hot experts in VRAM): 2-2.6x at 24 GB VRAM (4.57 -> 10-12 tok/s), only 1.0-1.1x at 6-8 GB, prior art KTransformers, Fiddler, HOBBIT, llama.cpp --n-cpu-moe; (3) top-k 10->6 or 4: compute ceiling 4.57 -> ~5.4-6 tok/s, 1.3x at 8 GB, accuracy-gated, prior art NAEE, LExI, REAP, llama.cpp expert_used_count override; (4) mixed 2.6-3 bpw experts: 1.8x serial at 8 GB, ~1.0x at 32 GB, realistic files 42-48 GB not 34; (5) install-time per-expert repack + explicit reads = our base path (4.1x serial / 7x overlap vs mmap at 8 GB; the 7x is a disk-rate ratio and overlap is unimplemented); (6) cache-aware routing: 1.3x serial at 8 GB, prior art Skliar et al 2412.00099 (2x over LRU on a phone), ReMoE, Cache-Aware Joint Router Adaptation; (7) expert-major prefill: 1.0x vs our best path (batched mul_mat_id already groups per expert), 3.5-6x TTFT vs mmap; (8) co-activation repack: mbolt ships it, ~1.1x, order learned elsewhere gives run length 1.29-1.39 vs 1.7-1.8 in-sample. Real remaining wins are engineering: build the explicit-read repack path and make the cheap measurements (2.7 MB read rate, small-batch GFLOP/s, OLMoE perplexity at lower k).
- see [[R14 Investigation of the 12 pipeline outputs]]
- see [[P01 mmap read amplification]]
